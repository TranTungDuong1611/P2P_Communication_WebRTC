"""
WebRTC Audio Client
Establishes P2P audio connection with another peer via signaling server
"""
import asyncio
import json
import logging
import sys
from typing import Optional

import websockets
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole
from av import AudioFrame
import pyaudio
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioLoopbackTrack(MediaStreamTrack):
    """
    Audio track that captures from microphone using MediaPlayer
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        # Use system default audio input
        # On macOS: uses default microphone (empty string or :0)
        # On Linux: uses ALSA default
        # On Windows: uses DirectShow default
        try:
            if sys.platform == "darwin":
                # macOS: Use ":0" to select default audio input device
                self.player = MediaPlayer(
                    ":0",
                    format="avfoundation",
                    options={"audio_buffer_size": "50"}
                )
            else:
                self.player = MediaPlayer(
                    "default",
                    format="alsa",
                    options={"sample_rate": "48000", "channels": "1"}
                )
            logger.info("Audio capture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio capture: {e}")
            logger.info("Will create silent audio track instead")
            self.player = None

    async def recv(self):
        """Receive audio frame from microphone"""
        if self.player and self.player.audio:
            frame = await self.player.audio.recv()
            return frame
        else:
            # Generate silent audio frame as fallback
            await asyncio.sleep(0.02)  # 20ms frame
            frame = AudioFrame(format="s16", layout="mono", samples=960)
            for p in frame.planes:
                p.update(bytes(p.buffer_size))
            frame.pts = getattr(self, "_timestamp", 0)
            frame.sample_rate = 48000
            frame.time_base = "1/48000"
            self._timestamp = getattr(self, "_timestamp", 0) + 960
            return frame


class WebRTCClient:
    """WebRTC client for peer-to-peer audio communication"""

    def __init__(self, signaling_server_url: str):
        self.signaling_server_url = signaling_server_url
        self.pc: Optional[RTCPeerConnection] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.role: Optional[str] = None
        self.recorder: Optional[MediaRecorder] = None

        # Buffer for ICE candidates that arrive before remote description is set
        self.ice_candidate_buffer = []

    async def connect(self):
        """Connect to signaling server and establish WebRTC connection"""
        logger.info(f"Connecting to signaling server: {self.signaling_server_url}")

        async with websockets.connect(self.signaling_server_url) as websocket:
            self.ws = websocket

            # Wait for initial message (waiting or matched)
            initial_msg = await websocket.recv()
            initial_data = json.loads(initial_msg)

            if initial_data["type"] == "waiting":
                logger.info("Waiting for another peer to connect...")
                # Wait for matched message
                matched_msg = await websocket.recv()
                matched_data = json.loads(matched_msg)
                self.role = matched_data["role"]
            else:
                self.role = initial_data["role"]

            logger.info(f"Matched! Role: {self.role}")

            # Create peer connection
            await self.create_peer_connection()

            # If caller, create and send offer
            if self.role == "caller":
                await self.create_offer()

            # Message handling loop
            try:
                async for message in websocket:
                    await self.handle_signaling_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection to signaling server closed")
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                await self.cleanup()

    async def create_peer_connection(self):
        """Initialize RTCPeerConnection with ICE servers"""
        # Using Google's public STUN server
        configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )

        self.pc = RTCPeerConnection(configuration=configuration)

        # Add audio track
        audio_track = AudioLoopbackTrack()
        self.pc.addTrack(audio_track)
        logger.info("Added local audio track")

        # Set up event handlers
        @self.pc.on("icecandidate")
        async def on_icecandidate(event):
            """Send ICE candidates to peer via signaling server"""
            if event.candidate:
                logger.info(f"New ICE candidate: {event.candidate.candidate[:50]}...")
                await self.send_signaling_message({
                    "type": "ice-candidate",
                    "data": {
                        "candidate": event.candidate.candidate,
                        "sdpMid": event.candidate.sdpMid,
                        "sdpMLineIndex": event.candidate.sdpMLineIndex,
                    }
                })

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                logger.error("ICE connection failed")
                await self.cleanup()

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                logger.info(" P2P audio connection established!")

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received {track.kind} track from peer")
            if track.kind == "audio":
                logger.info("🔊 Playing incoming audio through speakers...")

                # Initialize PyAudio for speaker output
                audio = pyaudio.PyAudio()
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=48000,
                    output=True,
                    frames_per_buffer=960
                )

                # Also record to file for verification
                self.recorder = MediaRecorder("received_audio.wav")
                self.recorder.addTrack(track)
                await self.recorder.start()
                logger.info("📝 Recording incoming audio to received_audio.wav")

                # Play audio through speakers
                async def play_audio():
                    try:
                        while True:
                            frame = await track.recv()
                            # Convert audio frame to bytes and play
                            audio_data = frame.to_ndarray()

                            # Ensure correct format for PyAudio
                            if audio_data.dtype != np.int16:
                                # Normalize float to int16 range
                                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_data = audio_data.astype(np.int16)

                            # Flatten if multi-channel
                            audio_data = audio_data.flatten()

                            # Play through speakers
                            stream.write(audio_data.tobytes())

                    except Exception as e:
                        logger.info(f"Audio playback ended: {e}")
                    finally:
                        stream.stop_stream()
                        stream.close()
                        audio.terminate()

                asyncio.create_task(play_audio())

        logger.info("Peer connection created")

    async def create_offer(self):
        """Create and send WebRTC offer"""
        logger.info("Creating offer...")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        await self.send_signaling_message({
            "type": "offer",
            "data": {
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type,
            }
        })
        logger.info("Offer sent")

    async def handle_signaling_message(self, message: str):
        """Process incoming signaling messages"""
        data = json.loads(message)
        msg_type = data["type"]

        logger.info(f"Received signaling message: {msg_type}")

        if msg_type == "offer":
            # Received offer, create answer
            offer = RTCSessionDescription(
                sdp=data["data"]["sdp"],
                type=data["data"]["type"]
            )
            await self.pc.setRemoteDescription(offer)
            logger.info("Remote offer set")

            # Process buffered ICE candidates
            await self.process_ice_candidate_buffer()

            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            await self.send_signaling_message({
                "type": "answer",
                "data": {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                }
            })
            logger.info("Answer sent")

        elif msg_type == "answer":
            # Received answer
            answer = RTCSessionDescription(
                sdp=data["data"]["sdp"],
                type=data["data"]["type"]
            )
            await self.pc.setRemoteDescription(answer)
            logger.info("Remote answer set")

            # Process buffered ICE candidates
            await self.process_ice_candidate_buffer()

        elif msg_type == "ice-candidate":
            # Received ICE candidate
            candidate_data = data["data"]

            if self.pc.remoteDescription:
                # Remote description is set, can add candidate immediately
                candidate = RTCIceCandidate(
                    candidate=candidate_data["candidate"],
                    sdpMid=candidate_data["sdpMid"],
                    sdpMLineIndex=candidate_data["sdpMLineIndex"],
                )
                await self.pc.addIceCandidate(candidate)
                logger.info("ICE candidate added")
            else:
                # Buffer candidate until remote description is set
                self.ice_candidate_buffer.append(candidate_data)
                logger.info("ICE candidate buffered (waiting for remote description)")

        elif msg_type == "peer-disconnected":
            logger.info("Peer disconnected")
            await self.cleanup()

    async def process_ice_candidate_buffer(self):
        """Process buffered ICE candidates after remote description is set"""
        if self.ice_candidate_buffer:
            logger.info(f"Processing {len(self.ice_candidate_buffer)} buffered ICE candidates")
            for candidate_data in self.ice_candidate_buffer:
                candidate = RTCIceCandidate(
                    candidate=candidate_data["candidate"],
                    sdpMid=candidate_data["sdpMid"],
                    sdpMLineIndex=candidate_data["sdpMLineIndex"],
                )
                await self.pc.addIceCandidate(candidate)
            self.ice_candidate_buffer.clear()

    async def send_signaling_message(self, message: dict):
        """Send message to peer via signaling server"""
        if self.ws:
            await self.ws.send(json.dumps(message))

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")

        if self.recorder:
            await self.recorder.stop()

        if self.pc:
            await self.pc.close()

        logger.info("Cleanup complete")


async def main():
    """Main entry point"""
    signaling_url = "ws://localhost:8000/ws"

    if len(sys.argv) > 1:
        signaling_url = sys.argv[1]

    logger.info("=" * 60)
    logger.info("WebRTC P2P Audio Client")
    logger.info("=" * 60)
    logger.info(f"Signaling server: {signaling_url}")
    logger.info("Press Ctrl+C to quit")
    logger.info("=" * 60)

    client = WebRTCClient(signaling_url)

    try:
        await client.connect()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
