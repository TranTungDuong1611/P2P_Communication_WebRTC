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
        self.player = None
        self._initialize_audio_capture()

    def _initialize_audio_capture(self):
        """Initialize audio capture with platform-specific settings"""
        logger.info(f"Initializing audio capture for platform: {sys.platform}")

        if sys.platform == "darwin":
            # macOS: Try different audio input methods
            audio_sources = [
                (":0", {"sample_rate": "48000"}),
                (":default", {"sample_rate": "48000"}),
                ("0", {"sample_rate": "48000"}),
            ]

            for source, options in audio_sources:
                try:
                    logger.info(f"Trying macOS audio source: '{source}' with options {options}")
                    self.player = MediaPlayer(
                        source,
                        format="avfoundation",
                        options=options
                    )
                    logger.info(f"✅ Successfully initialized audio capture with source: '{source}'")
                    return
                except Exception as e:
                    logger.warning(f"Failed with source '{source}': {e}")

        elif sys.platform.startswith("linux"):
            # Linux: Try different ALSA/PulseAudio configurations
            audio_sources = [
                ("default", "alsa", {"sample_rate": "48000", "channels": "1"}),
                ("pulse", "pulse", {}),
                ("hw:0", "alsa", {"sample_rate": "48000", "channels": "1"}),
            ]

            for source, fmt, options in audio_sources:
                try:
                    logger.info(f"Trying Linux audio source: '{source}' format: '{fmt}' with options {options}")
                    self.player = MediaPlayer(
                        source,
                        format=fmt,
                        options=options
                    )
                    logger.info(f"✅ Successfully initialized audio capture with source: '{source}'")
                    return
                except Exception as e:
                    logger.warning(f"Failed with source '{source}': {e}")

        else:
            # Windows
            try:
                self.player = MediaPlayer(
                    "audio=Microphone",
                    format="dshow"
                )
                logger.info("✅ Successfully initialized audio capture (Windows)")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Windows audio: {e}")

        logger.error("❌ FAILED to initialize audio capture - will send SILENT audio!")
        logger.error("Please check:")
        logger.error("  1. Microphone is connected and working")
        logger.error("  2. App has microphone permissions")
        if sys.platform == "darwin":
            logger.error("  3. macOS: System Preferences → Security & Privacy → Privacy → Microphone")
        elif sys.platform.startswith("linux"):
            logger.error("  3. Linux: Run 'arecord -l' to list audio devices")

    async def recv(self):
        """Receive audio frame from microphone"""
        if self.player and self.player.audio:
            try:
                frame = await self.player.audio.recv()
                # Verify we're actually getting audio data
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 1
                    logger.info(f"🎤 Successfully receiving audio from microphone")

                return frame
            except Exception as e:
                logger.error(f"Error receiving audio frame: {e}")
                self.player = None

        # Fallback: Generate silent audio frame
        if not hasattr(self, '_warned_silent'):
            logger.warning("⚠️ Sending SILENT audio - microphone not working!")
            self._warned_silent = True

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
                logger.info("P2P audio connection established!")

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received {track.kind} track from peer")
            if track.kind == "audio":
                logger.info("🔊 Setting up audio playback...")

                # Record to file for verification
                self.recorder = MediaRecorder("received_audio.wav")
                self.recorder.addTrack(track)
                await self.recorder.start()
                logger.info("📝 Recording to received_audio.wav")

                # Initialize PyAudio for speaker output
                audio = pyaudio.PyAudio()

                # Log available audio devices for debugging
                logger.info(f"Available audio devices: {audio.get_device_count()}")

                stream = None
                try:
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=48000,
                        output=True,
                        frames_per_buffer=1024
                    )
                    logger.info("✅ Audio output stream opened successfully")
                except Exception as e:
                    logger.error(f"Failed to open audio stream: {e}")
                    # Try with default settings
                    try:
                        stream = audio.open(
                            format=pyaudio.paInt16,
                            channels=2,  # Stereo
                            rate=44100,  # Standard CD quality
                            output=True
                        )
                        logger.info("✅ Opened audio stream with default settings (44.1kHz stereo)")
                    except Exception as e2:
                        logger.error(f"Failed to open audio stream with defaults: {e2}")

                # Play audio through speakers
                async def play_audio():
                    frame_count = 0
                    try:
                        while True:
                            frame = await track.recv()
                            frame_count += 1

                            if frame_count % 100 == 0:
                                logger.info(f"Received {frame_count} audio frames")

                            if stream is None:
                                continue

                            # Get audio data as numpy array
                            audio_data = frame.to_ndarray()

                            # Log first frame info for debugging
                            if frame_count == 1:
                                logger.info(f"Frame info - dtype: {audio_data.dtype}, shape: {audio_data.shape}, "
                                          f"rate: {frame.sample_rate}, samples: {frame.samples}")

                            # Convert to int16 format for PyAudio
                            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                                # Audio is in float format [-1.0, 1.0], convert to int16
                                audio_data = np.clip(audio_data, -1.0, 1.0)
                                audio_data = (audio_data * 32767).astype(np.int16)
                            elif audio_data.dtype != np.int16:
                                audio_data = audio_data.astype(np.int16)

                            # Handle channel layout
                            if len(audio_data.shape) == 2:
                                # Multi-channel, convert to mono or handle appropriately
                                if stream._channels == 1:
                                    audio_data = audio_data.mean(axis=1).astype(np.int16)
                                elif stream._channels == 2 and audio_data.shape[1] == 1:
                                    # Mono to stereo
                                    audio_data = np.repeat(audio_data, 2, axis=1)
                            else:
                                # Single channel
                                if stream._channels == 2:
                                    # Duplicate mono to stereo
                                    audio_data = np.column_stack([audio_data, audio_data])

                            # Ensure contiguous array
                            audio_data = np.ascontiguousarray(audio_data)

                            # Write to audio output
                            try:
                                stream.write(audio_data.tobytes(), exception_on_underflow=False)
                            except Exception as e:
                                if frame_count % 50 == 0:
                                    logger.warning(f"Audio write error: {e}")

                    except Exception as e:
                        logger.info(f"Audio playback ended: {e}")
                    finally:
                        logger.info(f"Total frames received: {frame_count}")
                        if stream:
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
