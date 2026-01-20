"""
WebRTC Signaling Server
Handles WebSocket connections and relays signaling messages between peers
"""
import asyncio
import json
import logging
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebRTC Signaling Server")

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections and peer matching"""

    def __init__(self):
        self.waiting_peer: Optional[WebSocket] = None
        self.active_connections: Dict[WebSocket, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> Optional[WebSocket]:
        """
        Connect a new peer. If there's a waiting peer, match them.
        Returns the matched peer or None if this peer should wait.
        """
        await websocket.accept()

        if self.waiting_peer is None:
            # No peer waiting, this one waits
            self.waiting_peer = websocket
            logger.info(f"Peer {id(websocket)} is waiting for a match")
            return None
        else:
            # Match with waiting peer
            peer = self.waiting_peer
            self.waiting_peer = None

            # Store the connection pair
            self.active_connections[websocket] = peer
            self.active_connections[peer] = websocket

            logger.info(f"Matched peer {id(websocket)} with {id(peer)}")
            return peer

    def disconnect(self, websocket: WebSocket):
        """Remove a peer and notify their partner if connected"""
        if websocket == self.waiting_peer:
            self.waiting_peer = None
            logger.info(f"Waiting peer {id(websocket)} disconnected")
        elif websocket in self.active_connections:
            peer = self.active_connections[websocket]
            del self.active_connections[websocket]
            del self.active_connections[peer]
            logger.info(f"Peer {id(websocket)} disconnected from {id(peer)}")
            return peer
        return None

    def get_peer(self, websocket: WebSocket) -> Optional[WebSocket]:
        """Get the connected peer for a given websocket"""
        return self.active_connections.get(websocket)


manager = ConnectionManager()


@app.get("/")
async def root():
    return {
        "service": "WebRTC Signaling Server",
        "status": "running",
        "endpoints": {
            "websocket": "/ws"
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for signaling.

    Message format:
    {
        "type": "offer" | "answer" | "ice-candidate",
        "data": <SDP or ICE candidate object>
    }
    """
    peer = await manager.connect(websocket)

    try:
        # If matched immediately, notify both peers to start negotiation
        if peer is not None:
            await websocket.send_json({
                "type": "matched",
                "role": "caller"  # This peer initiates the call
            })
            await peer.send_json({
                "type": "matched",
                "role": "callee"  # This peer waits for offer
            })
        else:
            # Peer is waiting
            await websocket.send_json({
                "type": "waiting",
                "message": "Waiting for another peer to connect..."
            })

        # Message relay loop
        while True:
            # Receive message from this peer
            data = await websocket.receive_text()
            message = json.loads(data)

            logger.info(f"Received {message.get('type')} from peer {id(websocket)}")

            # Get the connected peer
            peer = manager.get_peer(websocket)
            if peer is None:
                logger.warning(f"No peer connected for {id(websocket)}")
                continue

            # Relay message to the other peer
            await peer.send_text(data)
            logger.info(f"Relayed {message.get('type')} to peer {id(peer)}")

    except WebSocketDisconnect:
        logger.info(f"Peer {id(websocket)} disconnected")
        peer = manager.disconnect(websocket)

        # Notify the other peer about disconnection
        if peer is not None:
            try:
                await peer.send_json({
                    "type": "peer-disconnected",
                    "message": "Your peer has disconnected"
                })
            except Exception as e:
                logger.error(f"Failed to notify peer about disconnection: {e}")

    except Exception as e:
        logger.error(f"Error in websocket handler: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
