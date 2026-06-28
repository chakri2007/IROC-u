import asyncio
import json
import math
import threading
import time
import urllib.request

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field


# ── Jetson IP — only needed for video. Change to your Jetson's IP ────────────
#JETSON_IP          = "10.202.3.2"   # <-- set this
#JETSON_IP          = "192.168.0.144"   # <-- set thi
JETSON_IP          = "192.168.0.29"   # <-- set this
JETSON_STREAM_PORT = 8765
JETSON_STREAM_URL  = f"http://{JETSON_IP}:{JETSON_STREAM_PORT}/"

# ── Push rate for the WebSocket telemetry channel ────────────────────────────
WS_PUSH_HZ = 10.0

# ── Try importing rclpy (ROS2) ───────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState
    from std_msgs.msg import String
    from mavros_msgs.msg import State, ExtendedState
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] rclpy not found – running with mock data")

# ── Shared state ─────────────────────────────────────────────────────────────
state = {
    "connected":   False,          # ROS bridge is receiving
    "fcuConnected":False,          # MAVROS ↔ flight controller link (from /mavros/state)
    "armed":       False,
    "mode":        "",             # flight mode, e.g. OFFBOARD / AUTO.LAND
    "landedState": "UNDEFINED",    # ON_GROUND / IN_AIR / TAKEOFF / LANDING
    "position":    {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "velocity":    {"horizontal": 0.0, "climbRate": 0.0},
    "battery":     {"percentage": 0.0, "voltage": 0.0},
    "altitude":    0.0,
    "detections":  [],
}
state_lock = threading.Lock()


def telemetry_snapshot() -> Dict[str, Any]:
    """Thread-safe copy of state without the (potentially large) detections list."""
    with state_lock:
        return {k: v for k, v in state.items() if k != "detections"}


def detections_snapshot() -> List[Dict[str, Any]]:
    with state_lock:
        return list(state["detections"])


# ── Telemetry schema (typed → shows up in /docs) ─────────────────────────────
class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Orientation(BaseModel):
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class Velocity(BaseModel):
    horizontal: float = 0.0
    climbRate: float = 0.0


class Battery(BaseModel):
    percentage: float = 0.0
    voltage: float = 0.0


class Telemetry(BaseModel):
    connected: bool = False
    fcuConnected: bool = False
    armed: bool = False
    mode: str = ""
    landedState: str = "UNDEFINED"
    position: Position = Field(default_factory=Position)
    orientation: Orientation = Field(default_factory=Orientation)
    velocity: Velocity = Field(default_factory=Velocity)
    battery: Battery = Field(default_factory=Battery)
    altitude: float = 0.0


class Status(BaseModel):
    connected: bool = False


# ── Quaternion → Euler (degrees) ─────────────────────────────────────────────
def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp) * (180 / math.pi)

    sinp = 2 * (w * y - z * x)
    pitch = (math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1
             else math.asin(sinp)) * (180 / math.pi)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp) * (180 / math.pi)

    return roll, pitch, yaw


# ── ROS2 Node ────────────────────────────────────────────────────────────────
if ROS_AVAILABLE:
    class DroneNode(Node):
        def __init__(self):
            super().__init__('gcs_flask_bridge')

            self.create_subscription(Odometry,     '/mavros/local_position/odom', self.on_odom,           qos_profile_sensor_data)
            self.create_subscription(BatteryState, '/mavros/battery',             self.on_battery,        qos_profile_sensor_data)
            self.create_subscription(String,       '/seed_detections',            self.on_detection,      10)
            self.create_subscription(State,        '/mavros/state',               self.on_state,          10)
            self.create_subscription(ExtendedState,'/mavros/extended_state',      self.on_extended_state, 10)

            with state_lock:
                state["connected"] = True
            self.get_logger().info('[GCS] Subscribed to ROS2 topics')

        def on_odom(self, msg):
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            vel = msg.twist.twist.linear
            roll, pitch, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)
            with state_lock:
                state["position"]    = {"x": pos.x, "y": pos.y, "z": pos.z}
                state["orientation"] = {"roll": roll, "pitch": pitch, "yaw": yaw}
                state["velocity"]    = {"horizontal": math.sqrt(vel.x**2 + vel.y**2), "climbRate": vel.z}
                state["altitude"]    = pos.z

        def on_battery(self, msg):
            with state_lock:
                state["battery"] = {
                    "percentage": (msg.percentage or 0.0) * 100,
                    "voltage":    msg.voltage or 0.0,
                }

        def on_detection(self, msg):
            try:
                detection = json.loads(msg.data)
                with state_lock:
                    state["detections"] = ([detection] + state["detections"])[:50]
            except Exception as e:
                self.get_logger().error(f'Detection parse error: {e}')

        def on_state(self, msg):
            with state_lock:
                state["fcuConnected"] = msg.connected
                state["armed"]        = msg.armed
                state["mode"]         = msg.mode

        # mavros_msgs/ExtendedState.landed_state enum → human-readable label
        _LANDED = {0: "UNDEFINED", 1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}

        def on_extended_state(self, msg):
            with state_lock:
                state["landedState"] = self._LANDED.get(msg.landed_state, "UNDEFINED")


# ── ROS2 spin thread ─────────────────────────────────────────────────────────
def ros_thread():
    if not ROS_AVAILABLE:
        t = 0
        while True:
            t += 0.05
            alt_mock = abs(math.sin(t*0.3))*10
            with state_lock:
                state["connected"]   = True
                state["fcuConnected"]= True
                state["armed"]       = True
                state["mode"]        = "OFFBOARD"
                state["landedState"] = "IN_AIR" if alt_mock > 0.3 else "ON_GROUND"
                state["position"]    = {"x": math.sin(t)*5, "y": math.cos(t)*5, "z": alt_mock}
                state["orientation"] = {"roll": math.sin(t*0.7)*15, "pitch": math.cos(t*0.5)*10, "yaw": (t*20)%360}
                state["velocity"]    = {"horizontal": abs(math.sin(t))*8, "climbRate": math.cos(t*0.4)*2}
                state["battery"]     = {"percentage": max(0, 80-t*0.1), "voltage": max(10, 14.8-t*0.01)}
                state["altitude"]    = alt_mock
            time.sleep(0.05)
        return

    rclpy.init()
    node = DroneNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"[ROS2] Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        with state_lock:
            state["connected"] = False


# ── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the ROS2 (or mock) bridge thread on startup
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    print("[FastAPI] Bridge thread started — docs at http://localhost:5000/docs")
    yield
    # (daemon thread exits with the process; nothing to clean up explicitly)


app = FastAPI(
    title="ANVESHAN GCS Backend",
    description="Ground-control bridge: ROS2 telemetry + detections + video proxy for the ASCEND drone.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints (unchanged contract — existing frontend keeps working) ─────
@app.get("/api/telemetry", response_model=Telemetry)
def telemetry():
    return telemetry_snapshot()


@app.get("/api/detections", response_model=List[Dict[str, Any]])
def detections():
    return detections_snapshot()


@app.get("/api/status", response_model=Status)
def status():
    with state_lock:
        return {"connected": state["connected"]}


# ── WebSocket: push telemetry + detections (no polling needed) ────────────────
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "telemetry":  telemetry_snapshot(),
                "detections": detections_snapshot(),
            })
            await asyncio.sleep(1.0 / WS_PUSH_HZ)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")


# ── Video proxy: relay the Jetson MJPEG stream to the browser ────────────────
@app.get("/video_feed")
def video_feed():
    """
    Proxy the MJPEG stream from the Jetson streamer directly to the browser.
    The browser keeps this connection open and receives frames continuously.
    """
    try:
        upstream = urllib.request.urlopen(JETSON_STREAM_URL, timeout=10)
    except Exception as e:
        print(f"[Video] Could not open Jetson stream: {e}")
        # Empty 503 → the <img> fires onerror and the UI shows "VIDEO: LOST"
        return Response(status_code=503)

    # Forward the upstream Content-Type so the MJPEG boundary always matches.
    content_type = upstream.headers.get(
        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
    )

    def generate():
        try:
            while True:
                chunk = upstream.read(4096)
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            print(f"[Video] Stream error: {e}")
        finally:
            upstream.close()

    return StreamingResponse(generate(), media_type=content_type)


if __name__ == "__main__":
    import uvicorn
    print("[FastAPI] Starting on http://localhost:5000  (interactive docs: /docs)")
    uvicorn.run(app, host="0.0.0.0", port=5000)
