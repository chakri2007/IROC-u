import asyncio
import base64
import collections
import json
import math
import os
import queue
import threading
import time
import urllib.request

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (FastAPI, WebSocket, WebSocketDisconnect, File, Form,
                     UploadFile, Depends, Header, HTTPException)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

import config_store


# ── Jetson IP — only needed for video. Change to your Jetson's IP ────────────
#JETSON_IP          = "10.202.3.2"   # <-- set this
#JETSON_IP          = "192.168.0.144"   # <-- set thi
JETSON_IP          = "192.168.0.29"   # <-- set this
JETSON_STREAM_PORT = 8765
JETSON_STREAM_URL  = f"http://{JETSON_IP}:{JETSON_STREAM_PORT}/"

# ── Push rate for the WebSocket telemetry channel ────────────────────────────
WS_PUSH_HZ = 10.0

# ── On-dock auto-indexing defaults (overridable via POST /api/trigger_indexing)
DEFAULT_ROSBAG_PATH   = "/home/jetson/mission_rosbag"
DEFAULT_OUTPUT_DB_PATH = "/home/jetson/embeddings.pt"

# ── GCS command interface ────────────────────────────────────────────────────
# Closed vocabulary published on /gcs/command (std_msgs/String). The flight team
# owns a subscriber that maps each string to a flight action (see HANDOFF.md);
# we ONLY publish the token — we never interpret it. Nothing outside this set is
# ever sent. See command_precondition() for the server-side gate.
COMMAND_VOCAB = ("START", "ABORT", "HOLD", "RTL", "ABORT_DOCK", "RECALL")

# ── Operator auth (GUI↔backend command path only; NOT the docking-station link).
#    Set GCS_COMMAND_TOKEN in the backend's environment to require the header
#    `X-GCS-Token` on mutating endpoints. Empty/unset → auth disabled (dev/mock).
GCS_TOKEN = os.environ.get("GCS_COMMAND_TOKEN", "").strip()

# ── Try importing rclpy (ROS2) ───────────────────────────────────────────────
try:
    import cv2
    import numpy as np

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                           ReliabilityPolicy, DurabilityPolicy)
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import String
    from mavros_msgs.msg import State, ExtendedState
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] rclpy not found – running with mock data")

# ── Semantic-retrieval interfaces (separate package; may not be built) ───────
try:
    from semantic_retrieval_interfaces.msg import SemanticMatch
    from semantic_retrieval_interfaces.srv import TriggerIndexing, GetHdFrame
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[WARN] semantic_retrieval_interfaces not found – semantic features limited")


# ── Log ring buffer (IN = ROS→backend, OUT = GUI→ROS, SYS = internal) ────────
MAX_LOGS   = 500
log_buffer = collections.deque(maxlen=MAX_LOGS)
log_lock   = threading.Lock()


def push_log(direction: str, topic: str, summary: str, data: Optional[dict] = None):
    entry = {
        "ts":        time.time(),
        "ts_human":  datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
        "direction": direction,
        "topic":     topic,
        "summary":   summary,
        "data":      data or {},
    }
    with log_lock:
        log_buffer.append(entry)


# ── Docking terminal stream (TCP traffic + live PID lines from dock_manager) ──
# Separate from the structured log above: this is the raw green-terminal feed the
# GUI's docking panel polls via /api/dock/log?since=<seq>.
MAX_DOCK_LOG  = 600
dock_log_buf  = collections.deque(maxlen=MAX_DOCK_LOG)
dock_log_lock = threading.Lock()
_dock_log_seq = 0


def push_dock_log(text: str):
    global _dock_log_seq
    with dock_log_lock:
        _dock_log_seq += 1
        dock_log_buf.append({
            "seq":  _dock_log_seq,
            "ts":   datetime.utcnow().strftime("%H:%M:%S"),
            "text": text,
        })


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

    # --- VSLAM ---
    "vslam_status": "OFFLINE",
    "vslam_pose":   {"x": 0.0, "y": 0.0, "z": 0.0},
    "vslam_path":   [],            # capped-200 list of {x,y}

    # --- semantic retrieval ---
    "semantic_status":  "IDLE",
    "semantic_results": {},        # seed_name → result dict
    "seeds":            [],        # registered seed names (publish order / mock)
    "seed_images":      {},        # seed_name → {"media_type": str, "data": bytes} (GUI thumbnails)

    # --- docking / autonomous charging ---
    "dock_status": "UNKNOWN",
    "dock_state":  "UNKNOWN",      # dock_manager state machine (BOOT/SETTLING/DOCKING/…)

    # --- indexing job state ---
    "indexing": {"in_progress": False, "frames_indexed": 0, "message": "", "submitted": False},

    # --- GCS command interface ---
    "last_command":     "",        # last command WE published (echo for the UI)
    "last_command_ts":  0.0,
    "last_command_ack": "",        # last ack from /gcs/command_ack (team-published)
    "last_command_ack_ts": 0.0,
}
state_lock = threading.Lock()

# Keys that make up the flight-telemetry snapshot (everything Telemetry models).
TELEMETRY_KEYS = (
    "connected", "fcuConnected", "armed", "mode", "landedState",
    "position", "orientation", "velocity", "battery", "altitude",
)

# ── Executor-task queue: rclpy runs in the daemon thread, so any publish /
#    service call from a request handler must be MARSHALLED onto that thread.
#    A node timer drains this queue, executing each task on the executor. ──────
_exec_queue: "queue.Queue" = queue.Queue()
_node = None  # type: Optional[Any]


def _enqueue_task(fn):
    """Schedule `fn(node)` to run on the rclpy executor thread."""
    _exec_queue.put(fn)


def telemetry_snapshot() -> Dict[str, Any]:
    """Thread-safe copy of the flight-telemetry slice of state."""
    with state_lock:
        return {k: state[k] for k in TELEMETRY_KEYS}


def full_snapshot() -> Dict[str, Any]:
    """Thread-safe snapshot for the WebSocket push (telemetry + semantic/VSLAM)."""
    with state_lock:
        return {
            "telemetry":        {k: state[k] for k in TELEMETRY_KEYS},
            "semantic_results": dict(state["semantic_results"]),
            "semantic_status":  state["semantic_status"],
            "vslam_status":     state["vslam_status"],
            "vslam_pose":       dict(state["vslam_pose"]),
            "vslam_path":       list(state["vslam_path"]),
            "dock_status":      state["dock_status"],
            "dock_state":       state["dock_state"],
            "last_command":     state["last_command"],
            "last_command_ts":  state["last_command_ts"],
            "last_command_ack": state["last_command_ack"],
            "last_command_ack_ts": state["last_command_ack_ts"],
        }


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


class TriggerIndexingRequest(BaseModel):
    rosbag_path: str = DEFAULT_ROSBAG_PATH
    output_db_path: str = DEFAULT_OUTPUT_DB_PATH
    force_reindex: bool = False


class CommandRequest(BaseModel):
    command: str = Field(..., description="One of: " + ", ".join(COMMAND_VOCAB))


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
        # mavros_msgs/ExtendedState.landed_state enum → human-readable label
        _LANDED = {0: "UNDEFINED", 1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}

        def __init__(self):
            super().__init__('gcs_fastapi_bridge')
            self.bridge = CvBridge()
            self._dock_triggered = False

            # ── MAVROS (unchanged) ─────────────────────────────────────────
            self.create_subscription(Odometry,     '/mavros/local_position/odom', self.on_odom,           qos_profile_sensor_data)
            self.create_subscription(BatteryState, '/mavros/battery',             self.on_battery,        qos_profile_sensor_data)
            self.create_subscription(State,        '/mavros/state',               self.on_state,          10)
            self.create_subscription(ExtendedState,'/mavros/extended_state',      self.on_extended_state, 10)

            # ── VSLAM ──────────────────────────────────────────────────────
            self.create_subscription(String,   '/visual_slam/status',            self.on_vslam_status, 10)
            self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_vslam_odom,   qos_profile_sensor_data)

            # ── Semantic retrieval status ──────────────────────────────────
            self.create_subscription(String, '/semantic_retrieval/status', self.on_semantic_status, 10)

            # ── Docking / autonomous charging ──────────────────────────────
            self.create_subscription(String, '/drone/status', self.on_drone_status, 10)

            # ── Publisher: runtime seeds (header.frame_id = seed name) ─────
            self.seed_pub = self.create_publisher(RosImage, '/semantic_retrieval/add_seed', 10)
            self.remove_pub = self.create_publisher(String, '/semantic_retrieval/remove_seed', 10)

            # ── Docking orchestrator (dock_manager) ────────────────────────
            self.dock_cmd_pub = self.create_publisher(String, '/dock/command', 10)
            dock_state_qos = QoSProfile(depth=1,
                                        reliability=ReliabilityPolicy.RELIABLE,
                                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, '/dock/state', self.on_dock_state, dock_state_qos)
            self.create_subscription(String, '/dock/log',   self.on_dock_log,   50)

            # ── GCS command publisher + ack subscriber ─────────────────────
            # RELIABLE so a safety command can't be silently dropped; VOLATILE
            # (NOT latched) so a node restart never re-fires the last command.
            cmd_qos = QoSProfile(depth=10,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.VOLATILE)
            self.cmd_pub = self.create_publisher(String, '/gcs/command', cmd_qos)
            # Optional team-side ack (see HANDOFF.md). If they publish it, the GUI
            # shows "drone acknowledged X" instead of just "sent".
            self.create_subscription(String, '/gcs/command_ack', self.on_command_ack, cmd_qos)

            # ── Semantic results + service clients (separate package) ──────
            self.trigger_indexing_client = None
            self.get_hd_frame_client     = None
            if SEMANTIC_AVAILABLE:
                self.create_subscription(SemanticMatch, '/semantic_retrieval/results', self.on_semantic_result, 10)
                self.trigger_indexing_client = self.create_client(TriggerIndexing, '/semantic_retrieval/trigger_indexing')
                self.get_hd_frame_client     = self.create_client(GetHdFrame,      '/semantic_retrieval/get_hd_frame')
                self.get_logger().info('[GCS] Subscribed to SemanticMatch results + service clients ready')
            else:
                self.get_logger().warn('semantic_retrieval_interfaces not built — semantic results disabled')

            # ── Drain queued publishes / service calls on the executor ─────
            self.create_timer(0.02, self._drain_tasks)

            with state_lock:
                state["connected"] = True
            push_log("SYS", "node", "DroneNode initialised — all subscriptions active")
            self.get_logger().info('[GCS] Subscribed to ROS2 topics')

        # ── Marshalled task drain (runs on executor thread) ────────────────
        def _drain_tasks(self):
            while True:
                try:
                    fn = _exec_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn(self)
                except Exception as e:
                    push_log("SYS", "exec", f"task error: {e}")
                    self.get_logger().error(f'exec task error: {e}')

        # ── MAVROS ─────────────────────────────────────────────────────────
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
            # Guard NaN: `or` lets NaN through (NaN is truthy), so test isfinite.
            pct = msg.percentage
            if not math.isfinite(pct):
                pct = 0.0
            volt = msg.voltage
            if not math.isfinite(volt):
                volt = 0.0
            with state_lock:
                state["battery"] = {"percentage": pct * 100, "voltage": volt}

        def on_state(self, msg):
            with state_lock:
                state["fcuConnected"] = msg.connected
                state["armed"]        = msg.armed
                state["mode"]         = msg.mode

        def on_extended_state(self, msg):
            with state_lock:
                state["landedState"] = self._LANDED.get(msg.landed_state, "UNDEFINED")

        # ── VSLAM ──────────────────────────────────────────────────────────
        def on_vslam_status(self, msg):
            with state_lock:
                state["vslam_status"] = msg.data
            push_log("IN", "/visual_slam/status", msg.data)

        def on_vslam_odom(self, msg):
            pos = msg.pose.pose.position
            with state_lock:
                state["vslam_pose"] = {"x": pos.x, "y": pos.y, "z": pos.z}
                trail = state["vslam_path"]
                trail.append({"x": pos.x, "y": pos.y})
                if len(trail) > 200:
                    del trail[:-200]

        # ── Semantic status ────────────────────────────────────────────────
        def on_semantic_status(self, msg):
            with state_lock:
                state["semantic_status"] = msg.data
                # Status often reports indexing progress.
                if "INDEX" in msg.data.upper():
                    state["indexing"]["message"] = msg.data
            push_log("IN", "/semantic_retrieval/status", msg.data)

        # ── Semantic results ───────────────────────────────────────────────
        def on_semantic_result(self, msg):
            seed = msg.seed_name

            # Unpack stride-3 positions + stride-4 orientations with length guards.
            positions  = list(msg.positions_xyz)
            pose_valid = list(msg.pose_valid)
            poses = []
            for i in range(msg.match_count):
                p_off = i * 3
                valid = bool(pose_valid[i]) if i < len(pose_valid) else False
                if len(positions) >= p_off + 3:
                    x, y, z = positions[p_off], positions[p_off + 1], positions[p_off + 2]
                else:
                    x, y, z = 0.0, 0.0, 0.0
                    valid = False
                poses.append({"valid": valid, "x": x, "y": y, "z": z})

            result = {
                "seed_name":            seed,
                "has_match":            msg.has_match,
                "match_count":          msg.match_count,
                "frame_indices":        list(msg.frame_indices),
                "timestamps_sec":       list(msg.timestamps_sec),
                "similarity_scores":    list(msg.similarity_scores),
                "similarity_threshold": msg.similarity_threshold,
                "poses":                poses,
                # NOTE: msg.hd_frames intentionally NOT read (always empty now;
                # fetched on demand via GetHdFrame service).
                "updated_at":           time.time(),
            }

            with state_lock:
                state["semantic_results"][seed] = result
                if seed not in state["seeds"]:
                    state["seeds"].append(seed)

            scores = list(msg.similarity_scores)
            summary = (
                f"seed={seed} has_match={msg.has_match} matches={msg.match_count} "
                f"best={scores[0]:.3f}"
                if msg.has_match and scores
                else f"seed={seed} NO_MATCH (thr {msg.similarity_threshold:.2f})"
            )
            push_log("IN", "/semantic_retrieval/results", summary,
                     {"seed": seed, "has_match": msg.has_match, "match_count": msg.match_count})

        # ── GCS command ack (team-published; optional) ─────────────────────
        def on_command_ack(self, msg):
            with state_lock:
                state["last_command_ack"]    = msg.data
                state["last_command_ack_ts"] = time.time()
            push_log("IN", "/gcs/command_ack", msg.data)

        # ── dock_manager state + terminal stream ───────────────────────────
        def on_dock_state(self, msg):
            with state_lock:
                state["dock_state"] = msg.data
            # don't echo to dock_log — dock_manager already streams a [STATE] line

        def on_dock_log(self, msg):
            push_dock_log(msg.data)

        # ── Docking → fire on-dock auto-trigger ONCE ───────────────────────
        def on_drone_status(self, msg):
            data = msg.data
            with state_lock:
                state["dock_status"] = data
            push_log("IN", "/drone/status", data)

            if data.strip().upper() in ("LANDED", "DOCKED") and not self._dock_triggered:
                self._dock_triggered = True
                push_log("SYS", "dock", "Docked → auto-triggering indexing (parallel with charging)")
                _trigger_indexing(DEFAULT_ROSBAG_PATH, DEFAULT_OUTPUT_DB_PATH, False)


# ── Service-call helpers (marshalled onto the executor thread) ───────────────
def _trigger_indexing(rosbag_path: str, output_db_path: str, force_reindex: bool) -> Dict[str, Any]:
    """Fire-and-forget TriggerIndexing. Updates state["indexing"] from the response."""
    with state_lock:
        state["indexing"] = {
            "in_progress":    True,
            "frames_indexed": 0,
            "message":        "submitted",
            "submitted":      True,
        }

    if not (ROS_AVAILABLE and SEMANTIC_AVAILABLE and _node is not None
            and _node.trigger_indexing_client is not None):
        # Mock / no-ROS: pretend it queued.
        with state_lock:
            state["indexing"]["message"] = "queued (mock — no ROS service)"
        push_log("OUT", "/semantic_retrieval/trigger_indexing",
                 f"(mock) indexing requested: {rosbag_path}")
        return {"submitted": True, "queued": True, "mock": True}

    def task(node):
        client = node.trigger_indexing_client
        if not client.service_is_ready():
            with state_lock:
                state["indexing"]["in_progress"] = False
                state["indexing"]["message"]     = "service unavailable"
            push_log("SYS", "trigger_indexing", "service not ready")
            return
        req = TriggerIndexing.Request()
        req.rosbag_path   = rosbag_path
        req.output_db_path = output_db_path
        req.force_reindex = force_reindex
        future = client.call_async(req)

        def done(fut):
            try:
                res = fut.result()
                with state_lock:
                    state["indexing"] = {
                        "in_progress":    False,
                        "frames_indexed": res.frames_indexed,
                        "message":        res.message,
                        "submitted":      True,
                        "success":        res.success,
                        "indexing_time_sec": res.indexing_time_sec,
                    }
                push_log("IN", "/semantic_retrieval/trigger_indexing",
                         f"done success={res.success} frames={res.frames_indexed}")
            except Exception as e:
                with state_lock:
                    state["indexing"]["in_progress"] = False
                    state["indexing"]["message"]     = f"error: {e}"
                push_log("SYS", "trigger_indexing", f"call failed: {e}")

        future.add_done_callback(done)

    _enqueue_task(task)
    push_log("OUT", "/semantic_retrieval/trigger_indexing",
             f"indexing requested: {rosbag_path} → {output_db_path} (force={force_reindex})")
    return {"submitted": True, "queued": True}


def _get_hd_frame(frame_index: int, timeout: float = 5.0):
    """Blocking (in a threadpool) GetHdFrame call, marshalled to the executor."""
    if not (ROS_AVAILABLE and SEMANTIC_AVAILABLE and _node is not None
            and _node.get_hd_frame_client is not None):
        return {"error": "GetHdFrame service not available"}

    holder: Dict[str, Any] = {}
    ev = threading.Event()

    def task(node):
        client = node.get_hd_frame_client
        if not client.service_is_ready():
            holder["error"] = "service unavailable"
            ev.set()
            return
        req = GetHdFrame.Request()
        req.frame_index = int(frame_index)
        future = client.call_async(req)

        def done(fut):
            try:
                holder["result"] = fut.result()
            except Exception as e:
                holder["error"] = str(e)
            ev.set()

        future.add_done_callback(done)

    _enqueue_task(task)
    if not ev.wait(timeout):
        return {"error": "timeout waiting for GetHdFrame"}
    return holder


def _publish_seed(ros_img):
    """Thread-safe seed publish: marshal onto the executor thread."""
    _enqueue_task(lambda node: node.seed_pub.publish(ros_img))


def _publish_remove_seed(name: str):
    """Thread-safe seed removal: marshal onto the executor thread."""
    msg = String()
    msg.data = name
    _enqueue_task(lambda node: node.remove_pub.publish(msg))


def _publish_dock_command(cmd: str):
    """Thread-safe docking command publish: marshal onto the executor thread."""
    msg = String()
    msg.data = cmd
    _enqueue_task(lambda node: node.dock_cmd_pub.publish(msg))


# ── GCS command: server-side precondition gate + marshalled publish ──────────
def command_precondition(cmd: str):
    """Defense-in-depth gate (the authoritative gating lives on the drone). The
    UI disables buttons too, but this stops a buggy/hostile LAN client from
    publishing e.g. START mid-flight. Returns (ok: bool, reason: str)."""
    with state_lock:
        connected = state["connected"]
        armed     = state["armed"]
        landed    = state["landedState"]
    in_air = armed or landed == "IN_AIR"

    if not connected:
        return False, "ROS link down"
    if cmd == "START":
        if in_air:
            return False, "already armed / in-air"
        return True, ""
    # ABORT / HOLD / RTL / ABORT_DOCK / RECALL are all in-air overrides.
    if not in_air:
        return False, "drone not armed / in-air"
    return True, ""


def _publish_command(cmd: str) -> Dict[str, Any]:
    """Publish a vocab command to /gcs/command (marshalled to the executor)."""
    with state_lock:
        state["last_command"]    = cmd
        state["last_command_ts"] = time.time()

    if not (ROS_AVAILABLE and _node is not None):
        push_log("OUT", "/gcs/command", f"(mock) {cmd}")
        return {"published": True, "command": cmd, "mock": True}

    def task(node):
        m = String()
        m.data = cmd
        node.cmd_pub.publish(m)

    _enqueue_task(task)
    push_log("OUT", "/gcs/command", cmd)
    return {"published": True, "command": cmd}


# ── Operator-token dependency (mutating endpoints only) ──────────────────────
def require_token(x_gcs_token: str = Header(default="")):
    """Enforce X-GCS-Token when GCS_COMMAND_TOKEN is set in the environment.
    Unset → open (dev/mock). Applies only to the GUI↔backend command path."""
    if GCS_TOKEN and x_gcs_token != GCS_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-GCS-Token")
    return True


# ── ROS2 spin thread ─────────────────────────────────────────────────────────
def ros_thread():
    global _node
    if not ROS_AVAILABLE:
        _run_mock()
        return

    rclpy.init()
    node = DroneNode()
    _node = node
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"[ROS2] Error: {e}")
        push_log("SYS", "node", f"ROS2 error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        with state_lock:
            state["connected"] = False


def _run_mock():
    """Mock data loop when ROS2 is not available — keeps the whole GUI demoable."""
    push_log("SYS", "mock", "Running in MOCK mode — no ROS2")
    t = 0.0
    seeded = False
    while True:
        t += 0.05
        alt_mock = abs(math.sin(t * 0.3)) * 10
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

            # VSLAM
            state["vslam_status"] = "TRACKING"
            state["vslam_pose"]   = {"x": math.sin(t)*5, "y": math.cos(t)*5, "z": 8.0}
            trail = state["vslam_path"]
            trail.append({"x": math.sin(t)*5, "y": math.cos(t)*5})
            if len(trail) > 200:
                del trail[:-200]

            # Semantic
            state["semantic_status"] = "RETRIEVAL ACTIVE — 1 seed loaded"
            state["dock_status"]     = "IN_AIR"
            if "seed_mock" not in state["semantic_results"]:
                state["semantic_results"]["seed_mock"] = {
                    "seed_name":            "seed_mock",
                    "has_match":            True,
                    "match_count":          3,
                    "frame_indices":        [1200, 1215, 1230],
                    "timestamps_sec":       [40.0, 40.5, 41.0],
                    "similarity_scores":    [0.82, 0.79, 0.76],
                    "similarity_threshold": 0.75,
                    "poses": [
                        {"valid": True, "x": 4.1, "y": 3.2, "z": 8.5},
                        {"valid": True, "x": 4.3, "y": 3.4, "z": 8.4},
                        {"valid": True, "x": 4.5, "y": 3.6, "z": 8.3},
                    ],
                    "updated_at": time.time(),
                }
                if "seed_mock" not in state["seeds"]:
                    state["seeds"].append("seed_mock")

        if not seeded:
            seeded = True
            push_log("SYS", "mock", "VSLAM TRACKING + 1 mock seed with 3 matches loaded")
            push_log("IN", "/semantic_retrieval/results",
                     "seed=seed_mock has_match=True matches=3 best=0.820")
        time.sleep(0.05)


# ── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the ROS2 (or mock) bridge thread on startup
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    push_log("SYS", "boot", "GCS backend starting")
    print("[FastAPI] Bridge thread started — docs at http://localhost:5000/docs")
    yield
    # (daemon thread exits with the process; nothing to clean up explicitly)


app = FastAPI(
    title="ANVESHAN GCS Backend",
    description="Ground-control bridge: ROS2 telemetry + semantic retrieval + VSLAM + video proxy.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies/credentials; keeps wildcard origin valid
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ───────────────────────────────────────────────────────────
@app.get("/api/telemetry", response_model=Telemetry)
def telemetry():
    return telemetry_snapshot()


@app.get("/api/detections")
def detections():
    """Repointed: /seed_detections was a seed INPUT, not output. Returns semantic results."""
    with state_lock:
        return dict(state["semantic_results"])


@app.get("/api/status", response_model=Status)
def status():
    with state_lock:
        return {"connected": state["connected"]}


# ── Semantic retrieval ───────────────────────────────────────────────────────
@app.get("/api/semantic_results")
def semantic_results():
    """seed_name → {has_match, match_count, frame_indices, timestamps_sec,
    similarity_scores, similarity_threshold, poses:[{valid,x,y,z}...]}."""
    with state_lock:
        return dict(state["semantic_results"])


@app.get("/api/semantic_status")
def semantic_status():
    with state_lock:
        return {
            "status": state["semantic_status"],
            "seeds":  list(state["seeds"]),
        }


@app.post("/api/add_seed")
def add_seed(file: UploadFile = File(...), seed_name: str = Form(...),
             _auth: bool = Depends(require_token)):
    """Upload an image → publish to /semantic_retrieval/add_seed (frame_id = seed name)."""
    raw = file.file.read()
    if not raw:
        return Response(content=json.dumps({"error": "empty file"}),
                        media_type="application/json", status_code=400)
    if len(raw) > 25 * 1024 * 1024:   # cap — seed references are small images
        return Response(content=json.dumps({"error": "file too large (max 25 MB)"}),
                        media_type="application/json", status_code=413)

    # Keep the raw bytes so the GUI can show a real thumbnail for this seed
    # (served via GET /api/seed_image/{seed_name}). Survives browser refresh and
    # is visible to every LAN client, unlike the browser-local preview.
    with state_lock:
        state["seed_images"][seed_name] = {
            "media_type": file.content_type or "image/png",
            "data": raw,
        }

    if ROS_AVAILABLE:
        np_arr = np.frombuffer(raw, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            return Response(content=json.dumps({"error": "could not decode image"}),
                            media_type="application/json", status_code=400)
        if _node is None:
            return Response(content=json.dumps({"error": "ROS node not ready"}),
                            media_type="application/json", status_code=503)
        ros_img = _node.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        ros_img.header.frame_id = seed_name
        _publish_seed(ros_img)            # thread-safe (marshalled to executor)
        h, w = cv_img.shape[0], cv_img.shape[1]
        with state_lock:
            if seed_name not in state["seeds"]:
                state["seeds"].append(seed_name)
        push_log("OUT", "/semantic_retrieval/add_seed",
                 f"Published seed: {seed_name} ({w}x{h})",
                 {"seed_name": seed_name, "width": w, "height": h})
        return {"success": True, "seed_name": seed_name, "message": "seed published to ROS2"}

    # Mock: store the seed so the GUI shows it.
    with state_lock:
        if seed_name not in state["seeds"]:
            state["seeds"].append(seed_name)
        state["semantic_results"].setdefault(seed_name, {
            "seed_name":            seed_name,
            "has_match":            False,
            "match_count":          0,
            "frame_indices":        [],
            "timestamps_sec":       [],
            "similarity_scores":    [],
            "similarity_threshold": 0.75,
            "poses":                [],
            "updated_at":           time.time(),
        })
    push_log("OUT", "/semantic_retrieval/add_seed", f"(mock) stored seed {seed_name}")
    return {"success": True, "seed_name": seed_name, "message": "seed stored (mock mode)"}


@app.get("/api/seed_image/{seed_name}")
def seed_image(seed_name: str):
    """Return the operator-uploaded seed reference image (GUI thumbnail).

    404 when the seed was loaded on-robot from a seeds_dir rather than uploaded
    through the GUI — the frontend falls back to a placeholder in that case.
    """
    with state_lock:
        entry = state["seed_images"].get(seed_name)
    if entry:
        return Response(content=entry["data"], media_type=entry["media_type"])

    # Fallback: if the backend shares the filesystem with the seeds_dir (the
    # all-on-one-box test), serve the image straight from disk by seed name.
    # GCS_SEEDS_DIR is set by gui_bringup.launch.py. Realpath-pinned so a crafted
    # seed_name can't escape the folder.
    seeds_dir = os.environ.get("GCS_SEEDS_DIR", "").strip()
    if seeds_dir and os.path.isdir(seeds_dir):
        root = os.path.realpath(seeds_dir)
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            p = os.path.join(seeds_dir, seed_name + ext)
            if os.path.isfile(p) and os.path.realpath(p).startswith(root + os.sep):
                mt = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".bmp": "image/bmp", ".tiff": "image/tiff"}[ext]
                with open(p, "rb") as f:
                    return Response(content=f.read(), media_type=mt)

    return Response(content=json.dumps({"error": "no image for seed"}),
                    media_type="application/json", status_code=404)


@app.delete("/api/seed/{seed_name}")
def remove_seed(seed_name: str, _auth: bool = Depends(require_token)):
    """Remove a seed: tell the retriever to drop it + clear it from GUI state."""
    if ROS_AVAILABLE:
        _publish_remove_seed(seed_name)
    with state_lock:
        if seed_name in state["seeds"]:
            state["seeds"].remove(seed_name)
        state["semantic_results"].pop(seed_name, None)
        state["seed_images"].pop(seed_name, None)
    push_log("OUT", "/semantic_retrieval/remove_seed", f"Removed seed: {seed_name}")
    return {"success": True, "seed_name": seed_name, "message": "seed removed"}


def _display_threshold(cfg: Dict[str, Any]) -> float:
    """Resolve the GUI's starting threshold. The launch-time env var
    GCS_DISPLAY_THRESHOLD (set by gui_bringup) still wins for back-compat with the
    autonomous-run workflow; otherwise fall back to the saved config value."""
    dt = cfg.get("semantic", {}).get("display_threshold", 0.57)
    env_dt = os.environ.get("GCS_DISPLAY_THRESHOLD")
    if env_dt is not None:
        try:
            dt = float(env_dt)
        except ValueError:
            pass
    return dt


@app.get("/api/config")
def get_config():
    """Full persisted GUI config (defaults ← config.default.json ← config.json).
    Also includes a top-level `display_threshold` for the Analysis threshold box
    (back-compat: the GCS_DISPLAY_THRESHOLD launch env still overrides it)."""
    cfg = config_store.load_config()
    return {**cfg, "display_threshold": _display_threshold(cfg)}


@app.put("/api/config")
def put_config(payload: Dict[str, Any], _auth: bool = Depends(require_token)):
    """Merge `payload` over the saved config and persist to config.json. Returns
    the full merged config. Mutating → requires X-GCS-Token when one is set."""
    try:
        merged = config_store.save_config(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid config: {e}")
    push_log("SYS", "config", "configuration updated via PUT /api/config")
    return {**merged, "display_threshold": _display_threshold(merged)}


# ── Docking control + terminal feed ──────────────────────────────────────────
_DOCK_VOCAB = ("DOCK", "UNDOCK", "STOP_CHARGING", "EMERGENCY", "IDLE")


class DockCommandRequest(BaseModel):
    command: str        # DOCK | UNDOCK | CHARGE | CHARGE:<mah> | STOP_CHARGING | EMERGENCY | IDLE


@app.post("/api/dock/command")
def dock_command(req: DockCommandRequest, _auth: bool = Depends(require_token)):
    """Send a manual docking command to dock_manager on /dock/command."""
    cmd = req.command.strip().upper()
    base = cmd.split(":", 1)[0]
    if base != "CHARGE" and base not in _DOCK_VOCAB:
        raise HTTPException(status_code=400, detail=f"unknown dock command: {cmd}")
    if not ROS_AVAILABLE:
        return {"published": False, "command": cmd, "message": "ROS unavailable (mock mode)"}
    _publish_dock_command(cmd)
    push_log("OUT", "/dock/command", cmd)
    return {"published": True, "command": cmd}


@app.get("/api/dock/log")
def dock_log(since: int = 0):
    """Docking terminal lines after `since` (sequence id) + current dock state.
    The GUI polls this while its docking panel is open."""
    with dock_log_lock:
        lines = [e for e in dock_log_buf if e["seq"] > since]
        last = dock_log_buf[-1]["seq"] if dock_log_buf else since
    with state_lock:
        ds = state["dock_state"]
    return {"lines": lines, "last_seq": last, "dock_state": ds}


@app.post("/api/trigger_indexing")
def trigger_indexing(req: TriggerIndexingRequest, _auth: bool = Depends(require_token)):
    """Kick off rosbag indexing (async). Returns submitted/queued immediately."""
    result = _trigger_indexing(req.rosbag_path, req.output_db_path, req.force_reindex)
    return result


@app.post("/api/command")
def command(req: CommandRequest, _auth: bool = Depends(require_token)):
    """Publish a fixed-vocab override command to /gcs/command.

    400 if the command is not in the vocabulary; 409 if the current flight
    state forbids it (e.g. START while in-air). A 200 means we PUBLISHED the
    command — not that the drone obeyed. Watch last_command_ack (if the flight
    team echoes /gcs/command_ack) for actual acknowledgement.
    """
    cmd = req.command.strip().upper()
    if cmd not in COMMAND_VOCAB:
        raise HTTPException(status_code=400,
                            detail=f"unknown command '{cmd}' (allowed: {', '.join(COMMAND_VOCAB)})")
    ok, reason = command_precondition(cmd)
    if not ok:
        raise HTTPException(status_code=409, detail=f"{cmd} blocked: {reason}")
    return _publish_command(cmd)


@app.get("/api/frame/{frame_index}")
def get_frame(frame_index: int):
    """Fetch an HD frame via GetHdFrame. CompressedImage.data is raw JPEG → base64 it."""
    holder = _get_hd_frame(frame_index)
    if "error" in holder:
        return Response(content=json.dumps({"error": holder["error"], "frame_index": frame_index}),
                        media_type="application/json", status_code=503)
    res = holder.get("result")
    if res is None or not getattr(res, "success", False):
        msg = getattr(res, "message", "frame not available") if res is not None else "no result"
        return Response(content=json.dumps({"error": msg, "frame_index": frame_index}),
                        media_type="application/json", status_code=404)
    # CompressedImage.data is already JPEG bytes — just base64, no cv_bridge needed.
    b64 = base64.b64encode(bytes(res.image.data)).decode("utf-8")
    return {"frame": "data:image/jpeg;base64," + b64, "frame_index": frame_index}


@app.get("/api/logs")
def logs(n: int = 100):
    """Last N entries of the IN/OUT/SYS ring buffer."""
    n = max(1, min(n, MAX_LOGS))
    with log_lock:
        entries = list(log_buffer)
    return {"logs": entries[-n:], "total": len(entries)}


# ── WebSocket: push telemetry + semantic + VSLAM (no polling needed) ─────────
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(full_snapshot())
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
