import threading
import math
import json
import time
import urllib.request
from flask import Flask, jsonify, Response
from flask_cors import CORS
from rclpy.qos import qos_profile_sensor_data


# ── Jetson IP — only needed for video. Change to your Jetson's IP ────────────
#JETSON_IP          = "10.202.3.2"   # <-- set this
#JETSON_IP          = "192.168.0.144"   # <-- set thi
JETSON_IP          = "192.168.0.29"   # <-- set this
JETSON_STREAM_PORT = 8765
JETSON_STREAM_URL  = f"http://{JETSON_IP}:{JETSON_STREAM_PORT}/"

# ── Try importing rclpy (ROS2) ───────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] rclpy not found – running with mock data")

# ── Shared state ─────────────────────────────────────────────────────────────
state = {
    "connected":   False,
    "position":    {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "velocity":    {"horizontal": 0.0, "climbRate": 0.0},
    "battery":     {"percentage": 0.0, "voltage": 0.0},
    "altitude":    0.0,
    "detections":  [],
}
state_lock = threading.Lock()


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
class DroneNode(Node):
    def __init__(self):
        super().__init__('gcs_flask_bridge')

        self.create_subscription(Odometry,     '/mavros/local_position/odom', self.on_odom,      qos_profile_sensor_data)
        self.create_subscription(BatteryState, '/mavros/battery',             self.on_battery,   qos_profile_sensor_data)
        self.create_subscription(String,       '/seed_detections',            self.on_detection, 10)

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


# ── ROS2 spin thread ─────────────────────────────────────────────────────────
def ros_thread():
    if not ROS_AVAILABLE:
        t = 0
        while True:
            t += 0.05
            with state_lock:
                state["connected"]   = True
                state["position"]    = {"x": math.sin(t)*5, "y": math.cos(t)*5, "z": abs(math.sin(t*0.3))*10}
                state["orientation"] = {"roll": math.sin(t*0.7)*15, "pitch": math.cos(t*0.5)*10, "yaw": (t*20)%360}
                state["velocity"]    = {"horizontal": abs(math.sin(t))*8, "climbRate": math.cos(t*0.4)*2}
                state["battery"]     = {"percentage": max(0, 80-t*0.1), "voltage": max(10, 14.8-t*0.01)}
                state["altitude"]    = abs(math.sin(t*0.3))*10
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


# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/api/telemetry")
def telemetry():
    with state_lock:
        return jsonify({k: v for k, v in state.items() if k != "detections"})


@app.route("/api/detections")
def detections():
    with state_lock:
        return jsonify(state["detections"])


@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({"connected": state["connected"]})


@app.route("/video_feed")
def video_feed():
    """
    Proxy the MJPEG stream from the Jetson streamer directly to the browser.
    The browser keeps this connection open and receives frames continuously.
    """
    def generate():
        try:
            req = urllib.request.urlopen(JETSON_STREAM_URL, timeout=10)
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            print(f"[Video] Stream error: {e}")

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    print("[Flask] Starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
