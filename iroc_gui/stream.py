#!/usr/bin/env python3
"""
stream.py — MJPEG restreamer for the nadir camera (config-driven, GUI-adopted).

This is the team's original ``~/Desktop/stream.py`` moved into the GUI so the
Setup orchestrator can launch it with the camera topic + port taken from
config.json (no hand-editing constants before every run). Run standalone it still
works and falls back to the config defaults:

    source /opt/ros/humble/setup.bash          # rclpy comes from the ROS env
    python3 stream.py                          # uses config.json
    python3 stream.py --topic /image_raw/compressed --port 8765   # explicit

It subscribes to a CompressedImage topic and serves multipart/x-mixed-replace
MJPEG on http://0.0.0.0:<port>/ — the backend's /video_feed proxies this, and the
same ROS frames feed the rosbag + semantic pipeline (ROS stays in the loop).
"""

import argparse
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()


def _config_defaults():
    """(camera_topic, port) from config.json — safe fallback if unavailable."""
    topic, port = "/image_raw/compressed", 8765
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
        import config_store
        cfg = config_store.load_config()
        topic = cfg.get("camera", {}).get("topic", topic)
        url = cfg.get("network", {}).get("video_source_url", "")
        port = urlparse(url).port or port
    except Exception:
        pass
    return topic, port


class CameraNode(Node):
    def __init__(self, topic):
        super().__init__("mjpeg_streamer")
        self.create_subscription(CompressedImage, topic, self.on_frame, 10)
        self.get_logger().info(f"Subscribed to {topic}")

    def on_frame(self, msg):
        global latest_frame
        with frame_lock:
            latest_frame = bytes(msg.data)
        frame_event.set()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                frame_event.wait(timeout=2.0)
                frame_event.clear()
                with frame_lock:
                    frame = latest_frame
                if frame is None:
                    continue
                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                    frame + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def ros_spin(topic):
    rclpy.init()
    node = CameraNode(topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    d_topic, d_port = _config_defaults()
    ap = argparse.ArgumentParser(description="Config-driven MJPEG restreamer")
    ap.add_argument("--topic", default=d_topic, help="CompressedImage topic")
    ap.add_argument("--port", type=int, default=d_port, help="HTTP MJPEG port")
    args = ap.parse_args()

    threading.Thread(target=ros_spin, args=(args.topic,), daemon=True).start()
    print(f"[Streamer] topic={args.topic}  http://0.0.0.0:{args.port}")
    HTTPServer(("0.0.0.0", args.port), MJPEGHandler).serve_forever()


if __name__ == "__main__":
    main()
