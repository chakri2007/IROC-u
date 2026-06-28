"""
indexer_node.py

Phase 1 — Semantic Indexer Node
================================
Triggered by a ROS2 service call from the mission manager after docking.

What it does:
  1. Receives rosbag path + output DB path via service request
  2. Uses rosbag2_py to read the bag directly (no need to `ros2 bag play`)
  3. Deserialises camera frames from the bag
  4. Samples every N frames, embeds with DINOv2 (6 crops/frame)
  5. Saves embedding DB (.pt) to disk
  6. Returns success/failure to mission manager via service response

Usage in launch file:
  ros2 run semantic_retrieval indexer_node

Service call example (from mission manager or terminal):
  ros2 service call /semantic_retrieval/trigger_indexing \
    semantic_retrieval_interfaces/TriggerIndexing \
    "{rosbag_path: '/data/mission_001', output_db_path: '/home/user/semantic_db/mission_001.pt', force_reindex: false}"
"""

import os
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# ROS2 message types
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# rosbag2 reading
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROSBAG2_AVAILABLE = True
except ImportError:
    ROSBAG2_AVAILABLE = False
    print("[IndexerNode] WARNING: rosbag2_py not available. Install with your ROS2 distro.")

import torch
import numpy as np

# Local engine
from semantic_retrieval.embedding_engine import (
    load_model,
    embed_frame,
    save_db,
)

# Custom service — generated from TriggerIndexing.srv
# Interfaces live in a SEPARATE package: semantic_retrieval_interfaces
try:
    from semantic_retrieval_interfaces.srv import TriggerIndexing
except ImportError:
    TriggerIndexing = None
    print("[IndexerNode] WARNING: TriggerIndexing service not yet built. Run colcon build first.")


# ============================================================
# INDEXER NODE
# ============================================================

class IndexerNode(Node):

    def __init__(self):
        super().__init__("semantic_indexer")

        # ---- Declare + read parameters ----
        self.declare_parameter("model_name",        "facebook/dinov2-small")
        self.declare_parameter("use_fp16",          True)
        self.declare_parameter("camera_topic",      "/camera/camera/color/image_raw")
        self.declare_parameter("compressed_camera_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("odometry_topic",    "/visual_slam/tracking/odometry")
        self.declare_parameter("rosbag_storage_id", "sqlite3")
        self.declare_parameter("sample_rate",       15)
        self.declare_parameter("embedding_db_path", "/home/user/semantic_db/video_embeddings.pt")
        self.declare_parameter("trigger_service",   "/semantic_retrieval/trigger_indexing")
        self.declare_parameter("status_topic",      "/semantic_retrieval/status")
        self.declare_parameter("target_size",       [128, 128])

        self.model_name              = self.get_parameter("model_name").value
        self.use_fp16                = self.get_parameter("use_fp16").value
        self.camera_topic            = self.get_parameter("camera_topic").value
        self.compressed_camera_topic = self.get_parameter("compressed_camera_topic").value
        self.odometry_topic          = self.get_parameter("odometry_topic").value
        self.storage_id              = self.get_parameter("rosbag_storage_id").value
        # int cast so frame_number % sample_rate behaves (sample_rate=1 => every frame)
        self.sample_rate             = max(1, int(self.get_parameter("sample_rate").value))
        self.default_db_path         = self.get_parameter("embedding_db_path").value
        self.target_size             = tuple(self.get_parameter("target_size").value)

        # ---- Load model at startup (warm up before first mission) ----
        self.get_logger().info("Loading DINOv2 model...")
        load_model(self.model_name, self.use_fp16)
        self.get_logger().info("Model ready.")

        self.bridge = CvBridge()

        # ---- Status publisher ----
        self.status_pub = self.create_publisher(
            String,
            self.get_parameter("status_topic").value,
            10
        )

        # ---- Indexing service ----
        # ReentrantCallbackGroup + MultiThreadedExecutor (see main) keep the
        # node responsive: the service callback returns immediately while the
        # actual indexing runs on a worker thread.
        self._cb_group = ReentrantCallbackGroup()
        if TriggerIndexing is not None:
            self.srv = self.create_service(
                TriggerIndexing,
                self.get_parameter("trigger_service").value,
                self._handle_trigger,
                callback_group=self._cb_group,
            )
            self.get_logger().info(
                f"Indexing service ready: {self.get_parameter('trigger_service').value}"
            )
        else:
            self.get_logger().warn(
                "TriggerIndexing service type not available. "
                "Build the package with colcon first."
            )

        self._indexing_lock = threading.Lock()
        self._worker = None
        self.get_logger().info("SemanticIndexerNode initialised.")

    # ----------------------------------------------------------
    # SERVICE HANDLER
    # ----------------------------------------------------------

    def _handle_trigger(self, request, response):
        """
        Called by mission manager after drone docks.

        Fire-and-return: validates the request, then dispatches the (minutes-long)
        indexing to a worker thread and returns IMMEDIATELY so the executor is
        never blocked. Progress and final result are reported on the status topic
        (/semantic_retrieval/status). The non-blocking lock guarantees only one
        indexing run happens at a time.
        """
        if not self._indexing_lock.acquire(blocking=False):
            response.success = False
            response.message = "Indexing already in progress."
            response.frames_indexed = 0
            response.indexing_time_sec = 0.0
            return response

        # We now hold the lock. Release it on any path that does NOT hand the
        # work off to the worker thread; the worker releases it otherwise.
        dispatched = False
        try:
            rosbag_path   = request.rosbag_path
            db_path       = request.output_db_path or self.default_db_path
            force_reindex = request.force_reindex

            # Security: confine paths to allowed data roots — blocks traversal and
            # writes to system dirs from an (unauthenticated) trigger call.
            _ALLOWED_ROOTS = ("/home", "/data", "/mnt", "/media", "/tmp", "/opt/anveshan")
            def _path_ok(p):
                rp = os.path.realpath(p)
                return os.path.isabs(rp) and any(
                    rp == r or rp.startswith(r + os.sep) for r in _ALLOWED_ROOTS)
            for _label, _p in (("rosbag_path", rosbag_path), ("output_db_path", db_path)):
                if not _path_ok(_p):
                    response.success = False
                    response.message = f"Refused: {_label} '{_p}' is outside the allowed data roots."
                    response.frames_indexed = 0
                    response.indexing_time_sec = 0.0
                    return response

            self._publish_status(f"Indexing triggered: {rosbag_path}")
            self.get_logger().info(f"Indexing rosbag: {rosbag_path}")

            # ---- Check if DB already exists ----
            if os.path.exists(db_path) and not force_reindex:
                self._publish_status(f"DB exists, skipping re-index: {db_path}")
                response.success = True
                response.message = f"DB already exists at {db_path}. Use force_reindex=true to re-embed."
                response.frames_indexed = -1
                response.indexing_time_sec = 0.0
                return response

            if not ROSBAG2_AVAILABLE:
                response.success = False
                response.message = "rosbag2_py not available on this system."
                response.frames_indexed = 0
                response.indexing_time_sec = 0.0
                return response

            if not os.path.exists(rosbag_path):
                response.success = False
                response.message = f"Rosbag not found: {rosbag_path}"
                response.frames_indexed = 0
                response.indexing_time_sec = 0.0
                return response

            # ---- Dispatch heavy work to a worker thread ----
            self._worker = threading.Thread(
                target=self._run_indexing,
                args=(rosbag_path, db_path),
                daemon=True,
            )
            self._worker.start()
            dispatched = True

            response.success           = True
            response.message           = (
                f"Indexing started for {rosbag_path}. "
                f"Watch {self.get_parameter('status_topic').value} for progress."
            )
            response.frames_indexed    = 0
            response.indexing_time_sec = 0.0
            return response

        finally:
            if not dispatched:
                self._indexing_lock.release()

    def _run_indexing(self, rosbag_path: str, db_path: str):
        """
        Worker-thread body. Owns the indexing lock for its whole lifetime and
        releases it in the finally block. Reports everything on the status topic.
        """
        t0 = time.time()
        try:
            frames_indexed, embeddings, frame_indices, timestamps, poses = \
                self._index_rosbag(rosbag_path)

            if frames_indexed == 0:
                msg = (
                    f"No frames found on topic '{self.camera_topic}'. "
                    f"Check camera_topic / compressed_camera_topic in config.yaml."
                )
                self._publish_status(f"FAILED: {msg}")
                self.get_logger().error(msg)
                return

            # ---- Save DB ----
            # dirname is "" for a bare filename — makedirs("") would raise.
            out_dir = os.path.dirname(db_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            save_db(db_path, embeddings, frame_indices, timestamps, poses)

            elapsed = time.time() - t0
            msg = (
                f"Indexing complete. {frames_indexed} frames embedded in "
                f"{elapsed:.1f}s. DB saved: {db_path}"
            )
            self._publish_status(f"DONE: {msg}")
            self.get_logger().info(msg)

        except Exception as e:
            self.get_logger().error(f"Indexing failed: {e}")
            self._publish_status(f"FAILED: {e}")

        finally:
            self._indexing_lock.release()

    # ----------------------------------------------------------
    # ROSBAG READER
    # ----------------------------------------------------------

    def _index_rosbag(self, rosbag_path: str):
        """
        Reads camera frames + VSLAM odometry from rosbag using rosbag2_py.
        Two-pass strategy:
          Pass 1 — read all odometry messages into a sorted timestamp→pose dict
          Pass 2 — read camera frames, sample, embed, interpolate nearest pose

        Returns:
            frames_indexed  : int
            embeddings      : torch.Tensor  (N, 6, D)
            frame_indices   : list[int]
            timestamps      : list[float]   seconds
            poses           : list[dict|None]  {x,y,z,qx,qy,qz,qw} or None
        """
        bag_uri, storage_id = self._resolve_bag(rosbag_path)
        self.get_logger().info(
            f"Opening rosbag: {bag_uri} (storage_id={storage_id})"
        )

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_uri,
            storage_id=storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        # ---- Discover topics ----
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}
        del reader  # close and reopen per pass

        # ---- Resolve camera topic ----
        # Prefer the configured raw topic; fall back to the configured
        # compressed topic (compressed_camera_topic from config.yaml).
        camera_topic = self.camera_topic
        if camera_topic not in type_map:
            compressed = self.compressed_camera_topic
            if compressed and compressed in type_map:
                camera_topic = compressed
                self.get_logger().warn(f"Raw camera not in bag — using: {camera_topic}")
            else:
                self.get_logger().error(
                    f"Camera topic '{self.camera_topic}' not in bag. "
                    f"Available: {list(type_map.keys())}"
                )
                return 0, None, [], [], []

        # ---- Resolve odometry topic ----
        odom_topic = self.odometry_topic
        has_odom = odom_topic in type_map
        if not has_odom:
            self.get_logger().warn(
                f"Odometry topic '{odom_topic}' not in bag — "
                f"frames will have no pose attached."
            )

        # ==============================================
        # PASS 1 — Read all odometry into memory
        # ==============================================
        # odom_lut: sorted list of (timestamp_ns, pose_dict)
        odom_lut = []

        if has_odom:
            self.get_logger().info(f"Pass 1: reading odometry from {odom_topic}...")
            odom_msg_type = get_message(type_map[odom_topic])

            reader1 = rosbag2_py.SequentialReader()
            reader1.open(storage_options, converter_options)
            reader1.set_filter(rosbag2_py.StorageFilter(topics=[odom_topic]))

            while reader1.has_next():
                _, data, ts_ns = reader1.read_next()
                try:
                    msg = deserialize_message(data, odom_msg_type)
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    odom_lut.append((ts_ns, {
                        "x":  p.x, "y": p.y, "z": p.z,
                        "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
                    }))
                except Exception:
                    pass

            odom_lut.sort(key=lambda x: x[0])
            self.get_logger().info(f"Pass 1 done: {len(odom_lut)} odometry poses loaded.")

        # Helper: nearest-neighbour lookup in odom_lut by timestamp
        def nearest_pose(ts_ns: int, max_gap_ns: int = 200_000_000):  # 200ms max gap
            if not odom_lut:
                return None
            lo, hi = 0, len(odom_lut) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if odom_lut[mid][0] < ts_ns:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is closest index
            best_idx = lo
            if lo > 0:
                if abs(odom_lut[lo - 1][0] - ts_ns) < abs(odom_lut[lo][0] - ts_ns):
                    best_idx = lo - 1
            gap = abs(odom_lut[best_idx][0] - ts_ns)
            return odom_lut[best_idx][1] if gap <= max_gap_ns else None

        # ==============================================
        # PASS 2 — Read camera frames + embed
        # ==============================================
        self.get_logger().info(
            f"Pass 2: embedding camera frames from {camera_topic} "
            f"(sample_rate={self.sample_rate})..."
        )

        cam_msg_type = get_message(type_map[camera_topic])

        reader2 = rosbag2_py.SequentialReader()
        reader2.open(storage_options, converter_options)
        reader2.set_filter(rosbag2_py.StorageFilter(topics=[camera_topic]))

        all_embeddings = []
        frame_indices  = []
        timestamps     = []
        poses          = []
        frame_number   = 0
        processed      = 0

        while reader2.has_next():
            _, data, ts_nanosec = reader2.read_next()

            if frame_number % self.sample_rate != 0:
                frame_number += 1
                continue

            try:
                msg = deserialize_message(data, cam_msg_type)
                if hasattr(msg, "format"):
                    cv_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                else:
                    cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().warn(f"Frame decode error @ frame {frame_number}: {e}")
                frame_number += 1
                continue

            emb  = embed_frame(cv_frame, self.target_size)  # (6, D)
            pose = nearest_pose(ts_nanosec) if has_odom else None

            all_embeddings.append(emb)
            frame_indices.append(frame_number)
            timestamps.append(ts_nanosec * 1e-9)
            poses.append(pose)
            processed += 1

            if processed % 50 == 0:
                pose_str = f"pose_ok={pose is not None}" if has_odom else "no_odom"
                self._publish_status(f"Indexed {processed} frames... ({pose_str})")
                self.get_logger().info(f"Embedded {processed} frames...")

            frame_number += 1

        if processed == 0:
            return 0, None, [], [], []

        embeddings = torch.stack(all_embeddings)  # (N, 6, D)
        poses_ok   = sum(1 for p in poses if p is not None)
        self.get_logger().info(
            f"Embedding complete: {processed} frames | "
            f"poses attached: {poses_ok}/{processed}"
        )

        return processed, embeddings, frame_indices, timestamps, poses

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _publish_status(self, msg: str):
        ros_msg = String()
        ros_msg.data = f"[Indexer] {msg}"
        self.status_pub.publish(ros_msg)

    def _resolve_bag(self, rosbag_path: str):
        """
        Normalise a rosbag path for rosbag2_py and auto-detect its storage_id.

        rosbag2_py expects the bag DIRECTORY (the one containing metadata.yaml),
        not an individual .db3/.mcap file. If a file path is given, use its
        parent directory. The storage_id is read from metadata.yaml when present
        (sqlite3 vs mcap), falling back to the rosbag_storage_id param.

        Returns (uri, storage_id).
        """
        bag_dir = rosbag_path
        if os.path.isfile(rosbag_path):
            bag_dir = os.path.dirname(rosbag_path) or "."

        storage_id = self.storage_id  # param fallback
        meta_path = os.path.join(bag_dir, "metadata.yaml")
        if os.path.isfile(meta_path):
            try:
                import yaml
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f) or {}
                info = meta.get("rosbag2_bagfile_information", {}) or {}
                sid = info.get("storage_identifier")
                if sid:
                    storage_id = sid
                    self.get_logger().info(
                        f"Auto-detected storage_id='{storage_id}' from metadata.yaml"
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"Could not parse {meta_path}: {e} — "
                    f"falling back to storage_id='{storage_id}'"
                )
        else:
            self.get_logger().warn(
                f"No metadata.yaml in '{bag_dir}' — "
                f"using configured storage_id='{storage_id}'"
            )
        return bag_dir, storage_id


# ============================================================
# ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = IndexerNode()
    # MultiThreadedExecutor so the service callback (and status publishing)
    # keep running while indexing executes on its worker thread.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
