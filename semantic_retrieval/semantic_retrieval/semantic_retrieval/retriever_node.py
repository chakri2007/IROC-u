"""
retriever_node.py

Phase 2 — Semantic Retriever Node
===================================
Loads embedding DB from disk (written by indexer_node).
Loads seeds from disk at startup + accepts new seeds via topic at runtime.
Runs cosine similarity search at a configurable interval.
Publishes SemanticMatch results per seed on a ROS2 topic.

Topic published per seed: /semantic_retrieval/results  (SemanticMatch.msg)

Runtime seed topic: /semantic_retrieval/add_seed  (sensor_msgs/Image)
  - Image.header.frame_id = seed name (used as identifier)
  - Publish a new seed image here at any time; retriever picks it up immediately

Usage:
  ros2 run semantic_retrieval retriever_node
"""

import os
import time
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer as RosTimer
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage

from cv_bridge import CvBridge
import numpy as np
import torch
from PIL import Image

# Local engine
from semantic_retrieval.embedding_engine import (
    load_model,
    embed_seed,
    cosine_search,
    load_db,
)

# Custom message (lives in the separate semantic_retrieval_interfaces package)
try:
    from semantic_retrieval_interfaces.msg import SemanticMatch
    CUSTOM_MSG_AVAILABLE = True
except ImportError:
    CUSTOM_MSG_AVAILABLE = False
    print(
        "[RetrieverNode] WARNING: SemanticMatch message not built yet. "
        "Run colcon build first."
    )


# ============================================================
# RETRIEVER NODE
# ============================================================

class RetrieverNode(Node):

    def __init__(self):
        super().__init__("semantic_retriever")

        # ---- Declare + read parameters ----
        self.declare_parameter("model_name",             "facebook/dinov2-small")
        self.declare_parameter("use_fp16",               True)
        self.declare_parameter("embedding_db_path",      "/home/user/semantic_db/video_embeddings.pt")
        self.declare_parameter("seeds_dir",              "/home/user/seeds/")
        self.declare_parameter("runtime_seed_topic",     "/semantic_retrieval/add_seed")
        self.declare_parameter("results_topic",          "/semantic_retrieval/results")
        self.declare_parameter("status_topic",           "/semantic_retrieval/status")
        self.declare_parameter("top_k",                  5)
        self.declare_parameter("similarity_threshold",   0.75)
        self.declare_parameter("retrieval_interval_sec", 2.0)

        self.model_name          = self.get_parameter("model_name").value
        self.use_fp16            = self.get_parameter("use_fp16").value
        self.db_path             = self.get_parameter("embedding_db_path").value
        self.seeds_dir           = self.get_parameter("seeds_dir").value
        self.top_k               = self.get_parameter("top_k").value
        self.threshold           = self.get_parameter("similarity_threshold").value
        self.retrieval_interval  = self.get_parameter("retrieval_interval_sec").value

        self.bridge = CvBridge()

        # ---- Load model ----
        self.get_logger().info("Loading DINOv2 model...")
        load_model(self.model_name, self.use_fp16)
        self.get_logger().info("Model ready.")

        # ---- State ----
        self._db_lock        = threading.Lock()
        self._seed_lock      = threading.Lock()
        self._retrieval_lock = threading.Lock()   # serialise retrieval runs

        self.video_embeddings = None   # (N, 6, D)
        self.frame_indices    = []
        self.timestamps       = []
        self.poses            = []     # list[dict|None] — VSLAM pose per frame
        self.db_loaded        = False

        # seed_name → seed_embedding (1, D)
        self.seeds: dict[str, torch.Tensor] = {}

        # ---- Publishers ----
        results_topic = self.get_parameter("results_topic").value
        status_topic  = self.get_parameter("status_topic").value

        # Latched QoS so a late-joining ground laptop GUI immediately receives
        # the last published result (TRANSIENT_LOCAL + RELIABLE + depth 1).
        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        if CUSTOM_MSG_AVAILABLE:
            self.results_pub = self.create_publisher(SemanticMatch, results_topic, latched_qos)
        else:
            self.get_logger().warn("SemanticMatch not available — results will not be published.")
            self.results_pub = None

        self.status_pub = self.create_publisher(String, status_topic, 10)

        # ---- Runtime seed subscriber ----
        runtime_seed_topic = self.get_parameter("runtime_seed_topic").value
        self.create_subscription(
            RosImage,
            runtime_seed_topic,
            self._handle_runtime_seed,
            10
        )
        self.get_logger().info(f"Listening for runtime seeds on: {runtime_seed_topic}")

        # ---- Load static seeds from disk ----
        self._load_seeds_from_disk()

        # ---- Try to load DB if it already exists (from previous mission) ----
        self._try_load_db()

        # ---- DB watcher — polls for DB file if not loaded yet ----
        # (Indexer writes DB after docking; retriever watches and loads it)
        self._db_watch_timer = self.create_timer(5.0, self._watch_for_db)

        # ---- Retrieval timer ----
        self._retrieval_timer = self.create_timer(
            self.retrieval_interval,
            self._run_retrieval_cycle
        )

        self.get_logger().info(
            f"SemanticRetrieverNode ready. "
            f"Threshold: {self.threshold} | Top-K: {self.top_k} | "
            f"Interval: {self.retrieval_interval}s"
        )

    # ----------------------------------------------------------
    # DB LOADING
    # ----------------------------------------------------------

    def _try_load_db(self):
        if os.path.exists(self.db_path):
            try:
                with self._db_lock:
                    self.video_embeddings, self.frame_indices, self.timestamps, self.poses = \
                        load_db(self.db_path)
                    self.db_loaded = True
                self.get_logger().info(
                    f"Embedding DB loaded: {len(self.frame_indices)} frames indexed."
                )
                self._publish_status(
                    f"DB loaded. {len(self.frame_indices)} frames. "
                    f"Threshold: {self.threshold}"
                )
                # Event-driven: run retrieval as soon as the DB first appears.
                self._trigger_retrieval_async()
            except Exception as e:
                self.get_logger().error(f"Failed to load DB: {e}")
        else:
            self.get_logger().info(
                f"DB not found yet at {self.db_path}. "
                f"Waiting for indexer to complete..."
            )

    def _watch_for_db(self):
        """
        Polls every 5s. Once the DB appears (written by indexer after docking),
        loads it and cancels the watcher.
        """
        if self.db_loaded:
            self._db_watch_timer.cancel()
            return

        if os.path.exists(self.db_path):
            self.get_logger().info("DB file detected — loading...")
            self._try_load_db()
            if self.db_loaded:
                self._db_watch_timer.cancel()

    # ----------------------------------------------------------
    # SEED MANAGEMENT
    # ----------------------------------------------------------

    def _load_seeds_from_disk(self):
        """
        Load all image files from seeds_dir at startup.
        Supported: .png .jpg .jpeg .bmp .tiff
        """
        seeds_path = Path(self.seeds_dir)
        if not seeds_path.exists():
            self.get_logger().warn(f"Seeds directory not found: {self.seeds_dir}")
            return

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        loaded = 0

        for f in sorted(seeds_path.iterdir()):
            if f.suffix.lower() not in extensions:
                continue
            try:
                pil_img = Image.open(f).convert("RGB")
                seed_emb = embed_seed(pil_img)                    # (1, D)
                seed_name = f.stem                                 # filename without extension
                with self._seed_lock:
                    self.seeds[seed_name] = seed_emb
                loaded += 1
                self.get_logger().info(f"Seed loaded: {seed_name}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load seed {f.name}: {e}")

        self.get_logger().info(f"Static seeds loaded: {loaded}")
        self._publish_status(f"Seeds loaded: {loaded} from disk.")

    def _handle_runtime_seed(self, msg: RosImage):
        """
        New seed published at runtime on /semantic_retrieval/add_seed.
        msg.header.frame_id is used as the seed name.
        """
        seed_name = msg.header.frame_id.strip()
        if not seed_name:
            seed_name = f"seed_{int(time.time())}"
            self.get_logger().warn(
                f"Runtime seed has no frame_id — using auto-name: {seed_name}"
            )

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_img = Image.fromarray(cv_img)
            seed_emb = embed_seed(pil_img)                        # (1, D)

            with self._seed_lock:
                self.seeds[seed_name] = seed_emb

            self.get_logger().info(f"Runtime seed added: {seed_name}")
            self._publish_status(f"Runtime seed registered: {seed_name}")
            # Event-driven: run retrieval immediately when a new seed arrives.
            self._trigger_retrieval_async()

        except Exception as e:
            self.get_logger().error(f"Failed to process runtime seed '{seed_name}': {e}")

    # ----------------------------------------------------------
    # RETRIEVAL CYCLE
    # ----------------------------------------------------------

    def _trigger_retrieval_async(self):
        """
        Run a retrieval cycle off the executor thread (event-driven trigger).
        Used when the DB first loads and whenever a new seed arrives so the
        GUI gets a fresh latched result without waiting for the periodic
        backstop timer.
        """
        threading.Thread(target=self._run_retrieval_cycle, daemon=True).start()

    def _run_retrieval_cycle(self):
        """
        Runs cosine search for every registered seed against the video DB and
        publishes one SemanticMatch message per seed.

        Invoked event-driven (DB load / new seed) and also by the periodic
        retrieval_interval_sec timer as a backstop. A non-blocking lock keeps
        concurrent triggers from overlapping.
        """
        if not self.db_loaded:
            return

        if not self.seeds:
            return

        if not CUSTOM_MSG_AVAILABLE or self.results_pub is None:
            return

        if not self._retrieval_lock.acquire(blocking=False):
            # A retrieval pass is already running — skip this trigger.
            return
        try:
            self._do_retrieval()
        finally:
            self._retrieval_lock.release()

    def _do_retrieval(self):
        with self._seed_lock:
            seed_snapshot = dict(self.seeds)

        with self._db_lock:
            emb_snapshot   = self.video_embeddings
            idx_snapshot   = self.frame_indices
            ts_snapshot    = self.timestamps
            pose_snapshot  = self.poses

        for seed_name, seed_emb in seed_snapshot.items():

            try:
                result = cosine_search(
                    seed_emb=seed_emb,
                    video_embeddings=emb_snapshot,
                    frame_indices=idx_snapshot,
                    timestamps=ts_snapshot,
                    top_k=self.top_k,
                    threshold=self.threshold,
                    poses=pose_snapshot,
                )

                ros_msg = self._build_ros_message(
                    seed_name=seed_name,
                    result=result,
                    emb_snapshot=emb_snapshot,
                    idx_snapshot=idx_snapshot,
                    ts_snapshot=ts_snapshot,
                )

                self.results_pub.publish(ros_msg)

                if result["has_match"]:
                    self.get_logger().info(
                        f"[{seed_name}] Match: frame={result['frame_indices'][0]} "
                        f"ts={result['timestamps_sec'][0]:.2f}s "
                        f"score={result['best_score']:.4f}"
                    )
                else:
                    self.get_logger().info(
                        f"[{seed_name}] No match above threshold {self.threshold:.2f}. "
                        f"Best score: {result['best_score']:.4f}"
                    )

            except Exception as e:
                self.get_logger().error(f"Retrieval error for seed '{seed_name}': {e}")

    # ----------------------------------------------------------
    # BUILD ROS MESSAGE
    # ----------------------------------------------------------

    def _build_ros_message(
        self,
        seed_name: str,
        result: dict,
        emb_snapshot,
        idx_snapshot,
        ts_snapshot,
    ) -> "SemanticMatch":
        """
        Build and return a SemanticMatch message.
        If has_match is False, frame arrays are empty.

        NOTE: hd_frames is intentionally always left EMPTY. HD frames are now
        served on demand via the GetHdFrame service (/semantic_retrieval/get_hd_frame)
        so the results topic stays light for a ground laptop with no rosbag access.
        The GUI uses frame_indices to request each HD frame from the service.
        """
        msg = SemanticMatch()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = "semantic_retrieval"
        msg.seed_name            = seed_name
        msg.has_match            = result["has_match"]
        msg.match_count          = len(result["frame_indices"])
        msg.frame_indices        = [int(i)   for i in result["frame_indices"]]
        msg.timestamps_sec       = [float(t) for t in result["timestamps_sec"]]
        msg.similarity_scores    = [float(s) for s in result["scores"]]
        msg.similarity_threshold = float(self.threshold)
        msg.hd_frames            = []

        # ---- Pack VSLAM poses ----
        positions_xyz   = []
        orientations    = []
        pose_valid_flags = []

        for pose in result.get("poses", []):
            if pose is not None:
                positions_xyz.extend([pose["x"], pose["y"], pose["z"]])
                orientations.extend([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
                pose_valid_flags.append(True)
            else:
                positions_xyz.extend([0.0, 0.0, 0.0])
                orientations.extend([0.0, 0.0, 0.0, 1.0])
                pose_valid_flags.append(False)

        msg.positions_xyz    = positions_xyz
        msg.orientations_xyzw = orientations
        msg.pose_valid       = pose_valid_flags

        # hd_frames stays empty by design — HD frames are served via the
        # GetHdFrame service (hd_frame_server), not bulk-published here.
        return msg

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _publish_status(self, text: str):
        msg = String()
        msg.data = f"[Retriever] {text}"
        self.status_pub.publish(msg)


# ============================================================
# ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = RetrieverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
