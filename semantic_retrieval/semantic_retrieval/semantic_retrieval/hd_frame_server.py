"""
hd_frame_server.py

HD Frame Server Node — ON-DEMAND SERVICE
========================================
The GUI backend runs on a GROUND LAPTOP that has no rosbag access. Instead of
bulk-publishing HD frames, this node exposes a ROS2 SERVICE that returns a
single HD frame on request, JPEG-encoded as a sensor_msgs/CompressedImage.

Service:
  /semantic_retrieval/get_hd_frame   (semantic_retrieval_interfaces/srv/GetHdFrame)
    REQUEST : int32 frame_index
    RESPONSE: bool success, sensor_msgs/CompressedImage image, int32 frame_index, string message

Flow:
  1. retriever_node publishes SemanticMatch with frame_indices per seed.
  2. GUI picks a frame_index and calls this service.
  3. Server reads that frame from the rosbag camera stream, JPEG-encodes it,
     and returns it as a CompressedImage (.format = "jpeg").

frame_index alignment:
  frame_index is the ABSOLUTE count of messages on the SAME camera_topic the
  indexer used (every message, no sampling). We therefore use the identical
  camera_topic param and resolve the compressed fallback the same way.

Efficiency:
  We build a frame_index -> bag-timestamp map ONCE per rosbag (a single cheap
  scan that records timestamps without deserializing), then seek() straight to
  the requested frame. A small LRU caches recently decoded frames so repeated
  GUI requests are essentially free.

Usage:
  ros2 run semantic_retrieval hd_frame_server
  ros2 param set /hd_frame_server rosbag_path /data/mission_001
"""

import os
import threading
from collections import OrderedDict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROSBAG2_AVAILABLE = True
except ImportError:
    ROSBAG2_AVAILABLE = False

# Custom service (lives in the separate semantic_retrieval_interfaces package)
try:
    from semantic_retrieval_interfaces.srv import GetHdFrame
    CUSTOM_SRV_AVAILABLE = True
except ImportError:
    CUSTOM_SRV_AVAILABLE = False


# ============================================================
# HD FRAME SERVER NODE
# ============================================================

class HdFrameServerNode(Node):

    def __init__(self):
        super().__init__("hd_frame_server")

        # ---- Parameters ----
        # camera_topic MUST match the indexer so frame_index alignment holds.
        self.declare_parameter("camera_topic",      "/camera/camera/color/image_raw")
        self.declare_parameter("rosbag_path",       "")   # set at launch or via param update
        self.declare_parameter("rosbag_storage_id", "sqlite3")  # fallback if auto-detect fails
        self.declare_parameter("get_hd_frame_service", "/semantic_retrieval/get_hd_frame")
        self.declare_parameter("status_topic",      "/semantic_retrieval/status")
        self.declare_parameter("frame_cache_size",  64)

        self.camera_topic     = self.get_parameter("camera_topic").value
        self.rosbag_path      = self.get_parameter("rosbag_path").value
        self.fallback_storage = self.get_parameter("rosbag_storage_id").value
        self.status_topic     = self.get_parameter("status_topic").value
        self.cache_size       = int(self.get_parameter("frame_cache_size").value)

        self.bridge = CvBridge()

        # ---- Index + cache state ----
        self._index_lock = threading.Lock()
        # frame_index -> timestamp_ns map (the "bag position" key for seek()).
        self._frame_offsets = None          # list[int] or None until built
        self._indexed_path  = None          # which rosbag the offsets belong to
        self._cam_topic_resolved = None     # camera topic actually present in bag
        self._cam_type_name = None          # ros msg type string of that topic
        self._storage_id    = None          # auto-detected storage id for the bag

        # Small LRU of decoded CompressedImage, keyed by frame_index.
        self._cache_lock = threading.Lock()
        self._frame_cache: "OrderedDict[int, CompressedImage]" = OrderedDict()

        # Status publisher
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        # ---- Service server ----
        if CUSTOM_SRV_AVAILABLE:
            service_name = self.get_parameter("get_hd_frame_service").value
            self.srv = self.create_service(
                GetHdFrame,
                service_name,
                self._handle_get_hd_frame,
            )
            self.get_logger().info(f"HD frame service ready: {service_name}")
        else:
            self.srv = None
            self.get_logger().warn(
                "GetHdFrame service type not available — HD server inactive. "
                "Run colcon build first."
            )

        # ---- Rosbag path watcher ----
        # Allows rosbag_path to be updated at runtime via ros2 param set.
        self.create_timer(3.0, self._refresh_rosbag_path)

        # Build the index eagerly if a bag is already configured.
        if self.rosbag_path:
            threading.Thread(target=self._ensure_index, daemon=True).start()

        self.get_logger().info("HdFrameServerNode ready.")

    # ----------------------------------------------------------
    # SERVICE HANDLER
    # ----------------------------------------------------------

    def _handle_get_hd_frame(self, request, response):
        """
        Return the HD frame at request.frame_index as a JPEG CompressedImage.
        On any miss: success=false and an empty image.
        """
        frame_index = int(request.frame_index)
        response.frame_index = frame_index
        response.image = CompressedImage()
        response.image.format = "jpeg"

        if not CUSTOM_SRV_AVAILABLE:
            response.success = False
            response.message = "GetHdFrame service type not built."
            return response

        if not ROSBAG2_AVAILABLE:
            response.success = False
            response.message = "rosbag2_py not available on this system."
            return response

        if not self.rosbag_path or not os.path.exists(self.rosbag_path):
            response.success = False
            response.message = (
                f"rosbag_path not set or not found: '{self.rosbag_path}'. "
                f"Set via: ros2 param set /hd_frame_server rosbag_path /path/to/bag"
            )
            return response

        # Serve from LRU if we already decoded this frame.
        cached = self._cache_get(frame_index)
        if cached is not None:
            cached.header.stamp = self.get_clock().now().to_msg()
            response.success = True
            response.image = cached
            response.message = f"frame {frame_index} (cached)"
            return response

        try:
            comp = self._read_frame(frame_index)
        except Exception as e:
            self.get_logger().warn(f"HD frame read failed (frame {frame_index}): {e}")
            response.success = False
            response.message = f"read error: {e}"
            return response

        if comp is None:
            response.success = False
            response.message = f"frame {frame_index} not found in bag"
            return response

        comp.header.stamp = self.get_clock().now().to_msg()
        comp.header.frame_id = f"hd_frame_{frame_index}"
        self._cache_put(frame_index, comp)

        response.success = True
        response.image = comp
        response.message = f"frame {frame_index}"
        return response

    # ----------------------------------------------------------
    # ROSBAG INDEX (frame_index -> timestamp map, built once)
    # ----------------------------------------------------------

    def _ensure_index(self):
        """
        Build the frame_index -> timestamp_ns map once for the current rosbag.
        A single scan records message timestamps on the camera topic WITHOUT
        deserializing payloads, so it is cheap. Thereafter lookups seek().
        """
        with self._index_lock:
            if self._frame_offsets is not None and self._indexed_path == self.rosbag_path:
                return
            if not self.rosbag_path or not os.path.exists(self.rosbag_path):
                return

            storage_id = self._detect_storage_id(self.rosbag_path)

            reader = self._open_reader(storage_id)
            topic_types = reader.get_all_topics_and_types()
            type_map = {t.name: t.type for t in topic_types}

            cam_topic = self._resolve_camera_topic(type_map)
            if cam_topic is None:
                self.get_logger().error(
                    f"Camera topic '{self.camera_topic}' (or /compressed) not in bag. "
                    f"Available: {list(type_map.keys())}"
                )
                self._frame_offsets = []
                self._indexed_path = self.rosbag_path
                return

            reader.set_filter(rosbag2_py.StorageFilter(topics=[cam_topic]))

            offsets = []
            while reader.has_next():
                _, _, ts_ns = reader.read_next()   # do not deserialize — cheap
                offsets.append(ts_ns)

            self._frame_offsets = offsets
            self._indexed_path = self.rosbag_path
            self._cam_topic_resolved = cam_topic
            self._cam_type_name = type_map[cam_topic]
            self._storage_id = storage_id

            self.get_logger().info(
                f"Indexed {len(offsets)} frames on '{cam_topic}' "
                f"(storage={storage_id}) from {self.rosbag_path}"
            )
            self._publish_status(
                f"HD index ready: {len(offsets)} frames on {cam_topic}."
            )

    def _read_frame(self, frame_index: int):
        """
        Read, decode, and JPEG-encode the frame at frame_index.
        Returns a sensor_msgs/CompressedImage or None on miss.
        """
        self._ensure_index()

        with self._index_lock:
            offsets    = self._frame_offsets
            cam_topic  = self._cam_topic_resolved
            type_name  = self._cam_type_name
            storage_id = self._storage_id

        if not offsets or cam_topic is None:
            return None
        if frame_index < 0 or frame_index >= len(offsets):
            return None

        target_ts = offsets[frame_index]
        msg_type = get_message(type_name)

        reader = self._open_reader(storage_id)
        reader.set_filter(rosbag2_py.StorageFilter(topics=[cam_topic]))

        data = self._seek_and_read(reader, target_ts, frame_index, offsets)
        if data is None:
            return None

        msg = deserialize_message(data, msg_type)
        if hasattr(msg, "format"):
            # CompressedImage in the bag — decode to raw for a clean re-encode.
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        comp = self.bridge.cv2_to_compressed_imgmsg(cv_img, dst_format="jpeg")
        comp.format = "jpeg"
        return comp

    def _seek_and_read(self, reader, target_ts, frame_index, offsets):
        """
        Jump to target_ts via seek() and return the serialized payload of the
        message at frame_index. Falls back to a counted forward scan if seek()
        is unavailable or lands imprecisely (e.g. duplicate timestamps).
        """
        # Preferred: O(1)-ish seek to the timestamp, then read the next message.
        try:
            reader.seek(target_ts)
            if reader.has_next():
                _, data, ts_ns = reader.read_next()
                if ts_ns == target_ts:
                    return data
                # Timestamp landed off (duplicates/ordering) — fall through.
        except Exception:
            pass

        # Fallback: counted forward scan from the start on a fresh reader.
        reader2 = self._open_reader(self._storage_id)
        reader2.set_filter(rosbag2_py.StorageFilter(topics=[self._cam_topic_resolved]))
        counter = 0
        while reader2.has_next():
            _, data, _ = reader2.read_next()
            if counter == frame_index:
                return data
            counter += 1
        return None

    # ----------------------------------------------------------
    # ROSBAG HELPERS
    # ----------------------------------------------------------

    def _open_reader(self, storage_id: str):
        storage_options = rosbag2_py.StorageOptions(
            uri=self.rosbag_path,
            storage_id=storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def _resolve_camera_topic(self, type_map: dict):
        """Same fallback logic the indexer uses: raw topic, else /compressed."""
        cam_topic = self.camera_topic
        if cam_topic in type_map:
            return cam_topic
        compressed = cam_topic + "/compressed"
        if compressed in type_map:
            self.get_logger().warn(f"Raw camera not in bag — using: {compressed}")
            return compressed
        return None

    def _detect_storage_id(self, path: str) -> str:
        """
        Auto-detect the storage_id from the bag's metadata.yaml (like the
        indexer's config intends). Falls back to the rosbag_storage_id param.
        """
        meta_dir = path if os.path.isdir(path) else os.path.dirname(path)
        meta_path = os.path.join(meta_dir, "metadata.yaml")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    for line in f:
                        if "storage_identifier" in line:
                            value = line.split(":", 1)[1].strip().strip('"').strip("'")
                            if value:
                                return value
        except Exception as e:
            self.get_logger().warn(f"storage_id auto-detect failed: {e}")
        return self.fallback_storage

    # ----------------------------------------------------------
    # LRU CACHE
    # ----------------------------------------------------------

    def _cache_get(self, frame_index: int):
        with self._cache_lock:
            if frame_index in self._frame_cache:
                self._frame_cache.move_to_end(frame_index)
                return self._frame_cache[frame_index]
        return None

    def _cache_put(self, frame_index: int, comp: CompressedImage):
        with self._cache_lock:
            self._frame_cache[frame_index] = comp
            self._frame_cache.move_to_end(frame_index)
            while len(self._frame_cache) > self.cache_size:
                self._frame_cache.popitem(last=False)   # evict least-recently-used

    # ----------------------------------------------------------
    # ROSBAG PATH WATCHER
    # ----------------------------------------------------------

    def _refresh_rosbag_path(self):
        """
        Pick up rosbag_path changes made via:
          ros2 param set /hd_frame_server rosbag_path /new/path
        Rebuilds the index and clears caches when the bag changes.
        """
        new_path = self.get_parameter("rosbag_path").value
        if new_path != self.rosbag_path:
            self.rosbag_path = new_path
            with self._index_lock:
                self._frame_offsets = None
                self._indexed_path = None
                self._cam_topic_resolved = None
                self._cam_type_name = None
                self._storage_id = None
            with self._cache_lock:
                self._frame_cache.clear()
            self.get_logger().info(f"rosbag_path updated: {self.rosbag_path}")
            if self.rosbag_path:
                threading.Thread(target=self._ensure_index, daemon=True).start()

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _publish_status(self, text: str):
        msg = String()
        msg.data = f"[HdFrameServer] {text}"
        self.status_pub.publish(msg)


# ============================================================
# ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = HdFrameServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
