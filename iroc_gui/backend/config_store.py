"""
config_store.py — persisted GUI configuration for the Anveshan GCS.

One JSON file (`iroc_gui/config.json`) is the single operator-editable config the
Config panel reads/writes and the Setup orchestrator consumes. Load precedence
(lowest → highest):

    _FALLBACK (this file)  ←  config.default.json (committed template)  ←  config.json (local, runtime)

`config.json` is git-ignored: it holds machine-specific paths/IPs and is written
on the first save, so pulling repo updates never conflicts with a team member's
local edits. `config.default.json` is the committed starting point.

No ROS / heavy deps here on purpose — importing this is always safe.
"""

import json
import os
import threading
from typing import Any, Dict

_HERE          = os.path.dirname(os.path.abspath(__file__))
_IROC_GUI_DIR  = os.path.dirname(_HERE)            # iroc_gui/
CONFIG_PATH    = os.path.join(_IROC_GUI_DIR, "config.json")
DEFAULT_PATH   = os.path.join(_IROC_GUI_DIR, "config.default.json")

_lock = threading.Lock()

# Safety net if config.default.json is missing. Keep in sync with that file;
# the file (when present) overrides this, so the file is the real source of truth.
_FALLBACK: Dict[str, Any] = {
    "network": {
        "ros_domain_id": 1, "backend_port": 5000, "frontend_port": 8080,
        "video_source_url": "http://127.0.0.1:8765", "command_token": "",
    },
    "camera": {
        "topic": "/image_raw/compressed", "source": "csi",
        "width": 1280, "height": 720, "framerate": 10,
        "sensor_id": 0, "video_device": "/dev/video6",
        "pixel_format": "yuyv2rgb",
        "camera_info_url": "file:///home/nidar/IROC/calibrationdata/ost_hd.yaml",
    },
    "mavros": {
        "fcu_url": "serial:///dev/ttyACM0:921600",
        "params_file": "/home/nidar/mavros_plugins.yaml",
        "run_system_sh": "~/Desktop/run_system.sh",
    },
    "vslam": {
        "workspace": "~/workspaces/isaac_ros-dev",
        "launch_pkg": "isaac_ros_visual_slam",
        "launch_file": "isaac_ros_visual_slam_realsense.launch.py",
    },
    "semantic": {
        "python_exe": "/home/nidar/seed_searcher_naive/img_p_new/bin/python3",
        "ros_ws": "~/anveshan_ws", "rosbag_path": "",
        "db_path": "~/semantic_db/mission.pt", "seeds_dir": "/home/nidar/anveshan_seeds",
        "camera_topic": "/image_raw", "threshold": 0.0, "display_threshold": 0.57,
        "top_k": 50, "downsample_seed": False, "sample_rate": 1,
        "with_indexer": True, "with_retriever": True, "with_hd_server": True,
    },
    "rosbag": {
        "live_folder": "~/IROC/live_rosbags",
        "topics": ["/image_raw", "/visual_slam/tracking/odometry", "/mavros/local_position/odom"],
        "auto_name": True,
    },
    "docking": {
        "enabled": False, "gcs_host": "192.168.0.114", "gcs_port": 55555,
        "settle_delay": 30.0, "charge_mah": 260, "auto_dock": True,
    },
    "mission": {
        "script_path": "~/lawnmover3.py", "takeoff_alt": 1.0, "hover_seconds": 5.0,
        "wp_accept_radius": 0.2, "land_z_threshold": 0.05,
    },
    "setup": {
        "ros_setup": "/opt/ros/humble/setup.bash",
        "steps": ["vslam", "mavros", "camera", "stream", "rosbag", "semantic"],
        "ready_timeout": 45, "settle_delay": 3,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge — override wins; nested dicts merged, scalars/lists replaced."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _coerce(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Light, forgiving validation — clamp the values where a bad number is unsafe."""
    sem = cfg.get("semantic")
    if isinstance(sem, dict):
        for key in ("threshold", "display_threshold"):
            try:
                sem[key] = max(0.0, min(1.0, float(sem[key])))
            except (KeyError, TypeError, ValueError):
                pass
    return cfg


def _base() -> Dict[str, Any]:
    """Defaults layer: _FALLBACK overlaid with config.default.json (if present)."""
    return _deep_merge(_FALLBACK, _read_json(DEFAULT_PATH))


def load_config() -> Dict[str, Any]:
    """Full effective config: defaults ← config.json. Always returns every key."""
    with _lock:
        return _deep_merge(_base(), _read_json(CONFIG_PATH))


def save_config(incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `incoming` over the current effective config and persist to config.json.
    Returns the merged result. Unknown keys are kept (forward-compatible)."""
    if not isinstance(incoming, dict):
        raise ValueError("config payload must be a JSON object")
    with _lock:
        current = _deep_merge(_base(), _read_json(CONFIG_PATH))
        merged  = _coerce(_deep_merge(current, incoming))
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        os.replace(tmp, CONFIG_PATH)   # atomic write
        return merged
