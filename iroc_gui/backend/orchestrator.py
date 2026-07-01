"""
orchestrator.py — Setup bring-up manager for the Anveshan GCS.

This is the engine behind the GUI's single **"Initiate Setup"** button. It replaces
the manual Jetson terminals (VSLAM → MAVROS → camera → stream → rosbag → semantic)
with one ordered, health-gated launch that the backend owns end-to-end:

    * every step is a child process WE own (own process group) → real stop/restart,
    * stdout+stderr are captured into a per-step ring buffer → live log tail in the UI,
    * each step is health-gated (a ROS topic appears / a TCP port opens / a settle
      delay elapses) before the next one starts → the panel shows honest status,
    * the whole thing is driven by config.json, so no command lives in the GUI code
      that the operator can't see and edit in the Config panel.

Design notes
------------
* We deliberately launch the steps directly (not via the team's ``run_system.sh``,
  which spawns detached ``gnome-terminal`` windows we could neither log nor stop).
  ``run_system.sh`` stays as the manual equivalent; ``mavros.run_system_sh`` in the
  config is kept for reference. The exact commands here mirror the ones the team
  runs by hand — see the per-builder comments.
* No rclpy / heavy imports: readiness is probed with the ``ros2`` CLI in a sourced
  shell, so importing this module is always safe (mock box included).
* Target is the Jetson (Linux). On POSIX we use a fresh session per child so a
  step's whole tree dies on stop; on Windows (dev box) we degrade gracefully.
"""

import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import archive
import config_store

# ══════════════════════════════════════════════════════════════════════════════
#  MAP OF THIS FILE (for later edits)
#  ─────────────────────────────────────────────────────────────────────────────
#  1. Helpers .............. shell prelude, topic/port utils         (_prelude ...)
#  2. Step ................. one service: command + readiness + state (class Step)
#  3. Command builders ..... ONE function per setup step; edit these  (_step_* )
#                            to change how a service is launched. They mirror the
#                            team's manual terminals — see each builder's comment.
#  4. Orchestrator ......... start / stop / restart the ordered chain (class Orchestrator)
#  5. manager .............. the singleton the backend imports
#
#  TO ADD A STEP:   write a _step_x(cfg, sel) → Step, register it in _BUILDERS,
#                   and add its key to config "setup.steps" (order matters).
#  TO REORDER/SKIP: just edit config "setup.steps" — no code change.
# ══════════════════════════════════════════════════════════════════════════════

POSIX = os.name == "posix"
LOG_LINES = 500                     # per-step ring-buffer depth
_IROC_GUI_DIR = config_store._IROC_GUI_DIR


def _expand(p) -> str:
    return os.path.expanduser(str(p)) if p else ""


def _bash(script: str) -> List[str]:
    """Wrap a shell snippet as a login-bash command (so `source` + PATH work)."""
    return ["bash", "-lc", script]


def _prelude(cfg: Dict) -> str:
    rid = cfg.get("network", {}).get("ros_domain_id", 1)
    ros = _expand(cfg.get("setup", {}).get("ros_setup", "/opt/ros/humble/setup.bash"))
    return f"export ROS_DOMAIN_ID={rid}; source {shlex.quote(ros)} 2>/dev/null; "


def _base_topic(topic: str) -> str:
    """`/image_raw/compressed` → `/image_raw` (the raw topic gscam/usb_cam publish)."""
    t = (topic or "").strip()
    for suf in ("/compressed", "/compressedDepth", "/theora"):
        if t.endswith(suf):
            return t[: -len(suf)]
    return t


def _video_port(cfg: Dict) -> int:
    url = cfg.get("network", {}).get("video_source_url", "http://127.0.0.1:8765")
    try:
        return urlparse(url).port or 8765
    except Exception:
        return 8765


# ── one setup step ────────────────────────────────────────────────────────────
class Step:
    """A single bring-up service: its command, how to tell it's ready, live state."""

    def __init__(self, key: str, name: str, cmd: List[str], *,
                 ready_topics: Optional[List[str]] = None,
                 ready_port: Optional[int] = None,
                 ready_timeout: int = 45, settle: int = 3,
                 optional: bool = False,
                 pre_spawn: Optional[Callable[[], None]] = None):
        self.key = key
        self.name = name
        self.cmd = cmd
        self.ready_topics = ready_topics or []
        self.ready_port = ready_port
        self.ready_timeout = ready_timeout
        self.settle = settle
        self.optional = optional
        # pre_spawn: side-effect run once, right before the process starts (e.g. the
        # rosbag step archives the previous bag here). Kept off the build path so
        # merely *constructing* a Step never touches the filesystem.
        self.pre_spawn = pre_spawn
        # runtime
        self.proc: Optional[subprocess.Popen] = None
        self.status = "idle"          # idle|starting|ready|running|failed|stopped
        self.error = ""
        self.started_at: Optional[float] = None
        self.log: deque = deque(maxlen=LOG_LINES)
        self._reader: Optional[threading.Thread] = None

    def to_dict(self, log_tail: int = 40) -> Dict:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "optional": self.optional,
            "pid": self.proc.pid if self.proc else None,
            "error": self.error,
            "ready_topics": self.ready_topics,
            "ready_port": self.ready_port,
            "uptime": (time.time() - self.started_at) if self.started_at else 0,
            "log": list(self.log)[-log_tail:],
        }

    # -- process lifecycle -----------------------------------------------------
    def spawn(self):
        self.error = ""
        self.log.clear()
        if self.pre_spawn:
            try:
                self.pre_spawn()
            except Exception as e:
                self._append(f"[pre] {e}")     # failsafe — never blocks the launch
        self._append(f"$ {self.cmd[-1] if self.cmd else ''}")
        kw = {}
        if POSIX:
            kw["start_new_session"] = True      # own process group → clean group-kill
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, cwd=_IROC_GUI_DIR, **kw)
        self.started_at = time.time()
        self.status = "starting"
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        try:
            for line in self.proc.stdout:            # blocks until EOF (proc exit)
                self._append(line.rstrip("\n"))
        except Exception:
            pass

    def _append(self, text: str):
        self.log.append({"ts": datetime.now().strftime("%H:%M:%S"), "text": text})

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 6.0):
        if not self.proc:
            return
        if self.alive():
            try:
                if POSIX:
                    pgid = os.getpgid(self.proc.pid)
                    os.killpg(pgid, signal.SIGINT)          # let ros2 shut down clean
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    if POSIX:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
                except Exception:
                    pass
        self.status = "stopped"
        self.started_at = None


# ══════════════════════════════════════════════════════════════════════════════
#  3. COMMAND BUILDERS  —  ONE PER SETUP STEP. Edit a step's command HERE.
#  ─────────────────────────────────────────────────────────────────────────────
#  Signature: _step_x(cfg, sel) -> Step, where
#     cfg = the full config.json (see backend/config_store.py)
#     sel = replay selection dict {rosbag_path, db_path, stamp} or None.
#           None  → AUTONOMOUS: live folder + fresh stamped archive .pt.
#           dict  → REPLAY: reuse a past mission's (bag, .pt) pair.
#           Only the "semantic" step reads sel; others accept it for a uniform call.
#  Each builder returns a bash -lc command that MIRRORS the manual terminal it
#  replaces — the mirrored line is quoted in the builder's comment.
# ══════════════════════════════════════════════════════════════════════════════
def _step_vslam(cfg, sel=None) -> Step:
    v = cfg.get("vslam", {})
    ws = _expand(v.get("workspace", "~/workspaces/isaac_ros-dev"))
    pkg = v.get("launch_pkg", "isaac_ros_visual_slam")
    lf = v.get("launch_file", "isaac_ros_visual_slam_realsense.launch.py")
    # manual equiv: cd ~/workspaces/isaac_ros-dev && source install/setup.bash &&
    #               ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py
    script = (_prelude(cfg) + f"cd {shlex.quote(ws)}; source install/setup.bash 2>/dev/null; "
              f"exec ros2 launch {shlex.quote(pkg)} {shlex.quote(lf)}")
    return Step("vslam", "Isaac ROS Visual SLAM", _bash(script),
                ready_topics=["/visual_slam/tracking/odometry"],
                ready_timeout=cfg.get("setup", {}).get("ready_timeout", 60), settle=3)


def _step_mavros(cfg, sel=None) -> Step:
    m = cfg.get("mavros", {})
    fcu = m.get("fcu_url", "serial:///dev/ttyACM0:921600")
    params = _expand(m.get("params_file", "/home/nidar/mavros_plugins.yaml"))
    # manual equiv: ros2 run mavros mavros_node --ros-args -p fcu_url:=... --params-file ...
    script = (_prelude(cfg) + f"exec ros2 run mavros mavros_node --ros-args "
              f"-p fcu_url:={shlex.quote(fcu)} --params-file {shlex.quote(params)}")
    return Step("mavros", "MAVROS (FCU link)", _bash(script),
                ready_topics=["/mavros/state"],
                ready_timeout=cfg.get("setup", {}).get("ready_timeout", 45), settle=2)


def _step_camera(cfg, sel=None) -> Step:
    c = cfg.get("camera", {})
    w, h, fr = c.get("width", 1280), c.get("height", 720), c.get("framerate", 10)
    if c.get("source", "csi") == "csi":
        # manual equiv: ros2 run gscam gscam_node --ros-args -p gscam_config:="nvargus..."
        gs = (f"nvarguscamerasrc sensor-id={c.get('sensor_id', 0)} ! "
              f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fr}/1 ! "
              f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert")
        script = (_prelude(cfg) + f"exec ros2 run gscam gscam_node --ros-args "
                  f"-p gscam_config:={shlex.quote(gs)} -p camera_name:=csi_cam")
    else:
        # manual equiv: ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=...
        ciu = c.get("camera_info_url", "")
        script = (_prelude(cfg) + f"exec ros2 run usb_cam usb_cam_node_exe --ros-args "
                  f"-p video_device:={shlex.quote(c.get('video_device', '/dev/video6'))} "
                  f"-p image_width:={w} -p image_height:={h} "
                  f"-p framerate:={float(fr)} "
                  f"-p pixel_format:={shlex.quote(c.get('pixel_format', 'yuyv2rgb'))}")
        if ciu:
            script += f" -p camera_info_url:={shlex.quote(ciu)}"
    return Step("camera", "Nadir camera", _bash(script),
                ready_topics=[_base_topic(c.get("topic", "/image_raw/compressed"))],
                ready_timeout=25, settle=2, optional=True)


def _step_stream(cfg, sel=None) -> Step:
    # LIVE VIDEO PATH: nadir feed reaches the GUI over HTTP (this MJPEG server),
    # NOT via rosbag. The same ROS frames still feed rosbag + semantic separately.
    c = cfg.get("camera", {})
    port = _video_port(cfg)
    stream_py = os.path.join(_IROC_GUI_DIR, "stream.py")
    # adopted, config-driven copy of the team's stream.py (rclpy → sourced python3)
    script = (_prelude(cfg) + f"exec python3 {shlex.quote(stream_py)} "
              f"--topic {shlex.quote(c.get('topic', '/image_raw/compressed'))} "
              f"--port {port}")
    return Step("stream", f"MJPEG stream :{port}", _bash(script),
                ready_port=port, ready_timeout=15, settle=1, optional=True)


def _step_rosbag(cfg, sel=None) -> Step:
    # Records THE mission bag into live_folder as rosbag_<STAMP> (image-processing
    # input). Before recording, any previous live bag is moved to rosbag_archives
    # in the background — see archive.archive_existing_bags (failsafe, low-latency,
    # instant rename when live/archive share a filesystem). Naming + locations are
    # all in config "rosbag" + "archive"; see backend/archive.py.
    rb = cfg.get("rosbag", {})
    folder = archive.live_folder(cfg)
    topics = rb.get("topics") or ["/image_raw"]
    stamp = archive.new_stamp(cfg)
    name = archive.rosbag_name(cfg, stamp) if rb.get("auto_name", True) \
        else _rosbag_prefix_static(cfg)
    topic_args = " ".join(shlex.quote(t) for t in topics)
    # manual equiv: cd <live_folder> && ros2 bag record -o rosbag_<STAMP> <topics...>
    script = (_prelude(cfg) + f"mkdir -p {shlex.quote(folder)}; cd {shlex.quote(folder)}; "
              f"exec ros2 bag record -o {shlex.quote(name)} {topic_args}")
    step = Step("rosbag", "Rosbag record (live)", _bash(script),
                ready_timeout=8, settle=3)
    step.pre_spawn = lambda: archive.archive_existing_bags(cfg, logger=None)
    return step


def _rosbag_prefix_static(cfg) -> str:
    """Fixed bag name when auto_name is off (rare — you overwrite each run)."""
    return archive._rosbag_prefix(cfg).rstrip("_") or "rosbag"


def _step_semantic(cfg, sel=None) -> Step:
    # IMAGE PROCESSING USES THE ROSBAG (never the live HTTP video).
    #   AUTONOMOUS (sel=None): index the CURRENT bag in live_folder, write/read the
    #     embedding at embeddings_archives/embddg_<STAMP>.pt (stamp inherited from
    #     the bag → the pair matches for later replay).
    #   REPLAY (sel set): reuse a past mission's archived (bag, .pt) pair verbatim.
    # Path resolution lives in backend/archive.py; only the ML knobs come from
    # config "semantic".
    s = cfg.get("semantic", {})
    py = _expand(s.get("python_exe", "python3"))
    ws = _expand(s.get("ros_ws", "~/anveshan_ws"))
    launch = os.path.join(_IROC_GUI_DIR, "launch", "gui_bringup.launch.py")

    if sel and sel.get("db_path"):
        rosbag_path, db_path = sel.get("rosbag_path"), sel["db_path"]
    else:
        auto = archive.autonomous_paths(cfg)
        rosbag_path, db_path = auto["rosbag_path"], auto["db_path"]
    # last-resort fallback so the step still builds before any bag exists
    if not db_path:
        db_path = _expand(s.get("db_path", "~/semantic_db/mission.pt"))

    def b(x):  # ROS launch wants lowercase booleans
        return str(x).lower() if isinstance(x, bool) else str(x)

    args = [
        f"python_exe:={shlex.quote(py)}",
        f"camera_topic:={shlex.quote(s.get('camera_topic', '/image_raw'))}",
        f"db_path:={shlex.quote(db_path)}",
        f"seeds_dir:={shlex.quote(_expand(s.get('seeds_dir', '~/anveshan_seeds')))}",
        f"threshold:={s.get('threshold', 0.0)}",
        f"display_threshold:={s.get('display_threshold', 0.57)}",
        f"top_k:={s.get('top_k', 50)}",
        f"downsample_seed:={b(s.get('downsample_seed', False))}",
        f"sample_rate:={s.get('sample_rate', 1)}",
        f"with_indexer:={b(s.get('with_indexer', True))}",
        f"with_retriever:={b(s.get('with_retriever', True))}",
        f"with_hd_server:={b(s.get('with_hd_server', True))}",
        "with_backend:=false", "with_frontend:=false", "with_dock:=false",
    ]
    if rosbag_path:
        args.append(f"rosbag_path:={shlex.quote(rosbag_path)}")
    # semantic nodes need the ML venv + the anveshan_ws overlay sourced first
    script = (_prelude(cfg) + f"source {shlex.quote(ws)}/install/setup.bash 2>/dev/null; "
              f"export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
              f"exec ros2 launch {shlex.quote(launch)} " + " ".join(args))
    return Step("semantic", "Semantic pipeline", _bash(script),
                ready_timeout=20, settle=3, optional=True)


_BUILDERS: Dict[str, Callable[[Dict], Step]] = {
    "vslam": _step_vslam,
    "mavros": _step_mavros,
    "camera": _step_camera,
    "stream": _step_stream,
    "rosbag": _step_rosbag,
    "semantic": _step_semantic,
}


# ══════════════════════════════════════════════════════════════════════════════
#  4. ORCHESTRATOR  —  runs the ordered chain; owns start/stop/restart + selection
# ══════════════════════════════════════════════════════════════════════════════
class Orchestrator:
    def __init__(self):
        self._lock = threading.Lock()
        self._steps: Dict[str, Step] = {}
        self._order: List[str] = []
        self._overall = "idle"       # idle|starting|running|failed|stopped
        self._chain: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._log_cb: Optional[Callable[[str, str, str], None]] = None
        self._topics_cache: tuple = (0.0, set())   # (ts, set-of-topics)
        # Mission source for the semantic step: None = AUTONOMOUS (live folder);
        # a {rosbag_path, db_path, stamp} dict = REPLAY a past archived mission.
        self._selected: Optional[Dict] = None

    def set_logger(self, cb):
        self._log_cb = cb

    def _emit(self, key: str, msg: str):
        if self._log_cb:
            try:
                self._log_cb("SYS", f"setup:{key}", msg)
            except Exception:
                pass

    # -- readiness probes ------------------------------------------------------
    def _list_topics(self, cfg) -> set:
        """`ros2 topic list` in a sourced shell, cached ~1.5s to avoid spamming."""
        now = time.time()
        if now - self._topics_cache[0] < 1.5:
            return self._topics_cache[1]
        topics: set = set()
        try:
            out = subprocess.run(
                _bash(_prelude(cfg) + "ros2 topic list"),
                capture_output=True, text=True, timeout=6)
            topics = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
        except Exception:
            topics = set()
        self._topics_cache = (now, topics)
        return topics

    @staticmethod
    def _port_open(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            return False

    def _wait_ready(self, step: Step, cfg) -> bool:
        """Block until the step is ready, dies, or times out. Returns True if ready."""
        deadline = time.time() + step.ready_timeout
        while time.time() < deadline:
            if self._abort.is_set():
                return False
            if not step.alive():
                step.error = "process exited during startup"
                return False
            if step.ready_topics:
                have = self._list_topics(cfg)
                if all(t in have for t in step.ready_topics):
                    return True
            elif step.ready_port:
                if self._port_open(step.ready_port):
                    return True
            else:
                # no explicit probe → ready once it has survived the settle window
                if step.started_at and time.time() - step.started_at >= step.settle:
                    return True
            time.sleep(1.0)
        # timed out: for probe-based steps, alive-but-not-detected is a soft pass
        if step.alive() and (step.ready_topics or step.ready_port):
            step.error = "ready probe timed out (process still running)"
            return step.optional      # optional → keep going; required → treat as fail
        return step.alive()

    # -- public API ------------------------------------------------------------
    def select(self, mission: Optional[Dict]) -> Dict:
        """Set the mission source for the semantic step.
        mission=None (or {}) → AUTONOMOUS (live folder). Otherwise a
        {rosbag_path, db_path, stamp} dict → REPLAY that archived pair. Takes
        effect on the next start()/restart('semantic')."""
        with self._lock:
            self._selected = mission or None
        label = (mission or {}).get("stamp", "live / autonomous")
        self._emit("semantic", f"mission source set → {label}")
        return {"selected": self._selected}

    def status(self) -> Dict:
        with self._lock:
            # refresh transient states (a running step that has since died)
            for st in self._steps.values():
                if st.status in ("ready", "running") and not st.alive():
                    st.status = "failed"
                    if not st.error:
                        st.error = "process exited"
            steps = [self._steps[k].to_dict() for k in self._order if k in self._steps]
            selected = self._selected
        return {"overall": self._overall, "steps": steps,
                "selected": selected,
                "mission_source": (selected or {}).get("stamp", "live")}

    def start(self) -> Dict:
        with self._lock:
            if self._chain and self._chain.is_alive():
                return {"started": False, "message": "setup already in progress"}
            cfg = config_store.load_config()
            order = cfg.get("setup", {}).get("steps") or list(_BUILDERS.keys())
            self._order = [k for k in order if k in _BUILDERS]
            self._steps = {}
            for k in self._order:
                try:
                    self._steps[k] = _BUILDERS[k](cfg, self._selected)
                except Exception as e:
                    s = Step(k, k, [], optional=True)
                    s.status = "failed"
                    s.error = f"build error: {e}"
                    self._steps[k] = s
            self._abort.clear()
            self._overall = "starting"
            self._chain = threading.Thread(
                target=self._run_chain, args=(cfg,), daemon=True)
            self._chain.start()
        self._emit("all", f"bring-up started ({', '.join(self._order)})")
        return {"started": True, "steps": self._order}

    def _run_chain(self, cfg):
        failed_required = False
        for key in self._order:
            if self._abort.is_set():
                break
            step = self._steps[key]
            if step.status == "failed":            # build error → skip
                continue
            self._emit(key, f"launching {step.name}")
            try:
                step.spawn()
            except Exception as e:
                step.status = "failed"
                step.error = f"spawn error: {e}"
                self._emit(key, f"FAILED to spawn: {e}")
                if not step.optional:
                    failed_required = True
                    break
                continue
            if self._wait_ready(step, cfg):
                step.status = "ready" if step.ready_topics or step.ready_port else "running"
                self._emit(key, f"{step.name} ready")
            else:
                step.status = "failed"
                self._emit(key, f"{step.name} failed: {step.error or 'not ready'}")
                if not step.optional:
                    failed_required = True
                    break
        with self._lock:
            if self._abort.is_set():
                self._overall = "stopped"
            elif failed_required:
                self._overall = "failed"
            else:
                self._overall = "running"
        self._emit("all", f"bring-up finished: {self._overall}")

    def stop(self) -> Dict:
        self._abort.set()
        with self._lock:
            steps = [self._steps[k] for k in reversed(self._order) if k in self._steps]
        for st in steps:
            if st.proc:
                self._emit(st.key, f"stopping {st.name}")
                st.stop()
        with self._lock:
            self._overall = "stopped"
        return {"stopped": True}

    def restart(self, key: str) -> Dict:
        with self._lock:
            if key not in _BUILDERS:
                return {"restarted": False, "message": f"unknown step: {key}"}
            cfg = config_store.load_config()
            old = self._steps.get(key)
            if old and old.proc:
                old.stop()
            step = _BUILDERS[key](cfg, self._selected)
            self._steps[key] = step
            if key not in self._order:
                self._order.append(key)
        self._emit(key, f"restarting {step.name}")

        def _one():
            try:
                step.spawn()
            except Exception as e:
                step.status = "failed"
                step.error = f"spawn error: {e}"
                return
            if self._wait_ready(step, cfg):
                step.status = "ready" if step.ready_topics or step.ready_port else "running"
            else:
                step.status = "failed"
        threading.Thread(target=_one, daemon=True).start()
        return {"restarted": True, "step": key}


# module-level singleton the backend imports
manager = Orchestrator()
