# IROC-u

This is the git repo for **IROC-u 2026**. It contains the ROS 2 implementation of an autonomous drone stack built around **PX4** + **MAVROS**, covering visual-inertial odometry bridging, autonomous offboard control, and vision-based precision landing.

> Some modules also include ArduPilot variants (kept alongside the PX4 versions for reference/comparison), but PX4 is the primary target flight stack for this project.

---

## Repository Layout

```
IROC-u/
├── iroc_ros_ws/                # ROS 2 workspace (colcon packages)
│   └── src/
│       ├── ros_interfaces/     # Custom service definitions (e.g. Takeoff.srv)
│       └── offboard_controller/# Offboard control node + interactive commanding node
│
├── vio/                        # Visual-Inertial Odometry → MAVROS bridges
│
├── precision_landing/          # Vision-based precision landing controllers
│   ├── px4/                    # PX4-specific landing node(s)
│   ├── aruco/                  # ArUco marker SVGs for printing tags
│   ├── evolution/               # Iteration history — earlier prototypes, kept for reference
│   └── *.py                    # Latest ArduPilot/PX4 landing + auto-tuning scripts
│
└── README.md
```

### `iroc_ros_ws/` — ROS 2 Workspace

A standard `colcon` workspace.

- **`ros_interfaces`** (`ament_cmake`): defines custom interfaces shared across nodes, currently `srv/Takeoff.srv` (altitude in, success/message out).
- **`offboard_controller`** (`ament_python`): wraps MAVROS arm/takeoff/land calls behind simple ROS 2 services (`/offboard/arm`, `/offboard/takeoff`, `/offboard/land`) and a position-command topic (`/offboard/position_cmd`), so higher-level mission logic doesn't need to talk to MAVROS directly.
  - `offboard_node.py` — the service/topic server.
  - `commanding_node.py` — a simple interactive CLI client for manually arming/taking off/landing.
  - `px4_offboard_node.py` — a minimal standalone PX4 OFFBOARD-mode demo (position setpoint streaming, arm, mode switch).

Build it with the standard ROS 2 workflow (see Quick Start below).

### `vio/` — VIO → MAVROS Bridges

Takes odometry from a visual-inertial source (e.g. an Intel RealSense tracking camera, or an external VIO pipeline) and republishes it on the topic and in the convention each flight stack expects:

- `simple_vio_ros2_bridge_tracking_camera.py` — relays RealSense `/camera/odom/sample` straight to `/mavros/odometry/out` and emits a `CompanionProcessStatus` heartbeat. Simplest case, used when the camera's frame convention already matches what MAVROS expects.
- `vio_bridge_ardupilot.py` — looks up a TF transform between configurable world/base frames, applies camera mounting-offset rotations, and publishes a `PoseStamped` to `mavros/vision_pose/pose` (ArduPilot's external vision pose topic).
- `vio_bridge_px4.py` / `vio_bridge_px4_reference.py` — convert VIO odometry into the frame/axis convention PX4 expects (ENU→NED, body-frame velocity rotation, 6×6 covariance rotation) and publish to `/mavros/odometry/out`, also broadcasting a TF and the companion-process heartbeat.

### `precision_landing/` — Vision-Based Precision Landing

Multiple control approaches were tried as the project evolved; each script is self-contained so they can be compared or run independently.

**Current scripts (top level):**
- `px4_landing.py` / `px4/px4_landing_stable_takeoff.py` — PX4 OFFBOARD precision landing using **AprilTag** detections looked up via TF2, with a PD controller on tag offset and altitude-based descent gating. The `_stable_takeoff` variant climbs via a streamed position setpoint instead of a raw velocity ramp.
- `landing_pose_ardupilot.py` / `drone_stable_test.py` / `optical_flow_landing.py` — ArduPilot (GUIDED mode) equivalents, each refining hover stability, centering-before-descent logic, or fixed hover durations.
- `auto_tuning.py` — an in-flight auto-tuner that sweeps Kp/Kd/Ki gains for the X/Y landing controller against live tag-tracking error, used to find usable PID gains before locking them in for an actual mission run.
- `aruco/` — printable ArUco marker SVGs (6×6, IDs 0–3) used as landing targets.
- `cmd.txt` — reference launch commands for RealSense, AprilTag detection, and MAVROS.

**`evolution/`** — kept intentionally, since each file represents a different stage/approach rather than dead code:
- `rpi_cam_detection_node.py`, `sitl_marker_detection.py`, `marker_detection_node.py` — OpenCV ArUco detectors at different stages (webcam, SITL, ROS `Image` topic) publishing raw pixel error on `/aruco_error`.
- `precision_landing_test.py`, `precision_landing_px4.py` — early PX4 OFFBOARD landing state machines driven by pixel error rather than TF/metric error.
- `precision_landing_ardupilot.py`, `pixel_based_precision_landing.py`, `ardupilot_velocity_test.py` — ArduPilot GUIDED-mode landing using pixel error and velocity setpoints, progressively adding stable-hover timers and tuned gains.
- `landing_pose_ardupilot.py` — switches from pixel error to **TF-based metric pose** error (via `april_ros`) with a full PID + low-altitude hover-and-confirm step before switching to LAND.

In short: the `evolution/` scripts trace the path from *pixel-error + webcam* → *TF/metric-error + AprilTag + auto-tuned PID*, which is what the top-level scripts now use.

---

## Quick Start

### Prerequisites
- ROS 2 (Humble or newer)
- MAVROS + a PX4 (or ArduPilot) flight controller/SITL
- `apriltag_ros` (for AprilTag-based landing) and/or OpenCV with `cv2.aruco` (for ArUco-based scripts)
- A depth/tracking camera publishing odometry (e.g. Intel RealSense) if using the VIO bridges

### Build the ROS 2 workspace
```bash
cd iroc_ros_ws
colcon build
source install/setup.bash
```

### Typical run order
```bash
# 1. Bring up the camera + tag detector + MAVROS (see precision_landing/cmd.txt)
ros2 launch realsense2_camera rs_launch.py unite_imu_method:=2 enable_accel:=true enable_gyro:=true enable_infra1:=true enable_infra2:=true
ros2 run apriltag_ros apriltag_node --ros-args -r image_rect:=/camera/camera/color/image_raw -r camera_info:=/camera/camera/color/camera_info --params-file <path-to-tag-config>.yaml
ros2 launch mavros px4.launch fcu_url:="<your-fcu-url>"

# 2. Bridge VIO into MAVROS
ros2 run vio vio_bridge_px4.py

# 3. Run the precision landing mission
ros2 run precision_landing px4_landing.py
```

Adjust topic names, tag config paths, and FCU URLs to match your hardware setup.

---

## Notes

- Parameters such as `takeoff_alt`, `kp`/`kd`, `max_vel_xy`, and `land_alt_threshold` are exposed as ROS 2 parameters on each landing node — tune them via a YAML params file or `--ros-args -p`.
- `auto_tuning.py` is intended to be run first on a new airframe/camera setup to find reasonable PID gains before flying the fixed-gain landing scripts.
