#!/usr/bin/env python3
"""
Autonomous Precision Landing Node — PX4 + MAVROS + ROS2
Mission flow:
  1. Wait for /mission_initiate (Int32 == 1)
  2. Wait for FCU connection
  3. Pre-stream setpoints (required by PX4 before OFFBOARD transition)
  4. Set mode → OFFBOARD
  5. Arm
  6. Takeoff via PoseStamped position setpoint (holds x/y, commands target z)
  7. Hover for N seconds
  8. Precision landing loop (PD controller → tag via TF2)
  9. Switch to LAND mode when close enough to ground
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time

from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
)

from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from apriltag_msgs.msg import AprilTagDetectionArray
from std_msgs.msg import Int32

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class AutonomousPrecisionLanding(Node):

    def __init__(self):
        super().__init__('autonomous_precision_landing')

        # ================== QOS ==================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ================== STATE ==================
        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = 0.0

        # Tag tracking
        self.tag_frame = ""
        self.err_x = 0.0
        self.err_y = 0.0
        self.err_z = 0.0

        # PD memory
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        # Mission trigger
        self.mission_initiated = False

        # ================== PARAMETERS ==================
        self.declare_parameter("takeoff_alt", 2.0)       # metres (relative)
        self.declare_parameter("hover_seconds", 5.0)     # seconds to hover before landing
        self.declare_parameter("kp", 0.5)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("max_vel_xy", 0.3)        # m/s horizontal clamp
        self.declare_parameter("land_alt_threshold", 0.3)  # relative alt to trigger LAND

        # ================== TF2 ==================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame = "camera_color_optical_frame"

        # ================== SUBSCRIPTIONS ==================
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos
        )
        self.create_subscription(
            AprilTagDetectionArray, '/detections', self.aruco_cb, qos_profile_sensor_data
        )
        self.create_subscription(Int32, '/mission_initiate', self.mission_initiate_cb, 10)

        # ================== PUBLISHERS ==================
        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10
        )
        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10
        )

        # ================== SERVICE CLIENTS ==================
        self.arm_srv  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode,     '/mavros/set_mode')

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def mission_initiate_cb(self, msg):
        if msg.data == 1 and not self.mission_initiated:
            self.get_logger().info("Mission Initiation Received!")
            self.mission_initiated = True

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.local_pos = msg

    def aruco_cb(self, msg):
        if len(msg.detections) > 0:
            det    = msg.detections[0]
            tag_id = det.id[0] if hasattr(det.id, '__iter__') else det.id
            self.tag_frame = f"tag25h9:{tag_id}"
        else:
            self.tag_frame = ""

    # ------------------------------------------------------------------ #
    # Service helpers
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> bool:
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            self.get_logger().error(f"set_mode({mode}) service call failed")
            return False
        return result.mode_sent

    def arm(self, status: bool) -> bool:
        req = CommandBool.Request()
        req.value = status
        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            self.get_logger().error("arm() service call failed")
            return False
        return result.success

    # ------------------------------------------------------------------ #
    # Velocity helper
    # ------------------------------------------------------------------ #
    def _zero_vel(self) -> TwistStamped:
        vel = TwistStamped()
        vel.header.stamp    = self.get_clock().now().to_msg()
        vel.header.frame_id = "base_link"
        return vel

    def publish_vel(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        vel = self._zero_vel()
        vel.twist.linear.x  = float(vx)
        vel.twist.linear.y  = float(vy)
        vel.twist.linear.z  = float(vz)
        vel.twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(vel)

    def publish_pose(self, x=0.0, y=0.0, z=0.0):
        """Publish a local NED position setpoint (x/y hold current, z = target)."""
        sp = PoseStamped()
        sp.header.stamp    = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x = float(x)
        sp.pose.position.y = float(y)
        sp.pose.position.z = float(z)
        # Hold current yaw (quaternion identity → no rotation)
        sp.pose.orientation.w = 1.0
        self.pos_pub.publish(sp)

    # ------------------------------------------------------------------ #
    # Main mission
    # ------------------------------------------------------------------ #
    def run_mission(self):

        # ---- 0. Wait for trigger ----------------------------------------
        self.get_logger().info("Waiting for mission initiation on /mission_initiate …")
        while rclpy.ok() and not self.mission_initiated:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Mission Started!")

        # ---- 1. FCU connection ------------------------------------------
        self.get_logger().info("Waiting for FCU connection …")
        while rclpy.ok() and not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("FCU connected.")

        # ---- 2. Wait for a valid local position -------------------------
        while rclpy.ok() and self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initial_alt = self.local_pos.pose.position.z
        self.get_logger().info(f"Initial altitude: {self.initial_alt:.3f} m")

        # ---- 3. Pre-stream setpoints (PX4 OFFBOARD requirement) ---------
        # PX4 requires at least ~2 s of setpoint stream before accepting OFFBOARD.
        # We pre-stream the takeoff target pose so the transition is seamless.
        target_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        target_abs_alt = self.initial_alt + target_alt_rel
        target_x = self.local_pos.pose.position.x
        target_y = self.local_pos.pose.position.y

        self.get_logger().info(
            f"Pre-streaming takeoff setpoint ({target_x:.2f}, {target_y:.2f}, "
            f"{target_abs_alt:.2f} m) for OFFBOARD …"
        )
        pre_stream_start = time.time()
        while rclpy.ok() and (time.time() - pre_stream_start) < 2.0:
            self.publish_pose(x=target_x, y=target_y, z=target_abs_alt)
            rclpy.spin_once(self, timeout_sec=0.05)

        # ---- 4. Switch to OFFBOARD --------------------------------------
        self.get_logger().info("Requesting OFFBOARD mode …")
        for _ in range(5):
            if self.set_mode("OFFBOARD"):
                break
            time.sleep(0.5)

        if self.state.mode != "OFFBOARD":
            self.get_logger().warn(
                f"Mode may not have switched. Current: {self.state.mode}"
            )

        # ---- 5. Arm -----------------------------------------------------
        self.get_logger().info("Arming …")
        for _ in range(5):
            if self.arm(True):
                break
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)

        # ---- 6. Takeoff via position setpoint ---------------------------
        self.get_logger().info(
            f"Climbing to {target_alt_rel:.1f} m relative "
            f"(absolute z = {target_abs_alt:.2f} m) …"
        )

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            # Keep re-asserting OFFBOARD
            if self.state.mode != "OFFBOARD":
                self.set_mode("OFFBOARD")

            self.publish_pose(x=target_x, y=target_y, z=target_abs_alt)

            if self.local_pos.pose.position.z >= target_abs_alt - 0.15:
                break

        self.get_logger().info("Target altitude reached.")

        # ---- 7. Hover for N seconds -------------------------------------
        hover_secs = self.get_parameter("hover_seconds").get_parameter_value().double_value
        self.get_logger().info(f"Hovering for {hover_secs:.1f} s …")
        hover_start = time.time()
        while rclpy.ok() and (time.time() - hover_start) < hover_secs:
            self.publish_pose(x=target_x, y=target_y, z=target_abs_alt)
            rclpy.spin_once(self, timeout_sec=0.05)

        # ---- 8. Precision landing ----------------------------------------
        self.precision_landing_loop()

    # ------------------------------------------------------------------ #
    # Precision Landing  (ported 1-to-1 from your ArduPilot version)
    # ------------------------------------------------------------------ #
    def precision_landing_loop(self):
        self.get_logger().info("Starting Precision Landing")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            # Keep OFFBOARD alive
            if self.state.mode != "OFFBOARD":
                self.set_mode("OFFBOARD")

            found_now = False

            if self.tag_frame:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.camera_frame,
                        self.tag_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.02)
                    )

                    self.err_x = trans.transform.translation.x
                    self.err_y = trans.transform.translation.y
                    self.err_z = trans.transform.translation.z
                    found_now  = True

                except (LookupException, ConnectivityException, ExtrapolationException):
                    found_now = False

            vel = TwistStamped()
            vel.header.stamp    = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link"

            if found_now:
                now = time.time()
                dt  = max(now - self.prev_time, 0.05)

                kp = self.get_parameter("kp").value
                kd = self.get_parameter("kd").value

                vx = -(kp * self.err_x) + (kd * (self.err_x - self.prev_err_x) / dt)
                vy =  (kp * self.err_y) + (kd * (self.err_y - self.prev_err_y) / dt)

                error_mag = np.sqrt(self.err_x**2 + self.err_y**2)
                vz = -0.15 if error_mag < 0.1 else 0.0

                max_v = self.get_parameter("max_vel_xy").value

                vel.twist.linear.x = float(np.clip(vx, -max_v, max_v))
                vel.twist.linear.y = float(np.clip(vy, -max_v, max_v))
                vel.twist.linear.z = float(vz)

                self.prev_err_x = self.err_x
                self.prev_err_y = self.err_y
                self.prev_time  = now

            else:
                vel.twist.linear.x = 0.0
                vel.twist.linear.y = 0.0
                vel.twist.linear.z = 0.0

            self.vel_pub.publish(vel)

            # Landing condition — same threshold logic as original
            alt_rel = self.local_pos.pose.position.z - self.initial_alt
            if alt_rel < self.get_parameter("land_alt_threshold").value:
                self.get_logger().info("Landing threshold reached — switching to AUTO.LAND")
                self.set_mode("AUTO.LAND")
                break


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main(args=None):
    rclpy.init(args=args)
    node = AutonomousPrecisionLanding()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()