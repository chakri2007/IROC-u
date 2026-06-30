#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time

from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy
)

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Int32


class AutonomousWaypointNav(Node):

    def __init__(self):
        super().__init__('autonomous_waypoint_nav')

        # ================== QOS ==================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ================== STATE ==================
        self.state      = State()
        self.local_pos  = PoseStamped()
        self.initial_alt = 0.0

        # Mission trigger
        self.mission_initiated = True

        # ================== PARAMETERS ==================
        self.declare_parameter("takeoff_alt",        1.0)   # metres (relative)
        self.declare_parameter("hover_seconds",      5.0)   # hover before waypoints
        self.declare_parameter("wp_accept_radius",   0.2)   # metres — waypoint hit radius
        self.declare_parameter("land_z_threshold",   0.05)  # metres — "on-ground" threshold

        # ================== WAYPOINTS ==================
        # Define your waypoints as (x, y, z_relative) in the local ENU frame.
        # z is RELATIVE to takeoff altitude (i.e. 0 = takeoff height).
        # The return-to-home and descent are handled automatically after these.
        self.waypoints = [
            ( 1.0,  0.0,  0.0),   # WP-1 : 2 m forward
            ( 1.0,  1.0,  0.0),   # WP-2 : 2 m right
            ( 0.0,  1.0,  0.0),   # WP-3 : back to x=0
            ( 0.0,  0.0,  0.0),   # WP-4 : back to y=0
        ]

        # ================== SUBSCRIPTIONS ==================
        self.create_subscription(State,       '/mavros/state',               self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb,   qos)
        self.create_subscription(Int32,       '/mission_initiate',            self.mission_initiate_cb, 10)

        # ================== PUBLISHERS ==================
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
            self.get_logger().info("Mission initiation received!")
            self.mission_initiated = True

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.local_pos = msg

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
    # Position setpoint helpers
    # ------------------------------------------------------------------ #
    def _make_pos(self, x=0.0, y=0.0, z=0.0) -> PoseStamped:
        """Build a PoseStamped setpoint in the local ENU frame."""
        sp = PoseStamped()
        sp.header.stamp    = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x = float(x)
        sp.pose.position.y = float(y)
        sp.pose.position.z = float(z)
        # Keep default quaternion (identity = no yaw change)
        sp.pose.orientation.w = 1.0
        return sp

    def publish_pos(self, x=0.0, y=0.0, z=0.0):
        self.pos_pub.publish(self._make_pos(x, y, z))

    def _keep_offboard(self):
        """Re-assert OFFBOARD if the FCU dropped out of it."""
        if self.state.mode != "OFFBOARD":
            self.get_logger().warn("OFFBOARD lost — re-requesting …")
            self.set_mode("OFFBOARD")

    def _dist_xy(self, tx, ty) -> float:
        dx = tx - self.local_pos.pose.position.x
        dy = ty - self.local_pos.pose.position.y
        return float(np.sqrt(dx**2 + dy**2))

    def _dist_3d(self, tx, ty, tz) -> float:
        dx = tx - self.local_pos.pose.position.x
        dy = ty - self.local_pos.pose.position.y
        dz = tz - self.local_pos.pose.position.z
        return float(np.sqrt(dx**2 + dy**2 + dz**2))

    # ------------------------------------------------------------------ #
    # Main mission
    # ------------------------------------------------------------------ #
    def run_mission(self):

        # ---- 0. Wait for trigger ----------------------------------------
        self.get_logger().info("Waiting for mission initiation on /mission_initiate …")
        while rclpy.ok() and not self.mission_initiated:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Mission started!")

        # ---- 1. FCU connection ------------------------------------------
        self.get_logger().info("Waiting for FCU connection …")
        while rclpy.ok() and not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("FCU connected.")

        # ---- 2. Wait for a valid local position -------------------------
        while rclpy.ok() and self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initial_alt = self.local_pos.pose.position.z
        home_x = self.local_pos.pose.position.x
        home_y = self.local_pos.pose.position.y
        self.get_logger().info(
            f"Home position locked — "
            f"({home_x:.3f}, {home_y:.3f}, {self.initial_alt:.3f}) m"
        )

        # ---- 3. Pre-stream setpoints (PX4 OFFBOARD requirement) ---------
        takeoff_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        target_abs_alt  = self.initial_alt + takeoff_alt_rel

        self.get_logger().info("Pre-streaming position setpoints for 2 s …")
        pre_stream_start = time.time()
        while rclpy.ok() and (time.time() - pre_stream_start) < 2.0:
            # Hold current XY, command takeoff altitude already so the
            # switch to OFFBOARD immediately starts climbing.
            self.publish_pos(x=home_x, y=home_y, z=target_abs_alt)
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # ---- Wait 10 seconds in OFFBOARD before arming ----
        self.get_logger().info(
            "OFFBOARD active. Waiting 10 seconds before arming..."
        )

        wait_start = time.time()
        while rclpy.ok() and (time.time() - wait_start) < 10.0:
            # Keep publishing setpoints so OFFBOARD stays active
            self.publish_pos(
                x=home_x,
                y=home_y,
                z=target_abs_alt
            )
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info("Delay complete. Arming now.")

        # ---- 4. Switch to OFFBOARD --------------------------------------
        self.get_logger().info("Requesting OFFBOARD mode …")
        for _ in range(5):
            if self.set_mode("OFFBOARD"):
                self.get_logger().info("OFFBOARD mode set.")
                break
            time.sleep(0.5)
        else:
            self.get_logger().warn(f"Mode may not have switched — current: {self.state.mode}")

        # ---- 5. Arm -----------------------------------------------------
        self.get_logger().info("Arming vehicle …")
        for _ in range(5):
            if self.arm(True):
                self.get_logger().info("Vehicle armed.")
                break
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)

        # ---- 6. Takeoff -------------------------------------------------
        self.get_logger().info(
            f"[TAKEOFF] Climbing to {takeoff_alt_rel:.1f} m relative "
            f"({target_abs_alt:.2f} m absolute) …"
        )

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self._keep_offboard()
            self.publish_pos(x=home_x, y=home_y, z=target_abs_alt)

            if self.local_pos.pose.position.z >= target_abs_alt - 0.1:
                break

        self.get_logger().info(
            f"[TAKEOFF] Target altitude reached: "
            f"{self.local_pos.pose.position.z:.2f} m"
        )

        # ---- 7. Pre-waypoint hover --------------------------------------
        hover_secs = self.get_parameter("hover_seconds").get_parameter_value().double_value
        self.get_logger().info(f"[HOVER] Hovering for {hover_secs:.1f} s before waypoints …")
        hover_start = time.time()
        while rclpy.ok() and (time.time() - hover_start) < hover_secs:
            self.publish_pos(x=home_x, y=home_y, z=target_abs_alt)
            rclpy.spin_once(self, timeout_sec=0.05)

        # ---- 8. Waypoint navigation --------------------------------------
        self.navigate_waypoints()

        # ---- 9. Return to home (0, 0) at takeoff altitude ---------------
        self.return_to_home()

        # ---- 10. Descend and land at (0, 0, 0) --------------------------
        self.descend_and_land()

    # ------------------------------------------------------------------ #
    # Waypoint Navigation
    # ------------------------------------------------------------------ #
    def navigate_waypoints(self):
        """Fly through each waypoint in self.waypoints sequentially."""

        accept_radius   = self.get_parameter("wp_accept_radius").get_parameter_value().double_value
        takeoff_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        total_wps       = len(self.waypoints)

        self.get_logger().info(
            f"[WAYPOINTS] Starting waypoint navigation — {total_wps} waypoint(s) loaded."
        )

        for idx, (wp_x, wp_y, wp_z_rel) in enumerate(self.waypoints):

            # Absolute z target
            wp_z_abs = self.initial_alt + takeoff_alt_rel + wp_z_rel

            self.get_logger().info(
                f"[WP {idx+1}/{total_wps}] Navigating to "
                f"x={wp_x:.2f} m, y={wp_y:.2f} m, z_rel={wp_z_rel:+.2f} m  "
                f"(abs z={wp_z_abs:.2f} m)"
            )

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                self._keep_offboard()

                # Publish position setpoint — FCU handles velocity/accel internally
                self.publish_pos(x=wp_x, y=wp_y, z=wp_z_abs)

                dist = self._dist_xy(wp_x, wp_y)

                if dist < accept_radius:
                    self.get_logger().info(
                        f"[WP {idx+1}/{total_wps}] Reached! "
                        f"(xy error: {dist:.3f} m)"
                    )
                    break

                # Progress log every ~2 s
                if not hasattr(self, '_last_wp_log') or \
                        time.time() - self._last_wp_log > 2.0:
                    cur = self.local_pos.pose.position
                    self.get_logger().info(
                        f"[WP {idx+1}/{total_wps}]  "
                        f"pos=({cur.x:.2f}, {cur.y:.2f}, {cur.z:.2f})  "
                        f"xy_dist={dist:.2f} m"
                    )
                    self._last_wp_log = time.time()

        self.get_logger().info("[WAYPOINTS] All waypoints completed!")

    # ------------------------------------------------------------------ #
    # Return to Home  (0, 0 at takeoff altitude)
    # ------------------------------------------------------------------ #
    def return_to_home(self):

        accept_radius   = self.get_parameter("wp_accept_radius").get_parameter_value().double_value
        takeoff_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        home_z_abs      = self.initial_alt + takeoff_alt_rel

        self.get_logger().info(
            f"[RTH] Returning to home (0.0, 0.0) at "
            f"z_abs={home_z_abs:.2f} m …"
        )

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self._keep_offboard()

            self.publish_pos(x=0.0, y=0.0, z=home_z_abs)

            dist = self._dist_xy(0.0, 0.0)

            if dist < accept_radius:
                self.get_logger().info(
                    f"[RTH] Home position reached. "
                    f"(xy error: {dist:.3f} m)"
                )
                break

            if not hasattr(self, '_last_rth_log') or \
                    time.time() - self._last_rth_log > 2.0:
                cur = self.local_pos.pose.position
                self.get_logger().info(
                    f"[RTH]  pos=({cur.x:.2f}, {cur.y:.2f}, {cur.z:.2f})  "
                    f"xy_dist={dist:.2f} m"
                )
                self._last_rth_log = time.time()

    # ------------------------------------------------------------------ #
    # Descend and Land at (0, 0, initial_alt → 0)
    # ------------------------------------------------------------------ #
    def descend_and_land(self):

        self.get_logger().info("[LAND] Switching to AUTO.LAND...")

        # Hold position for a short time before changing mode
        hold_start = time.time()
        while rclpy.ok() and (time.time() - hold_start) < 2.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.publish_pos(
                x=0.0,
                y=0.0,
                z=self.local_pos.pose.position.z
            )

        # Switch to AUTO.LAND
        if self.set_mode("AUTO.LAND"):
            self.get_logger().info("[LAND] AUTO.LAND enabled.")
        else:
            self.get_logger().error("[LAND] Failed to enter AUTO.LAND.")
            return

        # Wait until the vehicle disarms
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)

            if not self.state.armed:
                self.get_logger().info("[LAND] Vehicle disarmed.")
                break

        self.get_logger().info("[LAND] Mission complete.")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main(args=None):
    rclpy.init(args=args)
    node = AutonomousWaypointNav()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
