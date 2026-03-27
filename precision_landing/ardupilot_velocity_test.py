#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import time
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import Float32MultiArray


GUIDED_MODE = "GUIDED"
CLIMB_HEIGHT = 1.2
CLIMB_RATE = 0.5      # m/s upward (positive Z = up, consistent with your descent sign)
HOVER_TIME = 5.0


class PrecisionLandingNode(Node):

    def __init__(self):
        super().__init__("precision_landing_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = None

        # Vision data
        self.detected = 0.0
        self.err_x = 0.0
        self.err_y = 0.0

        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        # Parameters
        self.kp = 0.004
        self.kd = 0.0
        self.max_vel_xy = 0.45
        self.threshold = 15.0
        self.descent_speed = 0.05
        self.stable_hover_timer_started = False
        self.stable_hover_time = 0.0

        # Subscribers
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_cb, qos)
        self.create_subscription(Float32MultiArray, '/aruco_error', self.aruco_cb, 10)

        # Publisher - velocity setpoint (same topic you already use)
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # Services (kept for arm/mode; takeoff service is no longer used)
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        for srv in [self.arm_srv, self.mode_srv, self.takeoff_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for MAVROS services...")

    # -------------------- New helper: velocity setpoint (exactly like your ROS1 example) --------------------
    def publish_velocity(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        """Publish velocity in body frame with proper header (this was the missing piece)."""
        vel = TwistStamped()
        vel.header.stamp = self.get_clock().now().to_msg()
        vel.header.frame_id = "base_link"          # Body frame (Forward-Right-?, exactly as in your snippet)
        vel.twist.linear.x = vx
        vel.twist.linear.y = vy
        vel.twist.linear.z = vz
        self.vel_pub.publish(vel)

    # -------------------- Callbacks --------------------

    def state_cb(self, msg):
        self.state = msg

    def local_pos_cb(self, msg):
        self.local_pos = msg

    def aruco_cb(self, msg):
        self.detected = msg.data[0]
        self.err_x = msg.data[1]
        self.err_y = msg.data[2]

    # -------------------- Basic Controls --------------------

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def disarm(self):
        req = CommandBool.Request()
        req.value = False
        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    # (takeoff service is kept but no longer called)

    # -------------------- Helpers --------------------

    def wait_for_connection(self):
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.5)

    def wait_for_local_position(self):
        while self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.initial_alt = self.local_pos.pose.position.z

    def hover(self, duration):
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.2)

    # -------------------- Precision Landing --------------------

    def precision_land(self):

        self.get_logger().info("Starting precision landing...")

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.05)

            now = time.time()
            dt = now - self.prev_time
            if dt <= 0:
                continue

            vx = vy = vz = 0.0

            if self.detected == 1.0:

                dx = (self.err_x - self.prev_err_x) / dt
                dy = (self.err_y - self.prev_err_y) / dt

                # Corrected PD control with proper camera → body rotation
                vx = -self.kp * self.err_y - self.kd * dy      # note the two minuses
                vy =  self.kp * self.err_x + self.kd * dx      # note the plus on derivative

                vx = float(np.clip(vx, -self.max_vel_xy, self.max_vel_xy))
                vy = float(np.clip(vy, -self.max_vel_xy, self.max_vel_xy))

                error_mag = np.sqrt(self.err_x**2 + self.err_y**2)
                if error_mag > self.threshold:
                    self.stable_hover_timer_started = False

                if error_mag < self.threshold:
                    if not self.stable_hover_timer_started:
                        self.stable_hover_timer_started = True
                        self.stable_hover_time = time.time()
                    
                    if time.time() - self.stable_hover_time > 2.0:
                        vx = vy = 0.0
                        vz = -self.descent_speed

            # Publish using the new helper (header + base_link frame)
            self.publish_velocity(vx, vy, vz)

            alt = self.local_pos.pose.position.z

            self.get_logger().info(
                f"Alt: {alt:.2f} | err: {self.err_x:.1f},{self.err_y:.1f} | vel: {vx:.2f},{vy:.2f},{vz:.2f}"
            )

            if alt <= self.initial_alt + 0.2:
                self.get_logger().info("Landed → Disarming")
                self.disarm()
                return

            self.prev_err_x = self.err_x
            self.prev_err_y = self.err_y
            self.prev_time = now


# -------------------- MAIN --------------------

def main(args=None):

    rclpy.init(args=args)
    node = PrecisionLandingNode()

    try:
        node.wait_for_connection()
        node.wait_for_local_position()

        target_alt = CLIMB_HEIGHT

        # GUIDED MODE
        while node.state.mode != GUIDED_MODE:
            node.set_mode(GUIDED_MODE)
            rclpy.spin_once(node, timeout_sec=0.5)

        # ARM
        while not node.state.armed:
            node.arm()
            rclpy.spin_once(node, timeout_sec=0.5)

        # === TAKEOFF REPLACED BY VELOCITY CLIMB (using your exact ROS1-style setpoint) ===
        node.get_logger().info(f"Climbing to {target_alt} m using velocity setpoints...")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.local_pos.pose.position.z >= target_alt - 0.2:
                break
            node.publish_velocity(vz=CLIMB_RATE)   # positive Z = climb

        # Stop climbing (send zero velocity)
        node.publish_velocity()
        node.get_logger().info("Reached target altitude → hovering")

        # HOVER
        node.hover(HOVER_TIME)

        # PRECISION LANDING (now also uses the fixed velocity publisher)
        node.precision_land()

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt → shutting down")
    except Exception as e:
        node.get_logger().error(f"Exception: {e}")
    finally:
        # Safety stop
        node.publish_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()