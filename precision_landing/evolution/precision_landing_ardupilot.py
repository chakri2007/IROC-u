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
CLIMB_HEIGHT = 2.0
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
        # Control params
        self.kp = 0.0025
        self.kd = 0.0
        self.max_vel_xy = 0.2
        self.threshold = 75.0
        self.descent_speed = 0.1
        self.stable_hover_timer_started = False
        self.stable_hover_time = 0.0
        # Subscribers
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_cb, qos)
        self.create_subscription(Float32MultiArray, '/aruco_error', self.aruco_cb, 10)
        # Publisher
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )
        # Services
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        for srv in [self.arm_srv, self.mode_srv, self.takeoff_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for MAVROS services...")
    # -------------------- Callbacks --------------------
    def state_cb(self, msg):
        self.state = msg
    def local_pos_cb(self, msg):
        self.local_pos = msg
    def aruco_cb(self, msg):
        self.detected = msg.data[0]
        self.err_x = msg.data[1]
        self.err_y = msg.data[2]
    # -------------------- Controls --------------------
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
    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = altitude # relative to HOME
        future = self.takeoff_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success
    # -------------------- Helpers --------------------
    def wait_for_connection(self):
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.5)
    def wait_for_local_position(self):
        while self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.initial_alt = self.local_pos.pose.position.z
        self.get_logger().info(f"Initial Altitude: {self.initial_alt:.2f}")
    def wait_until_altitude(self, target_alt):
        self.get_logger().info(f"Waiting for altitude: {target_alt:.2f}")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            current_alt = self.local_pos.pose.position.z
            print(f"Altitude: {current_alt:.2f}", flush=True)
            if current_alt >= target_alt - 0.25:
                self.get_logger().info("Reached target altitude")
                return
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
                vy = self.kp * self.err_y + self.kd * dy
                vx = -self.kp * self.err_x - self.kd * dx
                vx = float(np.clip(vx, -self.max_vel_xy, self.max_vel_xy))
                vy = float(np.clip(vy, -self.max_vel_xy, self.max_vel_xy))
                error_mag = np.sqrt(self.err_x**2 + self.err_y**2)
                if error_mag > self.threshold:
                    self.stable_hover_timer_started = False
                if error_mag < self.threshold:
                    vx = vy = 0.0
                    if not self.stable_hover_timer_started:
                        self.stable_hover_timer_started = True
                        self.stable_hover_time = time.time()
                    if (time.time() - self.stable_hover_time) > 0.5:
              
                        vz = -self.descent_speed
            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link" # ✅ CRITICAL
            vel.twist.linear.x = vx
            vel.twist.linear.y = vy
            vel.twist.linear.z = vz
            self.vel_pub.publish(vel)
            alt = self.local_pos.pose.position.z
            print(f"Alt: {alt:.2f} | err: {self.err_x:.1f},{self.err_y:.1f}", flush=True)
            # ---- landing condition ----
            if abs(alt - self.initial_alt) < 0.4:
                self.get_logger().info("Near ground → switching to LAND")
                self.set_mode("LAND")
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
        # GUIDED MODE
        while node.state.mode != GUIDED_MODE:
            node.set_mode(GUIDED_MODE)
            rclpy.spin_once(node, timeout_sec=0.5)
        # ARM
        while not node.state.armed:
            node.arm()
            rclpy.spin_once(node, timeout_sec=0.5)
        time.sleep(2) # stabilize after arming
        # ---- TAKEOFF (service-based) ----
        target_alt = node.initial_alt + CLIMB_HEIGHT
        node.get_logger().info(f"Takeoff target: {target_alt:.2f}")
        node.takeoff(CLIMB_HEIGHT) # relative to HOME
        node.wait_until_altitude(target_alt)
        # HOVER
        node.hover(HOVER_TIME)
        # PRECISION LAND
        node.precision_land()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()