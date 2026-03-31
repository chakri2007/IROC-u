#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
import tf2_ros

class PositionBasedLanding(Node):

    def __init__(self):
        super().__init__('vision_position_based_landing')

        # ================== STATE ==================
        self.current_state = State()
        self.current_alt = 0.0

        self.err_x = 0.0
        self.err_y = 0.0
        self.err_z = 0.0
        self.detected = 0.0

        # PD memory
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_err_z = 0.0
        self.prev_time = time.time()

        # Integral terms
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0

        # Hover timer for low-altitude centering
        self.hover_time_start = None

        # TF for april_ros (pose comes from TF, not message)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.current_tag_id = -1
        self.tag_frame = ""

        # ================== PARAMETERS ==================
        self.declare_parameter("kp", 0.002)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("ki", 0.0)

        self.declare_parameter("max_vel_xy", 0.3)
        self.declare_parameter("max_vel_z", 0.2)

        self.declare_parameter("error_threshold", 0.2)
        self.declare_parameter("descent_speed", 0.1)

        self.declare_parameter("takeoff_alt", 1.0)
        self.declare_parameter("hover_time", 3.0)
        self.declare_parameter("z_error_threshold", 0.2)
        self.declare_parameter("landing_timeout", 30.0)
        self.declare_parameter("low_alt_hover_time", 3.0)

        # Load initial gains
        self.kp = self.get_parameter("kp").get_parameter_value().double_value
        self.kd = self.get_parameter("kd").get_parameter_value().double_value
        self.ki = self.get_parameter("ki").get_parameter_value().double_value

        # ================== SUBSCRIBERS ==================
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        self.create_subscription(
             PoseStamped,
             '/mavros/local_position/pose',
             self.pos_cb,
             qos_profile_sensor_data
         )

        #Get detected data from /detection topic (april_ros)
        self.create_subscription(
            AprilTagDetectionArray,
            '/detection',
            self.aruco_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Float32MultiArray,
            '/controller_gains',
            self.gains_cb,
            10
        )

        # ================== PUBLISHERS ==================
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # ================== SERVICES ==================
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # ================== CONTROL TIMER ==================
        self.create_timer(0.05, self.control_loop)  # 20 Hz

    def state_cb(self, msg):
        self.current_state = msg
    def pos_cb(self, msg):
        self.current_alt = msg.pose.position.z

    def aruco_cb(self, msg):
        if len(msg.detections) > 0:
            detection = msg.detections[0]
            # april_ros only gives detection info + TF (no pose in msg)
            self.current_tag_id = getattr(detection, 'id', 0)
            self.tag_frame = f"tag_{self.current_tag_id}"
            self.detected = 1.0
        else:
            self.err_x = 0.0
            self.err_y = 0.0
            self.err_z = 0.0
            self.detected = 0.0
            self.current_tag_id = -1
            self.tag_frame = ""

    def gains_cb(self, msg):
        self.get_logger().info(f"Received new gains: {msg.data}")
        if len(msg.data) >= 3:
            self.kp = msg.data[0]
            self.kd = msg.data[1]
            self.ki = msg.data[2]
    
    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')
        try:
            future = self.mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info(f"Mode changed to {mode}")
            else:
                self.get_logger().error(f"Failed to change mode: {future.exception()}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
    def arm(self, arm):
        req = CommandBool.Request()
        req.value = arm
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        try:
            future = self.arm_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None and future.result().success:
                self.get_logger().info(f"Arming {'successful' if arm else 'disarming successful'}")
            else:
                self.get_logger().error(f"Failed to {'arm' if arm else 'disarm'}: {future.exception()}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
    
    def control_loop(self):
        if not self.current_state.connected:
            return

        # === GET POSE FROM TF (april_ros only publishes TF, not pose in msg) ===
        if self.detected and self.tag_frame:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    self.tag_frame,
                    rclpy.time.Time()  # latest available transform
                )
                self.err_x = trans.transform.translation.x
                self.err_y = trans.transform.translation.y
                self.err_z = trans.transform.translation.z
            except Exception:
                # TF not ready yet or frame not found
                self.detected = 0.0

        if self.detected:
            if abs(self.current_alt - self.err_z) < self.get_parameter("z_error_threshold").get_parameter_value().double_value:
                if abs(self.err_x) < self.get_parameter("error_threshold").get_parameter_value().double_value and abs(self.err_y) < self.get_parameter("error_threshold").get_parameter_value().double_value:
                    if self.hover_time_start is None:
                        self.hover_time_start = time.time()
                    elif time.time() - self.hover_time_start > self.get_parameter("low_alt_hover_time").get_parameter_value().double_value:
                        self.get_logger().info("Landing...")
                        self.set_mode("LAND")
                        return

            # Compute PID control
            current_time = time.time()
            dt = current_time - self.prev_time if self.prev_time else 0.01
            self.prev_time = current_time

            # Proportional control
            vx = self.kp * self.err_x
            vy = self.kp * self.err_y
            vz = self.kp * self.err_z

            # Derivative control
            vx += self.kd * (self.err_x - self.prev_err_x) / dt
            vy += self.kd * (self.err_y - self.prev_err_y) / dt
            vz += self.kd * (self.err_z - self.prev_err_z) / dt

            # Integral control (running sum)
            self.integral_x += self.err_x * dt
            self.integral_y += self.err_y * dt
            self.integral_z += self.err_z * dt

            vx += self.ki * self.integral_x
            vy += self.ki * self.integral_y
            vz += self.ki * self.integral_z

            # Save current error for next derivative calculation
            self.prev_err_x = self.err_x
            self.prev_err_y = self.err_y
            self.prev_err_z = self.err_z

            # Limit velocities
            max_vel_xy = self.get_parameter("max_vel_xy").get_parameter_value().double_value
            max_vel_z = self.get_parameter("max_vel_z").get_parameter_value().double_value
            vx = np.clip(vx, -max_vel_xy, max_vel_xy)
            vy = np.clip(vy, -max_vel_xy, max_vel_xy)
            vz = np.clip(vz, -max_vel_z, max_vel_z)

            # Publish velocity command
            vel_msg = TwistStamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.header.frame_id = "base_link"
            vel_msg.twist.linear.x = vx
            vel_msg.twist.linear.y = vy
            vel_msg.twist.linear.z = vz
            self.vel_pub.publish(vel_msg)

        else:
            # If no tag detected, hover in place
            vel_msg = TwistStamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.header.frame_id = "base_link"
            vel_msg.twist.linear.x = 0.0
            vel_msg.twist.linear.y = 0.0
            vel_msg.twist.linear.z = 0.0
            self.vel_pub.publish(vel_msg)

    def takeoff(self, altitude):
        self.set_mode("GUIDED")
        self.arm(True)
        # Send takeoff command (this is a placeholder, replace with actual takeoff logic)
        self.get_logger().info(f"Taking off to {altitude} meters...")
        # Wait until we reach the desired altitude
        self.wait_until_altitude(altitude)
    
    def land(self):
        self.get_logger().info("Landing...")
        self.set_mode("LAND")
        # Wait until we are on the ground
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.current_alt < 0.1:  # Assuming we are on the ground if altitude is less than 10 cm
                self.get_logger().info("Landed successfully")
                self.arm(False)  # Disarm after landing
                break

    def wait_until_altitude(self, target_alt, timeout=20.0):
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            if abs(self.current_alt - target_alt) < 0.1:
                self.get_logger().info(f"Reached altitude {target_alt:.1f}m")
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("Altitude timeout!")

    def hover(self, duration):
        self.get_logger().info(f"Hovering for {duration} seconds...")
        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link"
            vel.twist.linear.x = 0.0
            vel.twist.linear.y = 0.0
            vel.twist.linear.z = 0.0
            self.vel_pub.publish(vel)
            rclpy.spin_once(self, timeout_sec=0.1)

    def precision_land(self):
        self.get_logger().info("Starting precision landing controller")
        self.set_mode("OFFBOARD")

def main(args=None):
    rclpy.init(args=args)
    node = PositionBasedLanding()
    node.takeoff(node.get_parameter("takeoff_alt").get_parameter_value().double_value)
    node.hover(node.get_parameter("hover_time").get_parameter_value().double_value)
    node.precision_land()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()