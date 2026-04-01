#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from apriltag_msgs.msg import AprilTagDetectionArray
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class AutonomousPrecisionLanding(Node):

    def __init__(self):
        super().__init__('autonomous_precision_landing')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = 0.0
        
        self.tag_frame = ""
        self.err_x = 0.0
        self.err_y = 0.0
        self.err_z = 0.0

        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        self.declare_parameter("takeoff_alt", 3.0)
        self.declare_parameter("kp", 0.5)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("max_vel_xy", 0.3)
        self.declare_parameter("land_alt_threshold", 0.1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame = "camera_color_optical_frame"

        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.aruco_cb, qos_profile_sensor_data)
        
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.local_pos = msg

    def aruco_cb(self, msg):
        if len(msg.detections) > 0:
            det = msg.detections[0]
            tag_id = det.id[0] if hasattr(det.id, '__iter__') else det.id
            self.tag_frame = f"tag25h9:{tag_id}"
        else:
            self.tag_frame = ""

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent

    def arm(self, status):
        req = CommandBool.Request()
        req.value = status
        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        future = self.takeoff_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def run_mission(self):
        self.get_logger().info("Waiting for FCU connection...")
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        while self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.initial_alt = self.local_pos.pose.position.z
        
        self.get_logger().info("Setting GUIDED mode and Arming...")
        self.set_mode("GUIDED")
        self.arm(True)
        time.sleep(2)

        target_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        self.get_logger().info(f"Taking off to {target_alt_rel}m")
        self.takeoff(target_alt_rel)
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.local_pos.pose.position.z >= (self.initial_alt + target_alt_rel - 0.3):
                break
        
        # ================== FIXED HOVER ==================
        self.get_logger().info("Hovering to search for tag...")

        hover_duration = 30.0
        start_time = time.time()

        while time.time() - start_time < hover_duration:
            rclpy.spin_once(self, timeout_sec=0.05)

            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link"

            vel.twist.linear.x = 0.0
            vel.twist.linear.y = 0.0
            vel.twist.linear.z = 0.0

            self.vel_pub.publish(vel)
        # =================================================

        self.precision_landing_loop()

    def precision_landing_loop(self):
        self.get_logger().info("Starting PID Precision Landing Phase")
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
            found_now = False
            if self.tag_frame:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.camera_frame, self.tag_frame, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.02)
                    )
                    self.err_x = trans.transform.translation.x
                    self.err_y = trans.transform.translation.y
                    self.err_z = trans.transform.translation.z
                    found_now = True
                except (LookupException, ConnectivityException, ExtrapolationException):
                    found_now = False

            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link"

            if found_now:
                now = time.time()
                dt = now - self.prev_time if (now - self.prev_time) > 0 else 0.05
                
                kp = self.get_parameter("kp").get_parameter_value().double_value
                kd = self.get_parameter("kd").get_parameter_value().double_value
                
                vx = -(kp * self.err_x) + (kd * (self.err_x - self.prev_err_x) / dt)
                vy = (kp * self.err_y) + (kd * (self.err_y - self.prev_err_y) / dt)
                
                error_mag = np.sqrt(self.err_x**2 + self.err_y**2)
                vz = -0.20 if error_mag < 0.05 else 0.0
                
                max_v = self.get_parameter("max_vel_xy").get_parameter_value().double_value
                vel.twist.linear.x = float(np.clip(vx, -max_v, max_v))
                vel.twist.linear.y = float(np.clip(vy, -max_v, max_v))
                vel.twist.linear.z = float(vz)

                self.prev_err_x, self.prev_err_y = self.err_x, self.err_y
                self.prev_time = now
            else:
                vel.twist.linear.x = 0.0
                vel.twist.linear.y = 0.0
                vel.twist.linear.z = 0.0

            self.vel_pub.publish(vel)

            alt_rel = self.local_pos.pose.position.z - self.initial_alt
            if alt_rel < self.get_parameter("land_alt_threshold").get_parameter_value().double_value:
                self.get_logger().info("Final Landing...")
                self.set_mode("LAND")
                break

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