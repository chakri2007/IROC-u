#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time

from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Float32MultiArray


class OffboardControl(Node):

    def __init__(self):
        super().__init__('vision_offboard_control')

        # ================== STATE ==================
        self.current_state = State()
        self.current_alt = 0.0

        # Aruco data
        self.detected = 0.0
        self.err_x = 0.0
        self.err_y = 0.0

        # PD memory
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        # ================== PARAMETERS ==================
        self.declare_parameter("kp", 0.002)
        self.declare_parameter("kd", 0.0)

        self.declare_parameter("max_vel_xy", 0.3)
        self.declare_parameter("max_vel_z", 0.3)

        self.declare_parameter("error_threshold", 20.0)
        self.declare_parameter("descent_speed", 0.1)

        self.declare_parameter("takeoff_alt", 2.0)
        self.declare_parameter("hover_time", 3.0)

        # ================== SUBSCRIBERS ==================
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)

        self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pos_cb,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            '/aruco_error',
            self.aruco_cb,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            '/controller_gains',
            self.gains_cb,
            10
        )

        # ================== PUBLISHER ==================
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # ================== SERVICES ==================
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.wait_for_services()

        # ================== CONTROL ==================
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.setpoint_counter = 0
        self.last_req_time = self.get_clock().now()

        # State machine
        self.phase = "INIT"
        self.start_time = None

    # ----------------------------------------------------
    def state_cb(self, msg):
        self.current_state = msg

    def pos_cb(self, msg):
        self.current_alt = msg.pose.position.z

    def aruco_cb(self, msg):
        self.detected = msg.data[0]
        self.err_x = msg.data[1]
        self.err_y = msg.data[2]

    def gains_cb(self, msg):
        self.set_parameters([
            rclpy.parameter.Parameter("kp", rclpy.Parameter.Type.DOUBLE, msg.data[0]),
            rclpy.parameter.Parameter("kd", rclpy.Parameter.Type.DOUBLE, msg.data[1])
        ])
        self.get_logger().info(f"Gains updated: kp={msg.data[0]}, kd={msg.data[1]}")

    # ----------------------------------------------------
    def wait_for_services(self):
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_client.call_async(req)

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        self.arming_client.call_async(req)

    # ----------------------------------------------------
    def control_loop(self):

        if not self.current_state.connected:
            return

        # Send initial dummy setpoints
        if self.setpoint_counter < 100:
            self.publish_velocity(0.0, 0.0, 0.0)
            self.setpoint_counter += 1
            return

        current_time = self.get_clock().now()
        dt_req = (current_time - self.last_req_time).nanoseconds / 1e9

        # Switch to OFFBOARD
        if self.current_state.mode != "OFFBOARD" and dt_req > 5.0:
            self.set_mode("OFFBOARD")
            self.get_logger().info("OFFBOARD requested")
            self.last_req_time = current_time

        # Arm
        elif not self.current_state.armed and dt_req > 5.0:
            self.arm()
            self.get_logger().info("Arming requested")
            self.last_req_time = current_time

        # ================== CONTROL ==================
        vx, vy, vz = 0.0, 0.0, 0.0

        now = time.time()
        dt = now - self.prev_time
        if dt == 0:
            return

        # ================== STATE MACHINE ==================

        # INIT → TAKEOFF
        if self.phase == "INIT":
            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                self.get_logger().info("Switching to TAKEOFF")
                self.phase = "TAKEOFF"

        # TAKEOFF
        elif self.phase == "TAKEOFF":

            vz = 0.5

            takeoff_alt = self.get_parameter("takeoff_alt").value

            if self.current_alt >= takeoff_alt - 0.2:
                self.get_logger().info("Takeoff reached → Hover")
                self.phase = "HOVER"
                self.start_time = time.time()

        # HOVER
        elif self.phase == "HOVER":

            vz = 0.0

            hover_time = self.get_parameter("hover_time").value

            if time.time() - self.start_time > hover_time:
                self.get_logger().info("Switching to VISION mode")
                self.phase = "VISION"

        # VISION LANDING
        elif self.phase == "VISION":

            kp = self.get_parameter("kp").value
            kd = self.get_parameter("kd").value
            max_xy = self.get_parameter("max_vel_xy").value
            threshold = self.get_parameter("error_threshold").value
            descent = self.get_parameter("descent_speed").value

            if self.detected == 1.0:

                dx = (self.err_x - self.prev_err_x) / dt
                dy = (self.err_y - self.prev_err_y) / dt

                # Correct mapping (downward camera)
                vx =  kp * self.err_y + kd * dy
                vy = -kp * self.err_x - kd * dx

                vx = float(np.clip(vx, -max_xy, max_xy))
                vy = float(np.clip(vy, -max_xy, max_xy))

                error_mag = np.sqrt(self.err_x**2 + self.err_y**2)

                if error_mag < threshold:
                    vz = -descent
                else:
                    vz = 0.0

            else:
                vx = vy = vz = 0.0

            # Auto land when close to ground
            if self.current_alt < 0.2:
                self.get_logger().info("Landing complete → AUTO.LAND")
                self.set_mode("AUTO.LAND")

        # Publish velocity
        self.publish_velocity(vx, vy, vz)
        # Debug log in which direction the velocities are gives in drones body frame (x forward, y right, z down)
        self.get_logger().info(f" vx={vx:.2f} m/s, vy={vy:.2f} m/s, vz={vz:.2f} m/s")

        # Update memory
        self.prev_err_x = self.err_x
        self.prev_err_y = self.err_y
        self.prev_time = now

    # ----------------------------------------------------
    def publish_velocity(self, vx, vy, vz):
        vel = TwistStamped()
        vel.twist.linear.x = vx
        vel.twist.linear.y = vy
        vel.twist.linear.z = vz
        self.vel_pub.publish(vel)


# ----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    node = OffboardControl()

    try:
        node.get_logger().info("Waiting for FCU connection...")
        while rclpy.ok() and not node.current_state.connected:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("FCU connected!")
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()