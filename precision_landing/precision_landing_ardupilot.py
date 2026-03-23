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
CLIMB_HEIGHT = 5.0


class CopterTakeoffMAVROS(Node):

    def __init__(self):
        super().__init__("vision_landing_controller")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ================== STATE ==================
        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = None

        # Aruco error
        self.detected = 0.0
        self.err_x = 0.0
        self.err_y = 0.0

        # PD controller memory
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        # ================== PARAMETERS ==================
        self.declare_parameter("kp", 0.002)
        self.declare_parameter("kd", 0.001)

        self.declare_parameter("max_vel_xy", 0.5)
        self.declare_parameter("max_vel_z", 0.3)

        self.declare_parameter("error_threshold", 20.0)
        self.declare_parameter("descent_speed", 0.2)

        # ================== SUBSCRIBERS ==================
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pos_cb, qos)

        # Aruco error subscriber
        self.create_subscription(Float32MultiArray, '/aruco_error', self.aruco_cb, 10)

        # Gain tuning topic
        self.create_subscription(Float32MultiArray, '/controller_gains', self.gains_cb, 10)

        # ================== PUBLISHERS ==================
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # ================== SERVICES ==================
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        for srv in [self.arm_srv, self.mode_srv, self.takeoff_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for MAVROS services...")

        # ================== CONTROL LOOP ==================
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    # ----------------------------------------------------
    def state_cb(self, msg):
        self.state = msg

    def local_pos_cb(self, msg):
        self.local_pos = msg

    def aruco_cb(self, msg):
        self.detected = msg.data[0]
        self.err_x = msg.data[1]
        self.err_y = msg.data[2]

    def gains_cb(self, msg):
        # [kp, kd]
        self.set_parameters([
            rclpy.parameter.Parameter("kp", rclpy.Parameter.Type.DOUBLE, msg.data[0]),
            rclpy.parameter.Parameter("kd", rclpy.Parameter.Type.DOUBLE, msg.data[1])
        ])
        self.get_logger().info(f"Updated gains: kp={msg.data[0]}, kd={msg.data[1]}")

    # ----------------------------------------------------
    def control_loop(self):

        if self.initial_alt is None:
            return

        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value
        max_xy = self.get_parameter("max_vel_xy").value
        max_z = self.get_parameter("max_vel_z").value
        threshold = self.get_parameter("error_threshold").value
        descent_speed = self.get_parameter("descent_speed").value

        now = time.time()
        dt = now - self.prev_time
        if dt == 0:
            return

        vx = 0.0
        vy = 0.0
        vz = 0.0

        if self.detected == 1.0:

            dx = (self.err_x - self.prev_err_x) / dt
            dy = (self.err_y - self.prev_err_y) / dt

            vx =  kp * self.err_y + kd * dy
            vy = -kp * self.err_x - kd * dx

            # clamp AFTER mapping
            vx = float(np.clip(vx, -max_xy, max_xy))
            vy = float(np.clip(vy, -max_xy, max_xy))

            vel.twist.linear.x = vx
            vel.twist.linear.y = vy
            vel.twist.linear.z = vz

            error_mag = np.sqrt(self.err_x**2 + self.err_y**2)

            if error_mag < threshold:
                vz = -descent_speed   # descend
            else:
                vz = 0.0              # hold altitude

        else:
            # no marker → hold
            vx = vy = vz = 0.0

        # Publish velocity
        vel = TwistStamped()
        vel.twist.linear.x = -vy
        vel.twist.linear.y = -vx
        vel.twist.linear.z = vz
        self.get_logger().debug(f"Publishing velocity: vx={-vy:.2f}, vy={-vx:.2f}, vz={vz:.2f}")

        self.vel_pub.publish(vel)

        # Update previous
        self.prev_err_x = self.err_x
        self.prev_err_y = self.err_y
        self.prev_time = now

    # ----------------------------------------------------
    def wait_for_local_position(self):

        self.get_logger().info("Waiting for local position...")

        while self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.2)

        self.initial_alt = self.local_pos.pose.position.z

        self.get_logger().info(
            f"Initial altitude recorded: {self.initial_alt:.3f} m")

    # ----------------------------------------------------
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

    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = altitude
        future = self.takeoff_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success


# ----------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = CopterTakeoffMAVROS()

    try:
        node.wait_for_local_position()

        target_alt = node.initial_alt + CLIMB_HEIGHT

        node.set_mode(GUIDED_MODE)
        time.sleep(1)

        node.arm()
        time.sleep(2)

        node.takeoff(target_alt)

        node.get_logger().info("Switching to vision control...")

        # Now control loop handles everything

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()