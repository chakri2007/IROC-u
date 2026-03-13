#!/usr/bin/env python3

"""
Takeoff test for Copter using MAVROS (ROS2)

Mission:
1. Record initial altitude
2. Climb +1 meter
3. Hover for 60 seconds
4. Start landing
5. Monitor descent speed
6. Disarm when altitude ≈ initial altitude
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import time

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


GUIDED_MODE = "GUIDED"
CLIMB_HEIGHT = 1.0
HOVER_TIME = 60.0


class CopterTakeoffMAVROS(Node):

    def __init__(self):

        super().__init__("copter_takeoff_test")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = None

        # Subscribers
        self.create_subscription(
            State,
            '/mavros/state',
            self.state_cb,
            qos
        )

        self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.local_pos_cb,
            qos
        )

        # Service clients
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        for srv in [self.arm_srv, self.mode_srv, self.takeoff_srv]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for MAVROS services...")

    # ----------------------------------------------------

    def state_cb(self, msg):
        self.state = msg

    # ----------------------------------------------------

    def local_pos_cb(self, msg):
        self.local_pos = msg

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

        if future.result() and future.result().mode_sent:
            self.get_logger().info(f"Mode set request: {mode}")
            return True

        self.get_logger().error("Failed to set mode")
        return False

    # ----------------------------------------------------

    def arm(self):

        req = CommandBool.Request()
        req.value = True

        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info("Vehicle armed")
            return True

        self.get_logger().error("Arming failed")
        return False

    # ----------------------------------------------------

    def disarm(self):

        req = CommandBool.Request()
        req.value = False

        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info("Vehicle disarmed")
            return True

        self.get_logger().error("Disarm failed")
        return False

    # ----------------------------------------------------

    def takeoff(self, altitude):

        req = CommandTOL.Request()

        req.altitude = altitude
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0

        future = self.takeoff_srv.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info(f"Takeoff command sent: {altitude:.2f}")
            return True

        self.get_logger().error("Takeoff command failed")
        return False

    # ----------------------------------------------------

    def wait_until_reach_altitude(self, target):

        self.get_logger().info("Climbing to target altitude...")

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.2)

            alt = self.local_pos.pose.position.z

            self.get_logger().info(f"Altitude: {alt:.2f} m")

            if alt >= target - 0.4:
                self.get_logger().info("Target altitude reached")
                return True

            time.sleep(0.5)

    # ----------------------------------------------------

    def hover_for_duration(self, duration_sec):

        self.get_logger().info(f"Hovering for {duration_sec} seconds...")

        start_time = time.time()

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.2)

            elapsed = time.time() - start_time
            alt = self.local_pos.pose.position.z

            self.get_logger().info(
                f"Hovering | Altitude: {alt:.2f} m | Time: {elapsed:.1f}/{duration_sec}s"
            )

            if elapsed >= duration_sec:
                self.get_logger().info("Hover time completed")
                return

            time.sleep(0.5)

    # ----------------------------------------------------

    def monitor_descent(self):

        self.get_logger().info("Monitoring descent...")

        prev_alt = self.local_pos.pose.position.z
        prev_time = time.time()

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.2)

            alt = self.local_pos.pose.position.z
            now = time.time()

            dt = now - prev_time
            dz = prev_alt - alt

            if dt > 0:

                velocity = dz / dt

                self.get_logger().info(
                    f"Altitude: {alt:.2f} m  Descent rate: {velocity:.2f} m/s")

                if velocity > 2.0:
                    self.get_logger().warn("Sudden drop detected!")
                else:
                    self.get_logger().info("Controlled descent")

            if alt <= self.initial_alt + 0.1:
                self.get_logger().info(
                    "Altitude close to initial height → Disarming")
                self.disarm()
                return

            prev_alt = alt
            prev_time = now

            time.sleep(0.4)


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

        node.wait_until_reach_altitude(target_alt)

        # Hover for 60 seconds
        node.hover_for_duration(HOVER_TIME)

        node.get_logger().info("Switching to LAND mode")

        node.set_mode("LAND")

        node.monitor_descent()

    except KeyboardInterrupt:

        node.get_logger().info("Interrupted")

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()