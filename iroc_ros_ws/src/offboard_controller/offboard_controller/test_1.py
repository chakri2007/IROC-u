#!/usr/bin/env python3

"""
Takeoff test for Copter using MAVROS (ROS 2)
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import time

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import Header


# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────

TAKEOFF_ALT = 10.5          # meters
GUIDED_MODE = "GUIDED_NOGPS"
LOCAL_FRAME_TIMEOUT = 30.0
MODE_CHANGE_TIMEOUT = 15.0
ARM_TIMEOUT = 20.0
TAKEOFF_TIMEOUT = 40.0


class CopterTakeoffMAVROS(Node):
    def __init__(self):
        super().__init__("copter_takeoff_mavros")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # State subscriber
        self.state = State()
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_cb,
            qos
        )

        # Local position subscriber (we'll use it to check altitude)
        self.local_pos = PoseStamped()
        self.local_pos_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.local_pos_cb,
            qos
        )

        # Services
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # Wait for services
        for srv, name in [
            (self.arm_srv, "arming"),
            (self.mode_srv, "set mode"),
            (self.takeoff_srv, "takeoff")
        ]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{name} service not available, waiting...')

        self.get_logger().info("MAVROS takeoff node initialized")

    def state_cb(self, msg: State):
        self.state = msg

    def local_pos_cb(self, msg: PoseStamped):
        self.local_pos = msg

    def wait_for_local_position(self, timeout_sec: float = 15.0) -> bool:
        start = self.get_clock().now()
        while self.local_pos.header.stamp.sec == 0:
            if (self.get_clock().now() - start) > Duration(seconds=timeout_sec):
                self.get_logger().error("No local position received")
                return False
            rclpy.spin_once(self, timeout_sec=0.2)
        return True

    def set_mode(self, mode: str, timeout_sec: float = MODE_CHANGE_TIMEOUT) -> bool:
        req = SetMode.Request()
        req.custom_mode = mode

        start = self.get_clock().now()
        while (self.get_clock().now() - start) < Duration(seconds=timeout_sec):
            future = self.mode_srv.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

            if future.result() is not None and future.result().mode_sent:
                self.get_logger().info(f"Mode change to {mode} requested — waiting confirmation...")
                # Wait a bit for state to update
                for _ in range(8):
                    if self.state.mode == mode:
                        self.get_logger().info(f"→ now in {mode} mode")
                        return True
                    time.sleep(0.4)
                    rclpy.spin_once(self)

            time.sleep(0.8)

        self.get_logger().error(f"Failed to set mode to {mode}")
        return False

    def arm(self, timeout_sec: float = ARM_TIMEOUT) -> bool:
        req = CommandBool.Request()
        req.value = True

        start = self.get_clock().now()
        while (self.get_clock().now() - start) < Duration(seconds=timeout_sec):
            future = self.arm_srv.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

            if future.result() is not None and future.result().success:
                self.get_logger().info("Arm command sent — waiting for armed state...")
                for _ in range(12):
                    if self.state.armed:
                        self.get_logger().info("Vehicle ARMED")
                        return True
                    time.sleep(0.4)
                    rclpy.spin_once(self)
            else:
                self.get_logger().warn("Arm call failed — retrying...")

            time.sleep(1.0)

        self.get_logger().error("Failed to arm vehicle")
        return False

    def takeoff(self, alt: float, timeout_sec: float = TAKEOFF_TIMEOUT) -> bool:
        req = CommandTOL.Request()
        req.altitude = float(alt)
        req.latitude = 0.0   # relative takeoff (0 = current)
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0        # keep current yaw

        future = self.takeoff_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info(f"Takeoff command sent — target {alt:.1f} m")
            return True
        else:
            self.get_logger().error("Takeoff command failed")
            return False

    def wait_until_reach_altitude(self, target_alt: float, timeout_sec: float = 60.0):
        start = self.get_clock().now()
        last_log = start

        while (self.get_clock().now() - start) < Duration(seconds=timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.3)

            if self.local_pos.header.stamp.sec == 0:
                continue

            current_alt = self.local_pos.pose.position.z

            now = self.get_clock().now()
            if (now - last_log) > Duration(seconds=2.5):
                self.get_logger().info(f"Altitude: {current_alt:.2f} m / {target_alt:.1f} m")
                last_log = now

            if current_alt >= target_alt - 0.6:  # small margin
                self.get_logger().info(f"Reached target altitude ≈ {current_alt:.2f} m")
                return True

            time.sleep(0.4)

        self.get_logger().error("Did NOT reach target altitude in time")
        return False


def main(args=None):
    rclpy.init(args=args)
    node = CopterTakeoffMAVROS()

    try:
        # 1. Wait for local position
        if not node.wait_for_local_position():
            raise RuntimeError("No local position available")

        # 2. Switch to GUIDED
        if not node.set_mode(GUIDED_MODE):
            raise RuntimeError(f"Could not switch to {GUIDED_MODE} mode")

        # 3. Arm
        if not node.arm():
            raise RuntimeError("Could not arm vehicle")

        # 4. Takeoff
        if not node.takeoff(TAKEOFF_ALT):
            raise RuntimeError("Takeoff command failed")

        # 5. Monitor climb
        if not node.wait_until_reach_altitude(TAKEOFF_ALT):
            raise RuntimeError("Failed to reach takeoff altitude")

        node.get_logger().info("Takeoff test completed successfully!")

    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()