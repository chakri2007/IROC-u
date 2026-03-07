#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger
from offboard_interfaces.srv import Takeoff
import sys

class CommandingNode(Node):
    def __init__(self):
        super().__init__('commanding_node')
        self.pos_pub = self.create_publisher(Point, '/offboard/position_cmd', 10)

        # Wait for services
        self.get_logger().info('Waiting for offboard services...')
        self.arm_client = self.create_client(Trigger, '/offboard/arm')
        self.takeoff_client = self.create_client(Takeoff, '/offboard/takeoff')
        self.land_client = self.create_client(Trigger, '/offboard/land')
        while not all([self.arm_client.wait_for_service(1.0),
                       self.takeoff_client.wait_for_service(1.0),
                       self.land_client.wait_for_service(1.0)]):
            self.get_logger().warn('Services not ready...')

        self.run_interactive()

    def run_interactive(self):
        input('Press Enter to arm...')
        arm_future = self.arm_client.call_async(Trigger.Request())
        arm_future.wait_for_completed()
        if not arm_future.result().success:
            self.get_logger().error('Arm failed! Exiting.')
            return

        alt = float(input('Enter takeoff altitude (m): '))
        takeoff_req = Takeoff.Request(altitude=alt)
        takeoff_future = self.takeoff_client.call_async(takeoff_req)
        takeoff_future.wait_for_completed()
        if not takeoff_future.result().success:
            self.get_logger().error('Takeoff failed!')
            return


        input('Press Enter to land...')
        land_future = self.land_client.call_async(Trigger.Request())
        land_future.wait_for_completed()
        self.get_logger().info(land_future.result().message)

def main(args=None):
    rclpy.init(args=args)
    node = CommandingNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()