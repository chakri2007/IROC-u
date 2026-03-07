#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from offboard_controller.msg.srv import Takeoff
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_srv.srv import Trigger
import time
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandTOL
import threading

class OffboardController(Node):
    def __init__(self):
        super().__intit__('offboard_node')
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSProfile.HistoryPolicy.KEEP_LAST)

       # Subscribers
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.pos_cmd_sub = self.create_subscription(Point, '/offboard/position_cmd', self.pos_cmd_cb, qos)

        # Publishers
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos)

        # Service clients (MAVROS)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # custom services
        self.arm_srv = self.create_service(Trigger, '/offboard/arm', self.arm_cb)
        self.takeoff_srv = self.create_service(Takeoff, '/offboard/takeoff', self.takeoff_cb)
        self.land_srv = self.create_service(Trigger, '/offboard/land', self.land_cb)

        # State tracking
        self.current_state = State()
        self.target_pos = Point(x=0.0, y=0.0, z=0.0)  # Default hover
        self.timer = self.create_timer(0.1, self.publish_setpoint)  # 10 Hz
        
        #Wait for drone to be in GUIDED mode
        while rclpy.ok() and self.current_state.mode != "GUIDED":
            rclpy.spin_once(self)
            self.get_logger().info("Waiting for drone to enter GUIDED mode...")
            time.sleep(1)

    def state_cb(self, msg):
        self.current_state = msg
    def pos_cmd_cb(self, msg):
        self.target_pos = msg
        self.get_logger().info(f"Received new position command: {self.target_pos}")
    def publish_setpoint(self):
        if not self.current_state.connected:
            self.get_logger().warn("Waiting for connection to FCU...")
            return
        if self.current_state.mode != "OFFBOARD":
            self.get_logger().warn("Drone not in OFFBOARD mode, cannot publish setpoints.")
            return
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = self.target_pos
        self.setpoint_pub.publish(pose)
    def arm_cb(self, request, response):
        if self.current_state.armed:
            response.success = False
            response.message = "Drone is already armed."
            return response
        arm_req = CommandBool.Request()
        arm_req.value = True
        future = self.arm_client.call_async(arm_req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().success:
            response.success = True
            response.message = "Drone armed successfully."
        else:
            response.success = False
            response.message = "Failed to arm drone."
        return response
    
    def takeoff_cb(self, request, response):
        if not self.current_state.armed:
            response.success = False
            response.message = "Drone must be armed before takeoff."
            return response
        self.target_pos.x = 0.0
        self.target_pos.y = 0.0
        self.target_pos.z = request.altitude

        response.success = True
        response.message = f"Takeoff initiated to altitude {request.altitude} meters."
        return response
    def land_cb(self, request, response):
        if not self.current_state.armed:
            response.success = False
            response.message = "Drone must be armed to land."
            return response
        land_req = CommandTOL.Request()
        land_req.altitude = 0.0
        land_req.latitude = 0.0
        land_req.longitude = 0.0
        future = self.land_client.call_async(land_req)
        future.wait_for_completed()

        if not future.result().success:
            response.success = False
            response.message = "Failed to initiate landing."
            return response
        self.get_logger().info("Landing... waiting for touchdown")
        while self.current_state.armed:
            self.get_logger().info("Still armed... waiting")
            time.sleep(1)
        response.success = True
        response.message = "Landing successful, drone is disarmed."
        return response
def main(args=None):
    rclpy.init(args=args)
    node = OffboardController()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()