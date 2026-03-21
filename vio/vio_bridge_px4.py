import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from mavros_msgs.msg import CompanionProcessStatus
from geometry_msgs.msg import TransformStamped

import numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
import tf2_ros

class VIOBridge(Node):
    def __init__(self):
        super().__init__('vio_bridge_px4_node')
        self.sub = self.create_subscription(
            Odometry,
            '/vio/odom',
            self.odom_callback,
            10
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            '/mavros/odometry/out',
            10
        )
        self.status_pub = self.create_publisher(
            CompanionProcessStatus,
            '/mavros/companion_process/status',
            10
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)


        self.cam_yaw_offset = 0.0
        self.cam_pitch_offset = 0.0
        self.cam_roll_offset = 0.0

        self.cam_quaternion = quaternion_from_euler(
            self.cam_roll_offset, self.cam_pitch_offset, self.cam_yaw_offset
        )

        self.frequency = 30.0

        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
    
    def odom_callback(self, msg):
        self.odom_msg = msg
    
    def timer_callback(self):
        if hasattr(self, 'odom_msg'):

            odom_in = self.odom_msg
            odom_out = Odometry()

            odom_out.header.stamp = self.get_clock().now().to_msg()
            odom_out.header.frame_id = "odom"
            odom_out.child_frame_id = "base_link"

            odom_out.pose.pose.position = odom_in.pose.pose.position

            quat_in = [
                odom_in.pose.pose.orientation.x,
                odom_in.pose.pose.orientation.y,
                odom_in.pose.pose.orientation.z,
                odom_in.pose.pose.orientation.w
            ]
            quat_out = quaternion_multiply(quat_in, self.cam_quaternion)

            odom_out.pose.pose.orientation.x = quat_out[0]
            odom_out.pose.pose.orientation.y = quat_out[1]
            odom_out.pose.pose.orientation.z = quat_out[2]
            odom_out.pose.pose.orientation.w = quat_out[3]

            v_world = np.array([odom_in.twist.twist.linear.x, odom_in.twist.twist.linear.y, odom_in.twist.twist.linear.z])
            R_world_to_body = quaternion_matrix(quat_out)[:3, :3].T
            v_body = R_world_to_body @ v_world

            odom_out.twist.twist.linear.x = v_body[0]
            odom_out.twist.twist.linear.y = v_body[1]
            odom_out.twist.twist.linear.z = v_body[2]
            self.odom_pub.publish(odom_out)

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"

            t.transform.translation = odom_out.pose.pose.position
            t.transform.rotation = odom_out.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)
            
            # Publish the companion process status message
            status_msg = CompanionProcessStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.state = 4  # MAV_STATE_ACTIVE
            status_msg.component = 197  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    vio_bridge = VIOBridge()
    rclpy.spin(vio_bridge)
    vio_bridge.destroy_node()
    rclpy.shutdown()