import rclpy
from rclpy.node import Node
import tf_transformations
import tf2_ros
from geometry_msgs.msg import PoseStamped
import math




class VIOBridgeArduPilot(Node):
    def __init__(self):
        super().__init__('vio_bridge_ardupilot')
        #publisher
        self.pose_publisher = self.create_publisher(PoseStamped, 'mavros/vision_pose/pose', 10)

        #parameters
        self.declare_parameter('world_frame_id', 'camera/odom')
        self.declare_parameter('base_frame_id', 'camera/base_link')
        self.declare_parameter('cam_yaw_offset', 0.0)
        self.declare_parameter('cam_pitch_offset', 0.0)
        self.declare_parameter('cam_roll_offset', 0.0)
        self.declare_parameter('gamma_world', 0.0)
        self.declare_parameter('pub_frequency', 20.0)

        self.timer = self.create_timer(1.0 / self.get_parameter('pub_frequency').get_parameter_value().double_value, self.timer_callback)
    
    def timer_callback(self):

        timout = rclpy.duration.Duration(seconds=3.0)

        if self.tf_buffer.can_transform(self.get_parameter('world_frame_id').get_parameter_value().string_value, self.get_parameter('base_frame_id').get_parameter_value().string_value, rclpy.time.Time(), timout):
            (trans, rot) = self.tf_buffer.lookup_transform(self.get_parameter('world_frame_id').get_parameter_value().string_value, self.get_parameter('base_frame_id').get_parameter_value().string_value, rclpy.time.Time())
            x, y , z = trans

            position_body_x = math.cos(self.get_parameter('gamma_world').get_parameter_value().double_value) * x + math.sin(self.get_parameter('gamma_world').get_parameter_value().double_value) * y
            position_body_y = -math.sin(self.get_parameter('gamma_world').get_parameter_value().double_value) * x + math.cos(self.get_parameter('gamma_world').get_parameter_value().double_value) * y
            position_body_z = z

            quat_cam = rot

            quat_x = tf_transformations.quaternion_from_euler(
                self.get_parameter('cam_roll_offset').get_parameter_value().double_value,0, 0)
            quat_y = tf_transformations.quaternion_from_euler(
                0, self.get_parameter('cam_pitch_offset').get_parameter_value().double_value, 0)
            quat_z = tf_transformations.quaternion_from_euler(
                0, 0, self.get_parameter('cam_yaw_offset').get_parameter_value().double_value)
            
            quat_rot_z = tf_transformations.quaternion_from_euler(0, 0, self.get_parameter('gamma_world').get_parameter_value().double_value)

            quat_body = tf_transformations.quaternion_multiply(quat_rot_z, quat_cam)
            quat_body = tf_transformations.quaternion_multiply(quat_body, quat_x)
            quat_body = tf_transformations.quaternion_multiply(quat_body, quat_y)
            quat_body = tf_transformations.quaternion_multiply(quat_body, quat_z)

            norm = math.sqrt(quat_body[0]**2 + quat_body[1]**2 + quat_body[2]**2 + quat_body[3]**2)
            quat_body = [quat_body[0]/norm, quat_body[1]/norm, quat_body[2]/norm, quat_body[3]/norm]

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
            pose_msg.pose.position.x = position_body_x
            pose_msg.pose.position.y = position_body_y
            pose_msg.pose.position.z = position_body_z

            pose_msg.pose.orientation.x = quat_body[0]
            pose_msg.pose.orientation.y = quat_body[1]
            pose_msg.pose.orientation.z = quat_body[2]
            pose_msg.pose.orientation.w = quat_body[3]

            self.pose_publisher.publish(pose_msg)
        else:
            self.get_logger().warn('Transform not available: {} to {}'.format(self.get_parameter('world_frame_id').get_parameter_value().string_value, self.get_parameter('base_frame_id').get_parameter_value().string_value))

def main(args=None):
    rclpy.init(args=args)
    node = VIOBridgeArduPilot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

