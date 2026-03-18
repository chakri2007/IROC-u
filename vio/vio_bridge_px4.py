import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from mavros_msgs.msg import CompanionProcessStatus

import numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply


class VIOBridge(Node):

    def __init__(self):
        super().__init__('vio_bridge')

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

        # Rotation matrix ENU → NED (matches your transform)
        self.R = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

        self.get_logger().info("VIO Bridge with Covariance Rotation Started")

    # =========================
    # Covariance Rotation (6x6)
    # =========================
    def rotate_covariance_6x6(self, cov_in):
        cov_in = np.array(cov_in).reshape(6, 6)

        R = self.R

        # Position covariance
        pos_in = cov_in[0:3, 0:3]
        pos_out = R @ pos_in @ R.T

        # Orientation covariance (approx using same R)
        att_in = cov_in[3:6, 3:6]
        att_out = R @ att_in @ R.T

        # Cross covariance
        cross_in = cov_in[0:3, 3:6]
        cross_out = R @ cross_in @ R.T

        # Reconstruct
        cov_out = np.zeros((6, 6))
        cov_out[0:3, 0:3] = pos_out
        cov_out[3:6, 3:6] = att_out
        cov_out[0:3, 3:6] = cross_out
        cov_out[3:6, 0:3] = cross_out.T

        return cov_out.flatten().tolist()

    def odom_callback(self, msg):

        # =========================
        # 1. Position ENU → NED
        # =========================
        x_enu = msg.pose.pose.position.x
        y_enu = msg.pose.pose.position.y
        z_enu = msg.pose.pose.position.z

        x_ned = y_enu
        y_ned = x_enu
        z_ned = -z_enu

        # =========================
        # 2. Orientation ENU → NED
        # =========================
        q = msg.pose.pose.orientation
        q_enu = np.array([q.x, q.y, q.z, q.w])
        
        q_cam = quaternion_from_euler(0,-np.pi/2, 0)  # Rotate from camera to ENU
        q_rot = quaternion_from_euler(np.pi, 0, np.pi / 2)
        q_corrected = quaternion_multiply(q_cam, q_enu)  # Apply camera correction
        q_ned = quaternion_multiply(q_rot, q_corrected)

        # =========================
        # 3. Velocity ENU → FRD
        # =========================
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        vx_frd = vy
        vy_frd = vx
        vz_frd = -vz

        # =========================
        # 4. Rotate Covariances
        # =========================
        pose_cov_out = self.rotate_covariance_6x6(msg.pose.covariance)
        twist_cov_out = self.rotate_covariance_6x6(msg.twist.covariance)

        # =========================
        # 5. Build output message
        # =========================
        out = Odometry()

        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_link"

        # Position
        out.pose.pose.position.x = x_ned
        out.pose.pose.position.y = y_ned
        out.pose.pose.position.z = z_ned

        # Orientation
        out.pose.pose.orientation.x = q_ned[0]
        out.pose.pose.orientation.y = q_ned[1]
        out.pose.pose.orientation.z = q_ned[2]
        out.pose.pose.orientation.w = q_ned[3]

        # Velocity
        out.twist.twist.linear.x = vx_frd
        out.twist.twist.linear.y = vy_frd
        out.twist.twist.linear.z = vz_frd

        # Covariances (FIXED)
        out.pose.covariance = pose_cov_out
        out.twist.covariance = twist_cov_out

        # Publish odometry
        self.odom_pub.publish(out)

        # =========================
        # 6. Publish status
        # =========================
        status = CompanionProcessStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.component = 197  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
        status.state = CompanionProcessStatus.MAV_STATE_ACTIVE

        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = VIOBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()