#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np


class ArucoErrorNode(Node):
    def __init__(self):
        super().__init__('aruco_error_node')

        # ================== PARAMETERS ==================
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('aruco_dict', 'DICT_6X6_250')

        camera_topic = self.get_parameter('camera_topic').value

        # ArUco setup
        dict_name = getattr(cv2.aruco, self.get_parameter('aruco_dict').value)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.bridge = CvBridge()

        # Subscriber & Publisher
        self.subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)

        self.publisher = self.create_publisher(
            Float32MultiArray, '/aruco_error', 10)

        self.get_logger().info(f'ArUco error node started! Listening on {camera_topic}')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(cv_image)

        height, width = cv_image.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0

        data = Float32MultiArray()

        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            marker_center_x = np.mean(marker_corners[:, 0])
            marker_center_y = np.mean(marker_corners[:, 1])

            error_x = float(marker_center_x - center_x)
            error_y = float(marker_center_y - center_y)

            data.data = [1.0, error_x, error_y]

            self.get_logger().debug(f'Detected: [1, {error_x:.1f}, {error_y:.1f}]')
        else:
            data.data = [0.0, 0.0, 0.0]

        self.publisher.publish(data)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoErrorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()