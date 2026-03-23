#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np


class ArucoErrorNode(Node):
    def __init__(self):
        super().__init__('aruco_error_node')

        # ================== PARAMETERS ==================
        self.declare_parameter('aruco_dict', 'DICT_6X6_250')

        # ArUco setup
        dict_name = getattr(cv2.aruco, self.get_parameter('aruco_dict').value)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Publisher
        self.publisher = self.create_publisher(
            Float32MultiArray, '/aruco_error', 10)

        # Open webcam
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
        else:
            self.get_logger().info("Camera started successfully.")

        # Timer to process frames
        self.timer = self.create_timer(0.03, self.process_frame)  # ~30 FPS

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning("Failed to read frame")
            return

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(frame)

        height, width = frame.shape[:2]
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

            # (Optional) draw for visualization
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (int(marker_center_x), int(marker_center_y)), 5, (0, 0, 255), -1)

        else:
            data.data = [0.0, 0.0, 0.0]

        self.publisher.publish(data)

        # Show image (optional but useful)
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoErrorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()