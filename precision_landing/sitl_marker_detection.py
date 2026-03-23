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
        self.declare_parameter('aruco_dict', 2)   # same as your working code
        self.declare_parameter('marker_id', -1)

        self.dictionary_id = self.get_parameter('aruco_dict').value
        self.marker_id = self.get_parameter('marker_id').value

        # ArUco setup (UPDATED)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Publisher
        self.publisher = self.create_publisher(
            Float32MultiArray, '/aruco_error', 10)

        # Camera
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Camera not opened!")
        else:
            self.get_logger().info("Camera started")

        # Loop
        self.timer = self.create_timer(0.03, self.process_frame)

    # ----------------------------------------------------
    def process_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        height, width = frame.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0

        # NEW DETECTION METHOD (from your working code)
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        msg = Float32MultiArray()

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Choose marker
            index = 0

            if self.marker_id != -1:
                for i, mid in enumerate(ids.flatten()):
                    if mid == self.marker_id:
                        index = i
                        break

            marker_corners = corners[index][0]

            marker_center_x = np.mean(marker_corners[:, 0])
            marker_center_y = np.mean(marker_corners[:, 1])

            error_x = float(marker_center_x - center_x)
            error_y = float(marker_center_y - center_y)

            msg.data = [1.0, error_x, error_y]

            # Debug draw
            cv2.circle(frame, (int(marker_center_x), int(marker_center_y)), 5, (0, 0, 255), -1)

        else:
            msg.data = [0.0, 0.0, 0.0]

        self.publisher.publish(msg)

        # Display
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)

    # ----------------------------------------------------
    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ArucoErrorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()