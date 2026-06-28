import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DroneStatusPublisher(Node):
    def __init__(self):
        super().__init__('drone_status_publisher')
        
        self.publisher = self.create_publisher(String, '/drone/status', 10)
        
        # Current state
        self.current_status = None
        
        # Timer to check status (you will update this based on your real logic)
        self.timer = self.create_timer(0.5, self.check_and_publish_status)  # 2 Hz

        self.get_logger().info("Drone Status Publisher Started - Publishing only on change")

    def update_status(self, new_status: str):
        """Call this function from your main code whenever status changes"""
        if new_status != self.current_status:
            self.current_status = new_status
            self.publish_status(new_status)

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.publisher.publish(msg)
        self.get_logger().info(f"Status Changed → {status}")

    def check_and_publish_status(self):
        """
        This is just an example.
        In real code, you should call update_status() whenever actual state changes.
        """
        # Example: Replace this with your real logic (SLAM, navigation, sensors, etc.)
        pass  # ← Remove this and use update_status() instead


# ====================== EXAMPLE USAGE ======================
def main(args=None):
    rclpy.init(args=args)
    node = DroneStatusPublisher()

    try:
        # === Example: Simulate state changes ===
        time.sleep(2)
        node.update_status("LANDED")

        time.sleep(5)
        node.update_status("DOCKED")

        time.sleep(15)
        node.update_status("CHARGING")

        time.sleep(150)
        node.update_status("CHARGED")

        time.sleep(5)
        node.update_status("IDLE")

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
