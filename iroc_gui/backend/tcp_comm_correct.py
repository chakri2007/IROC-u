import socket
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

HOST = '192.168.0.114'
PORT = 55555

class TCPCommNode(Node):
    def __init__(self):
        super().__init__('tcp_comm_node')

        self.last_status = None
        self.docking_in_progress = False

        # Subscribe to drone status
        self.subscription = self.create_subscription(
            String, '/drone/status', self.drone_status_callback, 10
        )

        # Publish to system
        self.publisher = self.create_publisher(String, '/gcs/status', 10)

        # NEW: Publish status updates
        self.status_update_pub = self.create_publisher(
            String,
            '/drone/status_update',
            10
        )

        # TCP Connection
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, PORT))
        self.get_logger().info('Connected to Ground Control Station (RPi)')

        # Listen thread
        listen_thread = threading.Thread(
            target=self.listen_to_gcs,
            daemon=True
        )
        listen_thread.start()

    # =================================================
    def drone_status_callback(self, msg):
        status = msg.data.strip().upper()

        if status == self.last_status:
            return  # Ignore duplicate same status

        self.last_status = status
        self.get_logger().info(f"New Drone Status: {status}")

        if status == "LANDED":

            self.get_logger().info(
                "Drone landed. Starting docking procedure."
            )

            self.send_command("START_DOCKING")

            

        elif status == "DOCKED":
            self.send_command("START_CHARGING 5")

        elif status == "CHARGED":
            self.send_command("UNDOCK")

        #need to add the part that if sufficient no. of seeds are not detected than have to go

        elif status == "UNDOCKED":
            self.send_command("IDLE")

    # =================================================
    def send_command(self, command):
        try:
            self.get_logger().info(f'Sending → {command}')
            data = command.encode('utf-8')

            self.client_socket.sendall(
                len(data).to_bytes(4, 'big')
            )
            self.client_socket.sendall(data)

        except Exception as e:
            self.get_logger().error(f'Send Error: {e}')

    # =================================================
    def listen_to_gcs(self):
        while True:
            try:
                len_bytes = self.client_socket.recv(4)

                if not len_bytes:
                    break

                length = int.from_bytes(len_bytes, 'big')

                message = self.client_socket.recv(length)\
                    .decode('utf-8')\
                    .strip()

                self.get_logger().info(
                    f'GCS Response: {message}'
                )

                ros_msg = String()

                if "Docking successful" in message:
                    ros_msg.data = "DOCKED"
                    self.docking_in_progress = False
                    update_msg = String()
                    update_msg.data = "docked"
                    self.status_update_pub.publish(update_msg)

                    self.get_logger().info(
                        "Published to /drone/status_update: docked"
                    )

                elif "Charging started" in message or \
                     "Charging with" in message:

                    ros_msg.data = "CHARGING"
                    update_msg = String()
                    update_msg.data = "charging"
                    self.status_update_pub.publish(update_msg)

                    self.get_logger().info(
                        "Published to /drone/status_update: charging"
                    )

                elif "Charging stopped" in message or \
                     "stopped" in message.lower():

                    ros_msg.data = "CHARGED"
                    update_msg = String()
                    update_msg.data = "charged"
                    self.status_update_pub.publish(update_msg)

                    self.get_logger().info(
                        "Published to /drone/status_update: charged"
                    )

                elif "Undocking" in message:

                    ros_msg.data = "UNDOCKED"

                    # NEW: Publish undocked status update
                    update_msg = String()
                    update_msg.data = "undocked"
                    self.status_update_pub.publish(update_msg)

                    self.get_logger().info(
                        "Published to /drone/status_update: undocked"
                    )

                else:
                    ros_msg.data = message

                self.publisher.publish(ros_msg)

                self.get_logger().info(
                    f'Published to /gcs/status: {ros_msg.data}'
                )

            except Exception as e:
                self.get_logger().error(
                    f'Receive Error: {e}'
                )
                break

    # =================================================
    def destroy_node(self):
        self.client_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TCPCommNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
