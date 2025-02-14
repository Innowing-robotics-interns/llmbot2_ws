import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import struct
import time

import tf2_ros
import rclpy

# from geometry_msgs.msg import TransformStamped
# import tf_transformations
import numpy as np


class SocketSender(Node):
    def __init__(self, camera_frame="camera_link", world_frame="map"):
        super().__init__("image_subscriber_socket_sender")

        self.image_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/aligned_depth_to_color/camera_info",
            self.info_callback,
            10,
        )

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.camera_frame = camera_frame
        self.world_frame = world_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_to_world = None

    def socket_setup(self, ip="fyp", port_num=5001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port_num))  # Replace with your local machine's IP

    def listen_tf(self):
        try:
            self.camera_to_world = self.tf_buffer.lookup_transform(
                self.camera_frame, self.world_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            self.get_logger().info("Cannot find camera to world transform")
            return False
        return True

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def info_callback(self, msg):
        self.camera_info = msg

    def send_transform(self):
        if self.camera_to_world is not None:
            translation = np.array(
                [
                    float(self.camera_to_world.transform.translation.x),
                    float(self.camera_to_world.transform.translation.y),
                    float(self.camera_to_world.transform.translation.z),
                ]
            )
            rotation = [
                float(self.camera_to_world.transform.rotation.x),
                float(self.camera_to_world.transform.rotation.y),
                float(self.camera_to_world.transform.rotation.z),
                float(self.camera_to_world.transform.rotation.w),
            ]
            self.get_logger().info(f"Sending transform")
            self.sock.sendall(
                struct.pack("<L", 1)
                + struct.pack("<3f", *translation)
                + struct.pack("<4f", *rotation)
            )
        else:
            translation = [0.0, 0.0, 0.0]
            rotation = [0.0, 0.0, 0.0, 0.0]
            self.get_logger().info("Sending empty transform")
            self.sock.sendall(
                struct.pack("<L", 0)
                + struct.pack("<3f", *translation)
                + struct.pack("<4f", *rotation)
            )

    def send_color_image(self):
        if self.color_image is not None:
            try:
                self.get_logger().info("Sending image")
                _, buffer = cv2.imencode(".jpg", self.color_image)
                size = len(buffer)
                self.sock.sendall(struct.pack("<L", size) + buffer.tobytes())
            except Exception as e:
                self.get_logger().error(f"Failed to send image: {e}")
        else:
            # also send a message of the same format, but not the image
            self.get_logger().info("Sending empty color")
            self.sock.sendall(struct.pack("<L", 0))

    def send_depth_image(self):
        if self.depth_image is not None:
            try:
                self.get_logger().info("Sending depth")
                _, buffer = cv2.imencode(".png", self.depth_image)
                size = len(buffer)
                self.sock.sendall(struct.pack("<L", size) + buffer.tobytes())
            except Exception as e:
                self.get_logger().error(f"Failed to send depth: {e}")
        else:
            self.get_logger().info("Sending empty depth")
            self.sock.sendall(struct.pack("<L", 0))

    def send_camera_info(self):
        if self.camera_info is not None:
            self.get_logger().info("Sending camera info")
            self.sock.sendall(struct.pack("<L", 1))
            print(
                self.camera_info.height,
                self.camera_info.width,
                float(self.camera_info.k[2]),
                float(self.camera_info.k[5]),
                float(self.camera_info.k[0]),
                float(self.camera_info.k[4]),
            )
            self.sock.sendall(
                struct.pack(
                    "<2I4d",
                    self.camera_info.height,
                    self.camera_info.width,
                    self.camera_info.k[2],
                    self.camera_info.k[5],
                    self.camera_info.k[0],
                    self.camera_info.k[4],
                )
            )
        else:
            self.get_logger().info("Sending empty camera info")
            self.sock.sendall(struct.pack("<L", 0))

    def wait_handshake(self, handshake_msg="handshake"):
        self.get_logger().info(f"Waiting for handshake")
        handshake = self.sock.recv(1024).decode()
        if handshake != handshake_msg:
            self.get_logger().error("Handshake failed")
            self.get_logger().error(f"Received: {handshake}, expect: {handshake_msg}")
            return False
        else:
            self.get_logger().info("Handshake successful")
            return True

    def close_socket(self):
        self.sock.close()


def main(args=None):
    rclpy.init(args=args)
    socket_sender = SocketSender()
    c = 0
    try:
        socket_sender.socket_setup()
        while rclpy.ok():
            c += 1
            rclpy.logging.get_logger("image_socket_send").info(f"Number: {c}")
            rclpy.spin_once(socket_sender, timeout_sec=6.0)
            # socket_sender.listen_tf()
            socket_sender.wait_handshake("info")
            socket_sender.send_camera_info()
            socket_sender.wait_handshake("trans")
            socket_sender.send_transform()
            socket_sender.wait_handshake("color")
            socket_sender.send_color_image()
            socket_sender.wait_handshake("depth")
            socket_sender.send_depth_image()
    except KeyboardInterrupt:
        pass
    finally:
        socket_sender.close_socket()
        socket_sender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
