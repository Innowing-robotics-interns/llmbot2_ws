import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import struct
import time
from sensor_msgs.msg import CameraInfo
import pyrealsense2 as rs

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber_socket_sender')

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw', 
            self.depth_callback, 
            10)
        
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.socket_setup()

    def socket_setup(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('zsc', 5000))  # Replace with your local machine's IP

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def send_color_image(self):
        if self.color_image is not None:
            try:
                self.get_logger().info("Sending image")
                _, buffer = cv2.imencode('.jpg', self.color_image)
                size = len(buffer)
                self.sock.sendall(struct.pack('<L', size) + buffer.tobytes())
            except Exception as e:
                self.get_logger().error(f"Failed to send image: {e}")
        else:
            # also send a message of the same format, but not the image
            self.get_logger().info("Sending empty color")
            self.sock.sendall(struct.pack('<L', 0))
    
    def send_depth_image(self):
        if self.depth_image is not None:
            try:
                self.get_logger().info("Sending depth")
                _, buffer = cv2.imencode('.png', self.depth_image)
                size = len(buffer)
                self.sock.sendall(struct.pack('<L', size) + buffer.tobytes())
            except Exception as e:
                self.get_logger().error(f"Failed to send depth: {e}")
        else:
            self.get_logger().info("Sending empty depth")
            self.sock.sendall(struct.pack('<L', 0))
    
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
    image_subscriber = ImageSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber, timeout_sec=6.0)
            image_subscriber.wait_handshake("color")
            image_subscriber.send_color_image()
            image_subscriber.wait_handshake("depth")
            image_subscriber.send_depth_image()
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        image_subscriber.close_socket()
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()