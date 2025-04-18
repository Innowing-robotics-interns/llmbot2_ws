import rclpy
from rclpy.node import Node
import socket
import pickle
from interfaces.msg import ColorDepthTrans
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
from .utils import read_config

class ImageTransformClient(Node):
    def __init__(self):
        super().__init__('image_transform_client')

        config_file = read_config("config_image_transform_client")
        self.port_num = config_file['socket_connection']['port_num']
        self.connect_to = config_file['socket_connection']['connect_to']
        self.update_interval = config_file['socket_connection']['update_interval']

        # Socket client setup
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.connect_to, self.port_num))
        self.get_logger().info("Connected to server.")

        self.pub = self.create_publisher(ColorDepthTrans, 'color_depth_trans', 10)
        self.color_pub = self.create_publisher(Image, 'color', 10)
        self.depth_pub = self.create_publisher(Image, 'depth', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.is_connected = True

        # Timer to periodically receive and process data
        self.timer = self.create_timer(self.update_interval, self.receive_data)

    def receive_data(self):
        if not self.is_connected:
            return
        try:
            # Receive size of the byte stream
            size_data = self.client_socket.recv(4)
            if not size_data:
                self.get_logger().warning("Server disconnected.")
                self.client_socket.close()
                return

            size = int.from_bytes(size_data, 'big')

            # Receive the byte stream
            byte_stream = b""
            while len(byte_stream) < size:
                packet = self.client_socket.recv(size - len(byte_stream))
                if not packet:
                    self.get_logger().warning("Server disconnected.")
                    self.client_socket.close()
                    return
                byte_stream += packet
            # Deserialize the data
            data = pickle.loads(byte_stream)

            # Separate images and transform
            color_image_msg = data.get("color_image")
            depth_image_msg = data.get("depth_image")
            transform = data.get("transform")

            self.publish_images(color_image_msg, depth_image_msg, transform)

            # Log the separated data
            # self.get_logger().info(f"Received Color Image: {color_image_msg}")
            # self.get_logger().info(f"Received Depth Image: {depth_image_msg}")
            # self.get_logger().info(f"Received Transform: {transform}")
        except ConnectionResetError as e:
            self.get_logger().error(f"Connection reset: {e}")
            self.cleanup_connection()
        except Exception as e:
            self.get_logger().error(f"Error receiving data: {e}")
            self.cleanup_connection()

    def publish_images(self, color_image, depth_image, trans):
        publish_message = ColorDepthTrans()
        publish_message.header.stamp = self.get_clock().now().to_msg()
        publish_message.header.frame_id = 'map'
        publish_message.height = 480
        publish_message.width = 640
        publish_message.encoding_color = 'rgb8'
        publish_message.is_bigendian_color = False
        publish_message.step_color = 1920
        publish_message.data_color = color_image.data
        publish_message.encoding_depth = '16UC1'
        publish_message.is_bigendian_depth = False
        publish_message.step_depth = 1280
        publish_message.data_depth = depth_image.data
        publish_message.translation_x = trans.transform.translation.x
        publish_message.translation_y = trans.transform.translation.y
        publish_message.translation_z = trans.transform.translation.z
        publish_message.rotation_x = trans.transform.rotation.x
        publish_message.rotation_y = trans.transform.rotation.y
        publish_message.rotation_z = trans.transform.rotation.z
        publish_message.rotation_w = trans.transform.rotation.w
        self.pub.publish(publish_message)

        color_msg = Image()
        color_msg.data = color_image.data
        color_msg.width = 640
        color_msg.height = 480
        color_msg.encoding = 'rgb8'
        color_msg.step = 1920
        color_msg.header.frame_id = 'color'
        color_msg.header.stamp = self.get_clock().now().to_msg()
        self.color_pub.publish(color_msg)
        
        depth_msg = Image()
        depth_msg.data = depth_image.data
        depth_msg.width = 640
        depth_msg.height = 480
        depth_msg.encoding = '16UC1'
        depth_msg.step = 1280
        depth_msg.header.frame_id = 'depth'
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

        # Publish a simple transform as an example
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_frame'
        t.transform.translation.x = trans.transform.translation.x
        t.transform.translation.y = trans.transform.translation.y
        t.transform.translation.z = trans.transform.translation.z
        t.transform.rotation.x = trans.transform.rotation.x
        t.transform.rotation.y = trans.transform.rotation.y
        t.transform.rotation.z = trans.transform.rotation.z
        t.transform.rotation.w = trans.transform.rotation.w
        self.tf_broadcaster.sendTransform(t)

    def cleanup_connection(self):
        if self.is_connected:
            self.is_connected = False
            self.timer.cancel()  # Stop the timer
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except Exception as e:
                self.get_logger().warning(f"Error closing socket: {e}")
    def destroy_node(self):
        if self.is_connected:
            self.cleanup_connection()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageTransformClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
