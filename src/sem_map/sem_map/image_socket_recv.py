from .utils import DepthCamSocketMaintainer
import rclpy
from rclpy.node import Node
import traceback
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
from interfaces.msg import ColorDepthTrans
import time

port_num = 8812

class SocketPublisher(Node):
    def __init__(self):
        super().__init__('socket_publisher')
        self.pub = self.create_publisher(ColorDepthTrans, 'color_depth_trans', 10)

        self.color_pub = self.create_publisher(Image, 'color', 10)
        self.depth_pub = self.create_publisher(Image, 'depth', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    
    def publish_images(self, color_image, depth_image, trans_array):
        publish_message = ColorDepthTrans()
        publish_message.header.stamp = self.get_clock().now().to_msg()
        publish_message.header.frame_id = 'map'
        publish_message.height = 480
        publish_message.width = 640
        publish_message.encoding_color = 'rgb8'
        publish_message.is_bigendian_color = False
        publish_message.step_color = 1920
        publish_message.data_color = color_image.tobytes()
        publish_message.encoding_depth = '16UC1'
        publish_message.is_bigendian_depth = False
        publish_message.step_depth = 1280
        publish_message.data_depth = depth_image.tobytes()
        publish_message.translation_x = trans_array[0]
        publish_message.translation_y = trans_array[1]
        publish_message.translation_z = trans_array[2]
        publish_message.rotation_x = trans_array[3]
        publish_message.rotation_y = trans_array[4]
        publish_message.rotation_z = trans_array[5]
        publish_message.rotation_w = trans_array[6]
        self.pub.publish(publish_message)

        color_msg = Image()
        color_msg.data = color_image.tobytes()
        color_msg.width = 640
        color_msg.height = 480
        color_msg.encoding = 'rgb8'
        color_msg.step = 1920
        color_msg.header.frame_id = 'color'
        color_msg.header.stamp = self.get_clock().now().to_msg()
        self.color_pub.publish(color_msg)
        
        depth_msg = Image()
        depth_msg.data = depth_image.tobytes()
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
        t.transform.translation.x = trans_array[0]
        t.transform.translation.y = trans_array[1]
        t.transform.translation.z = trans_array[2]
        t.transform.rotation.x = trans_array[3]
        t.transform.rotation.y = trans_array[4]
        t.transform.rotation.z = trans_array[5]
        t.transform.rotation.w = trans_array[6]
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    dcm = DepthCamSocketMaintainer()
    sp = SocketPublisher()
    sp.get_logger().info(f"waiting for socket connection in {port_num}")
    dcm.socket_connect(port_num=port_num)
    print("Socket connected")
    try:
        while rclpy.ok():
            dcm.send_handshake("depth")
            dcm.receive_depth()
            depth_image = dcm.return_depth()
            dcm.send_handshake("color")
            dcm.receive_color()
            color_image = dcm.return_color()
            dcm.send_handshake("trans")
            dcm.receive_trans()
            trans_array = dcm.return_trans()
            sp.publish_images(color_image, depth_image, trans_array)
            # time.sleep(0.3)

    except KeyboardInterrupt:
        print("Shutting down due to KeyboardInterrupt")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        dcm.socket_close()
        print("Socket closed")

    rclpy.shutdown()
