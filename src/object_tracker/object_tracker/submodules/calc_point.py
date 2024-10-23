import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs
import tf2_ros
import tf_transformations
from geometry_msgs.msg import TransformStamped, Quaternion

class RealSensePointCalculator(Node):
	def __init__(self, frame_size = [720, 1280], focal_scaler = 0.7, depth_scaler = 1.0):
		super().__init__('realsense_point_calculator')
		self.bridge = CvBridge()
		self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
		self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
		self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.info_callback, 10)
		self.depth_image = None
		self.color_image = None
		self.intrinsics = None
		self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
		self.frame_size = frame_size
		self.focal_scaler = focal_scaler
		self.depth_scaler = depth_scaler

	def depth_callback(self, msg):
		self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

	def color_callback(self, msg):
		self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

	def info_callback(self, msg):
		if self.intrinsics is None:
			self.intrinsics = rs.intrinsics()
			self.intrinsics.width = msg.width
			self.intrinsics.height = msg.height
			self.intrinsics.ppx = msg.k[2]
			self.intrinsics.ppy = msg.k[5]
			self.intrinsics.fx = msg.k[0]/self.focal_scaler
			self.intrinsics.fy = msg.k[4]/self.focal_scaler
			self.intrinsics.model = rs.distortion.none
			self.intrinsics.coeffs = [i for i in msg.d]

	def calculate_point(self, pixel_y, pixel_x):
		if self.depth_image is None or self.intrinsics is None:
			return None
		
		depth_pixel_y = int((pixel_y - self.frame_size[0] // 2)/self.depth_scaler + self.frame_size[0] // 2)
		depth_pixel_x = int((pixel_x - self.frame_size[1] // 2)/self.depth_scaler + self.frame_size[1] // 2)
		depth = self.depth_image[depth_pixel_y, depth_pixel_x] * 0.001  # Convert from mm to meters
		point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth)
		point = [point[2], -point[0], -point[1]]
		return point

	def publish_transform(self, point, frame_id='camera_link', child_frame_id='point_frame'):
		t = TransformStamped()
		t.header.stamp = self.get_clock().now().to_msg()
		t.header.frame_id = frame_id
		t.child_frame_id = child_frame_id
		t.transform.translation.x = point[0]
		t.transform.translation.y = point[1]
		t.transform.translation.z = point[2]
		t.transform.rotation = Quaternion()
		self.tf_broadcaster.sendTransform(t)

def main(args=None):
	rclpy.init(args=args)
	node = RealSensePointCalculator()
	print("begin")

	try:
		while rclpy.ok():
			rclpy.spin_once(node)
			pixel_x = 40  # Example pixel x-coordinate
			pixel_y = 60  # Example pixel y-coordinate
			point = node.calculate_point(pixel_x, pixel_y)
			if point:
				node.get_logger().info(f"3D point: {point}")
				node.publish_transform(point)
				

	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()