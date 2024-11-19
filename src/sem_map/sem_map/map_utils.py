from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage, PngImagePlugin
import numpy as np
import socket
import struct

class SocketReceiver():
    def __init__(self):
        self.server_socket = None
        self.conn, self.addr = None, None
        self.depth = None
        self.color = None
        self.pil_image = None

    def socket_connect(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', 5000))
        self.server_socket.listen(1)
        self.conn, self.addr = self.server_socket.accept()
    
    def send_handshake(self, handshake_message):
        print(f"Sending handshake message:{handshake_message}")
        self.conn.sendall(handshake_message.encode())
    
    def get_color(self):
        print("Waiting for image data...")
        data_size = struct.unpack('<L', self.conn.recv(4))[0]
        data = b''
        if data_size == 0:
            self.get_logger().info("Received empty color data")
            pass
        else:
            while len(data) < data_size:
                packet = self.conn.recv(4096)
                if not packet:
                    break
                data += packet
            if len(data) == data_size:
                np_array = np.frombuffer(data, np.uint8)
                color_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                self.color = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                self.pil_image = PILImage.fromarray(self.color)
            else:
                raise Exception("Color image size does not match")
            self.get_logger().info("Received color image")

    def get_depth(self):
        data_size = struct.unpack('<L', self.conn.recv(4))[0]
        data = b''
        if data_size == 0:
            self.get_logger().info("Received empty depth data")
            pass
        else:
            # Receive the image data
            while len(data) < data_size:
                packet = self.conn.recv(4096)
                if not packet:
                    break
                data += packet

            if len(data) == data_size:
                np_array = np.frombuffer(data, np.uint8)
                self.depth = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            else:
                raise Exception("Depth image size does not match")
            self.get_logger().info("Received depth image")

from skimage.measure import label
def relabel_connected_components(class_image, n_classes=10):
    # Initialize an output image with the same shape
    relabeled_image = np.zeros_like(class_image)

    for class_label in range(n_classes):
        mask = (class_image == class_label)
        labeled_mask, num_features = label(mask, connectivity=2, return_num=True)
        relabeled_mask = labeled_mask + (relabeled_image.max() + 1) * (labeled_mask > 0)
        relabeled_image += relabeled_mask
    return relabeled_image
# example use
# relabeled_image = relabel_connected_components(clustered_image.squeeze())

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def PCA_and_Cluster(feat, n_components=20, n_clusters=10, H=480, W=480, D=512):
    pca = PCA(n_components=n_components)
    # squeeze first two dimension of feat 
    feat_map = feat.flatten(start_dim=0, end_dim=1).detach().cpu().numpy()
    features = pca.fit_transform(feat_map)
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(features)
    # Reshape the labels back to the original image shape
    clustered_image = labels.reshape(H, W)
    clustered_image = relabel_connected_components(clustered_image, n_classes=n_clusters)
    return clustered_image

def obtain_key_pixels(feat, clustered_image, n_pixels=30, rule_out_threshold=500):
    num_class = clustered_image.max() + 1
    key_pixels = []
    for i in range(num_class):
        class_feat = feat[clustered_image == i]
        if len(class_feat) == 0:
            continue
        feat_mean = class_feat.mean(dim=0)
        class_pixel = np.where(clustered_image == i)
        if len(class_pixel[0]) < rule_out_threshold:
            continue
        indices = np.random.choice(len(class_pixel[0]), n_pixels, replace=False)
        key_pixels.append((feat_mean, [class_pixel[0][indices], class_pixel[1][indices]]))
    return key_pixels

from sensor_msgs.msg import CameraInfo
import pyrealsense2 as rs
class RealSensePointCalculator():
    def __init__(self, cam_frame_size = [480, 640], image_frame_size = [480, 480]):
        self.bridge = CvBridge()
        self.depth_image = None
        self.intrinsics = None
        self.intr_set()
        self.cam_frame_size = cam_frame_size
        self.x_offset = self.cam_frame_size[1] // 2 - image_frame_size[1] // 2
        self.y_offset = self.cam_frame_size[0] // 2 - image_frame_size[0] // 2
    
    def update_depth(self, depth_img):
        self.depth_image = depth_img

    def intr_set(self):
        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = 848
            self.intrinsics.height = 480
            self.intrinsics.ppx = 430.2650451660156
            self.intrinsics.ppy = 238.0896759033203
            self.intrinsics.fx = 425.21417236328125
            self.intrinsics.fy = 425.21417236328125
            self.intrinsics.model = rs.distortion.none
            self.intrinsics.coeffs = [0.0 for i in range(5)]

    def calculate_point(self, pixel_y, pixel_x):
        depth_pixel_x = pixel_x + self.x_offset
        depth_pixel_y = pixel_y + self.y_offset
        
        # depth_pixel_y = int((pixel_y - self.cam_frame_size[0] // 2) + self.cam_frame_size[0] // 2)
        # depth_pixel_x = int((pixel_x - self.cam_frame_size[1] // 2) + self.cam_frame_size[1] // 2)
        depth = self.depth_image[depth_pixel_y, depth_pixel_x] * 0.001  # Convert from mm to meters
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth)
        point = [point[2], -point[0], -point[1]]
        return point
    
    def obtain_key_points(self, key_pixels):
        key_points = []
        for feat_mean, pixels in key_pixels:
            points = []
            for i in range(len(pixels[0])):
                point = self.calculate_point(float(pixels[0][i]), float(pixels[1][i]))
                points.append(point)
            point_mean = np.mean(points, axis=0)
            key_points.append((feat_mean, point_mean))
        return key_points

import tf2_ros
import rclpy
from geometry_msgs.msg import TransformStamped
import tf_transformations
import pickle
class PointCloudFeatureMap(Node):
    def __init__(self, round_to=0.2, camera_frame='camera_link', world_frame='map'):
        super().__init__('point_cloud_feature_map')
        self.round_to = round_to
        self.camera_frame = camera_frame
        self.world_frame = world_frame
        self.pcfm = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_to_world = None
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9
        self.curr_time = self.get_clock().now().nanoseconds * 1e-9

    def listen_tf(self):
        try:
            self.camera_to_world = self.tf_buffer.lookup_transform(self.camera_frame, self.world_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info('Cannot find camera to world transform')
            return False
        return True    

    def update_pcfm(self, key_points, pcfm_threshold=4000, drop_range=0.5, drop_ratio=0.2):
        if self.camera_to_world is not None:
            if len(self.pcfm) > pcfm_threshold:
                length = len(self.pcfm)
                key_i = np.arange(int(length*drop_range))
                drop_keys_i = np.random.choice(key_i, int(length * drop_ratio), replace=False)
                keys = list(self.pcfm.keys())
                for key_index in drop_keys_i:
                    self.pcfm.pop(keys[key_index])

            translation = np.array([self.camera_to_world.transform.translation.x,
                                    self.camera_to_world.transform.translation.y,
                                    self.camera_to_world.transform.translation.z])
            rotation = [self.camera_to_world.transform.rotation.x,
                        self.camera_to_world.transform.rotation.y,
                        self.camera_to_world.transform.rotation.z,
                        self.camera_to_world.transform.rotation.w]
            rotation_matrix = tf_transformations.quaternion_matrix(rotation)[:3, :3]

            for feat_mean, point_mean in key_points:
                point_mean = np.dot(rotation_matrix, point_mean) + translation
                point_mean = point_mean // self.round_to * self.round_to
                self.pcfm[tuple(point_mean)] = feat_mean
            return True
        else:
            return False
    
    def save_pcfm(self, file_name, update_interval=10):
        self.curr_time = self.get_clock().now().nanoseconds * 1e-9
        if self.curr_time - self.prev_time > update_interval:
            with open(file_name, 'wb') as f:
                pickle.dump(self.pcfm, f)
            self.prev_time = self.curr_time
            return True
        return False