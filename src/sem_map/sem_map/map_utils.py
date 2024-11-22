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
        self.translation = None
        self.rotation = None
        self.depth = None
        self.color = None
        self.pil_image = None
        self.info = None

    def socket_connect(self, port_num=5001):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', port_num))
        self.server_socket.listen(1)
        self.conn, self.addr = self.server_socket.accept()
    
    def send_handshake(self, handshake_message):
        print(f"Sending handshake message:{handshake_message}")
        self.conn.sendall(handshake_message.encode())
    
    def get_trans(self):
        data_valid = struct.unpack('<L', self.conn.recv(4))[0]
        # obtain 7 float
        data = self.conn.recv(12)
        self.translation = np.array(struct.unpack('<3f', data))
        data = self.conn.recv(16)
        self.rotation = np.array(struct.unpack('<4f', data))
        print("Received transformation data")
    
    def get_color(self):
        print("Waiting for image data...")
        data_size = struct.unpack('<L', self.conn.recv(4))[0]
        data = b''
        if data_size == 0:
            print("Received empty color data")
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
            print("Received color image")

    def get_depth(self):
        data_size = struct.unpack('<L', self.conn.recv(4))[0]
        data = b''
        if data_size == 0:
            print("Received empty depth data")
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
            print("Received depth image")
    
    def get_info(self):
        data_valid = struct.unpack('<L', self.conn.recv(4))[0]
        if data_valid == 0:
            print("Received empty info data")
        else:
            data = self.conn.recv(40)
            self.info = np.array(struct.unpack('<2I4d', data))
            print("Received info data")


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
        # key_pixels stores list of (feat_mean, [[pixel_y ...], [pixel_x...]])
        key_pixels.append((feat_mean, [class_pixel[0][indices], class_pixel[1][indices]]))
    return key_pixels

from sensor_msgs.msg import CameraInfo
import pyrealsense2 as rs
class RealSensePointCalculator():
    def __init__(self, depth_frame_size = [480, 640], image_frame_size = [480, 480]):
        self.bridge = CvBridge()
        self.depth_image = None
        self.intrinsics = rs.intrinsics()
        self.intrinsics.model = rs.distortion.none
        self.intrinsics.coeffs = [0.0 for i in range(5)]
        self.intr_set()
        self.depth_frame_size = depth_frame_size
        self.x_offset = self.depth_frame_size[1] // 2 - image_frame_size[1] // 2
        self.y_offset = self.depth_frame_size[0] // 2 - image_frame_size[0] // 2
    
    def update_depth(self, depth_img):
        self.depth_image = depth_img
    
    def update_intr(self, camera_info):
        self.intrinsics.width = int(camera_info[0])
        self.intrinsics.height = int(camera_info[1])
        self.intrinsics.ppx = camera_info[2]
        self.intrinsics.ppy = camera_info[3]
        self.intrinsics.fx = camera_info[4]
        self.intrinsics.fy = camera_info[5]

    def intr_set(self):
        # set the intrinsics of the D435i camera (obtain them from "ros2 topic echo /camera/camera/camera_info")
        self.intrinsics.width = 848
        self.intrinsics.height = 480
        self.intrinsics.ppx = 430.2650451660156
        self.intrinsics.ppy = 238.0896759033203
        self.intrinsics.fx = 425.21417236328125
        self.intrinsics.fy = 425.21417236328125

    def calculate_point(self, pixel_y, pixel_x):
        # Deprojection of depth camera pixel to 3D point
        depth_pixel_y = pixel_y + self.y_offset
        depth_pixel_x = pixel_x + self.x_offset
        # depth_pixel_y = pixel_y
        # depth_pixel_x = pixel_x
        
        # depth_pixel_y = int((pixel_y - self.cam_frame_size[0] // 2) + self.cam_frame_size[0] // 2)
        # depth_pixel_x = int((pixel_x - self.cam_frame_size[1] // 2) + self.cam_frame_size[1] // 2)
        depth = self.depth_image[depth_pixel_y, depth_pixel_x] * 0.001  # Convert from mm to meters
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth)
        point = [point[2], -point[0], -point[1]]
        # point = [point[2], point[1], -point[0]]

        return point
    
    def obtain_key_points(self, key_pixels):
        # Convert the key pixels to key points by
        # deprojecting the pixes to 3D point cloud using
        # depth of D435i camera
        key_points = []
        for feat_mean, pixels in key_pixels:
            points = []
            for i in range(len(pixels[0])):
                point = self.calculate_point(int(pixels[0][i]), int(pixels[1][i]))
                points.append(point)
            point_mean = np.mean(points, axis=0)
            key_points.append((feat_mean, point_mean))
        return key_points

from geometry_msgs.msg import TransformStamped
import tf_transformations
import pickle
class PointCloudFeatureMap():
    def __init__(self, round_to=0.2):
        self.round_to = round_to
        self.pcfm = {}

    def update_pcfm(self, key_points, translation, rotation, pcfm_threshold=4000, drop_range=0.5, drop_ratio=0.2):
        
        # Drop certain amount of feature points when a threshold is reached
        if len(self.pcfm) > pcfm_threshold:
            length = len(self.pcfm)
            key_i = np.arange(int(length*drop_range))
            drop_keys_i = np.random.choice(key_i, int(length * drop_ratio), replace=False)
            keys = list(self.pcfm.keys())
            for key_index in drop_keys_i:
                self.pcfm.pop(keys[key_index])
        
        # Obtain the inverse transform of translation and rotation and then translate point_mean using it
        # translation = -translation
        # rotation = tf_transformations.quaternion_inverse(rotation)

        # Update the feature map with the new key points (to the world frame)
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)[:3, :3]

        for feat_mean, point_mean in key_points:
            point_mean = np.dot(rotation_matrix, point_mean) + translation
            point_mean = point_mean // self.round_to * self.round_to
            self.pcfm[tuple(point_mean)] = feat_mean
    
    def save_pcfm(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.pcfm, f)

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
class PointCloudManager(Node):
    def __init__(self, topic_name='/point_cloud'):
        super().__init__('point_cloud_manager')
        self.publisher = self.create_publisher(PointCloud, topic_name, 10)
    
    def publish_point_cloud(self, pcfm_map_keys):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = 'map'
        for point in pcfm_map_keys:
            msg_point = Point32()
            msg_point.x = point[0]
            msg_point.y = point[1]
            msg_point.z = point[2]
            point_cloud_msg.points.append(msg_point)
        self.publisher.publish(point_cloud_msg)

    def publish_transformed_point_cloud(self, translation, rotation, pointcloud_list):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = 'map'
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)
        translation_matrix = tf_transformations.translation_matrix(translation)
        transformation = tf_transformations.concatenate_matrices(translation_matrix, rotation_matrix)
        self.get_logger().info(f"transformation matrix: {transformation}")
        transformation = tf_transformations.inverse_matrix(transformation)
        self.get_logger().info(f"transformation matrix: {transformation}")
        for point in pointcloud_list:
            point = np.append(point, 1)
            point = np.dot(transformation, point)
            
            msg_point = Point32()
            msg_point.x = point[0] + translation[0]
            msg_point.y = point[1] + translation[1]
            msg_point.z = point[2] + translation[2]
            point_cloud_msg.points.append(msg_point)
        self.publisher.publish(point_cloud_msg)


###### Testing Code ######
###| Test Display \###
# if socket_receiver.color is not None and socket_receiver.color.size > 0:
#     rclpy.logging.get_logger('update_map').info(str(socket_receiver.color.shape))
#     cv2.imshow('Color Image', cv2.cvtColor(socket_receiver.color, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(1)
# else:
#     rclpy.logging.get_logger('update_map').info("Received empty or invalid image.")
###\ Test Display |###

# display depth image
# cv2.imshow('Depth Image', socket_receiver.depth)
# cv2.waitKey(1)

# point_dict = {}
# for x in range(640):
#     for y in range(480):
#         if x % 10 == 0 and y % 10 == 0:
#             point_dict[tuple(rscalc.calculate_point(y, x))] = socket_receiver.depth[y, x]
# pc_manager.publish_point_cloud(point_dict)