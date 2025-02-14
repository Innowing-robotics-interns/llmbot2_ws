from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage, PngImagePlugin
import numpy as np
import socket
import struct
from skimage.measure import label
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pyrealsense2 as rs
import tf_transformations
import pickle
import torchvision.transforms as transforms
import torch

'''
2. remember to add obtain_key_points
'''

class SocketReceiver:
    '''
    A class to handle socket communication for receiving transformation, color, depth, and info data.

    Attributes:
        server_socket (socket.socket): The server socket object.
        conn (socket.socket): The connection socket object.
        addr (tuple): The address bound to the socket.
        translation (numpy.ndarray): The translation data received from the socket.
        rotation (numpy.ndarray): The rotation data received from the socket.
        depth (numpy.ndarray): The depth image data received from the socket.
        color (numpy.ndarray): The color image data received from the socket.
        pil_image (PIL.Image.Image): The color image in PIL format.
        info (numpy.ndarray): The additional info data received from the socket.

    Methods:
        socket_connect(port_num=5001): Establishes a socket connection on the given port.
        send_handshake(handshake_message): Sends a handshake message to the connected client.
        get_data(variable_length=False, formats=["<3f", "<4f"]): Receives data from the socket.
    '''
    def __init__(self):
        self.server_socket = None
        self.conn, self.addr = None, None

    def socket_connect(self, port_num=5001):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", port_num))
        self.server_socket.listen(1)
        self.conn, self.addr = self.server_socket.accept()

    def send_handshake(self, handshake_message):
        print(f"Sending handshake message:{handshake_message}")
        self.conn.sendall(handshake_message.encode())

    def get_data(self, variable_length=False, formats="<3f<4f"):
        '''
        Receives data from the socket, either of variable length or fixed length based on provided formats.
        if the data is fixed length, the sender needs to send a valid data flag 1 of "<L" before sending the data.
        if the data is variable length, the sender needs to send the data size of "<L" before sending the data.

        Args:
            variable_length (bool): If True, receives data of variable length. If False, receives fixed length data.
            formats (string): A struct format string to unpack the data.

        Returns:
            data (bytes or list): The received data. 
                                  If variable_length is True, returns the raw data bytes.
                                  If variable_length is False, returns a list of unpacked numpy arrays.
                                  e.g. [np.array, np.array, ...]
        '''
        if variable_length:
            data_size = struct.unpack("<L", self.conn.recv(4))[0]
            data = b""
            if data_size == 0:
                print("Received empty data")
                return None
            else:
                while len(data) < data_size:
                    packet = self.conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                if len(data) == data_size:
                    return data
                else:
                    raise Exception("Data size does not match")
        else:
            if len(formats) == 0:
                print("No format provided")
                return None
            data_valid = struct.unpack("<L", self.conn.recv(4))[0]
            if data_valid == 1:
                data = self.conn.recv(struct.calcsize(formats))
                if len(data) == struct.calcsize(formats):
                    return np.array(struct.unpack(formats, data))
                else:
                    print("Received invalid data")
                    return None
            else:
                print("Received invalid data")
                return None

class RealSensePointCalculator:
    def __init__(self, depth_frame_size=[480, 640], image_frame_size=[480, 480]):
        self.bridge = CvBridge()
        self.depth_image = None
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = 848
        self.intrinsics.height = 480
        self.intrinsics.ppx = 430.2650451660156
        self.intrinsics.ppy = 238.0896759033203
        self.intrinsics.fx = 425.21417236328125
        self.intrinsics.fy = 425.21417236328125
        self.intrinsics.model = rs.distortion.none
        self.intrinsics.coeffs = [0.0 for i in range(5)]
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

    def calculate_point(self, pixel_y, pixel_x):
        # Deprojection of depth camera pixel to 3D point
        depth_pixel_y = pixel_y + self.y_offset
        depth_pixel_x = pixel_x + self.x_offset
        # depth_pixel_y = pixel_y
        # depth_pixel_x = pixel_x

        # depth_pixel_y = int((pixel_y - self.cam_frame_size[0] // 2) + self.cam_frame_size[0] // 2)
        # depth_pixel_x = int((pixel_x - self.cam_frame_size[1] // 2) + self.cam_frame_size[1] // 2)
        depth = (
            self.depth_image[depth_pixel_y, depth_pixel_x] * 0.001
        )  # Convert from mm to meters
        point = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [pixel_x, pixel_y], depth
        )
        point = [point[2], -point[0], -point[1]]
        return point

class FeatImageProcessor:
    def relabel_connected_components(self, class_image, n_classes=10):
        # Initialize an output image with the same shape
        relabeled_image = np.zeros_like(class_image)

        for class_label in range(n_classes):
            mask = class_image == class_label
            labeled_mask, num_features = label(mask, connectivity=2, return_num=True)
            relabeled_mask = labeled_mask + (relabeled_image.max() + 1) * (labeled_mask > 0)
            relabeled_image += relabeled_mask
        return relabeled_image

    def PCA(self, feat, n_components=20):
        '''
        Applies PCA to reduce the dimensionality of the feature map.

        Args:
            feat (torch.Tensor): The input feature map.
            n_components (int): The number of principal components to keep.

        Returns:
            numpy.ndarray: The transformed feature map with reduced dimensionality.
        '''
        pca = PCA(n_components=n_components)
        feat_map = feat.flatten(start_dim=0, end_dim=1).detach().cpu().numpy()
        features = pca.fit_transform(feat_map)
        return features

    def Cluster(self, features, n_clusters=10, H=480, W=480):
        '''
        Applies K-Means clustering to the feature map.

        Args:
            features (numpy.ndarray): The input feature map with reduced dimensionality.
            n_clusters (int): The number of clusters to form.
            H (int): The height of the original image.
            W (int): The width of the original image.

        Returns:
            numpy.ndarray: The clustered image.
        '''
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(features)
        clustered_image = labels.reshape(H, W)
        clustered_image = self.relabel_connected_components(clustered_image, n_classes=n_clusters)
        return clustered_image

    def obtain_key_pixels(self, feat, clustered_image, n_pixels=30, rule_out_threshold=500):
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
            key_pixels.append(
                (feat_mean, [class_pixel[0][indices], class_pixel[1][indices]])
            )
        return key_pixels

class ServerFeaturePointCloudMap:
    def __init__(self, round_to=0.2):
        self.round_to = round_to
        self.fpc = {}
        self.info = None
        self.trans = None
        self.pil_image = None
        self.depth = None
        self.socket_receiver = SocketReceiver()
        self.rscalc = RealSensePointCalculator()
        self.fip = FeatImageProcessor()

        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    480
                ),  # Resize the shorter side to 480 while maintaining aspect ratio
                transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
                transforms.ToTensor(),  # Convert to tensor
            ]
        )
        self.current_feature = None
    
    def receive_info(self):
        self.info = self.socket_receiver.receive_data(variable_length=False, formats="<2I4d")
    
    def receive_trans(self):
        # first 3 floats are translation, next 4 floats are rotation
        self.trans = self.socket_receiver.receive_data(variable_length=False, formats="<3f<4f")
    
    def receive_color(self):
        data = self.socket_receiver.receive_data(variable_length=True)
        if data is not None:
            np_array = np.frombuffer(data, np.uint8)
            color_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            color = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            self.pil_image = PILImage.fromarray(color)
        else:
            print("Received empty color data")
            return None
    
    def receive_depth(self):
        data = self.socket_receiver.get_data(variable_length=True)
        if data is not None:
            np_array = np.frombuffer(data, np.uint8)
            self.depth = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
        else:
            print("Received empty depth data")
            return None
    
    def handshake_receive_data(self):
        self.socket_receiver.send_handshake("info")
        self.socket_receiver.receive_info()
        self.socket_receiver.send_handshake("trans")
        self.socket_receiver.receive_trans()
        self.socket_receiver.send_handshake("color")
        self.socket_receiver.receive_color()
        self.socket_receiver.send_handshake("depth")
        self.socket_receiver.receive_depth()
    
    def set_model(self, model):
        self.model = model
    
    def update_feature(self):
        image_tensor = self.transform(self.pil_image)
        with torch.no_grad():
            feat = self.model(image_tensor.unsqueeze(0).cuda())
        self.current_feature = feat.half()

    def obtain_key_points(self, key_pixels):
        # Convert the key pixels to key points by
        # deprojecting the pixes to 3D point cloud using
        # depth of D435i camera
        key_points = []
        for feat_mean, pixels in key_pixels:
            points = []
            for i in range(len(pixels[0])):
                point = self.rscalc.calculate_point(int(pixels[0][i]), int(pixels[1][i]))
                points.append(point)
            point_mean = np.mean(points, axis=0)
            key_points.append((feat_mean, point_mean))
        return key_points
    
    def feat_to_points(self):
        features = self.fip.PCA(self.current_feature)
        clustered_image = self.fip.Cluster(features)
        key_pixels = self.fip.obtain_key_pixels(self.current_feature, clustered_image)
        self.rscalc.update_depth(self.depth)
        key_points = self.obtain_key_points(key_pixels)
        return key_points

    def update_fpc(
        self,
        key_points,
        translation,
        rotation,
        fpc_threshold=4000,
        drop_range=0.5,
        drop_ratio=0.2,
    ):

        # Drop certain amount of feature points when a threshold is reached
        if len(self.fpc) > fpc_threshold:
            length = len(self.fpc)
            key_i = np.arange(int(length * drop_range))
            drop_keys_i = np.random.choice(
                key_i, int(length * drop_ratio), replace=False
            )
            keys = list(self.fpc.keys())
            for key_index in drop_keys_i:
                self.fpc.pop(keys[key_index])

        # Update the feature map with the new key points (to the world frame)
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)[:3, :3]

        for feat_mean, point_mean in key_points:
            point_mean = np.dot(rotation_matrix, point_mean) + translation
            point_mean = point_mean // self.round_to * self.round_to
            self.fpc[tuple(point_mean)] = feat_mean
    
    def save_fpc(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.fpc, f)


from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32

class PointCloudManager(Node):
    def __init__(self, topic_name="/point_cloud"):
        super().__init__("point_cloud_manager")
        self.publisher = self.create_publisher(PointCloud, topic_name, 10)

    def publish_point_cloud(self, pcfm_map_keys):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "map"
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
        point_cloud_msg.header.frame_id = "map"
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)
        translation_matrix = tf_transformations.translation_matrix(translation)
        transformation = tf_transformations.concatenate_matrices(
            translation_matrix, rotation_matrix
        )
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

'''
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
'''