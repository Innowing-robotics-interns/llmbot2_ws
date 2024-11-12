from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage, PngImagePlugin
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Change this to your image topic
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.pil_image = None
        self.cv_image = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rgb_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        self.cv_image = rgb_image  # Store OpenCV image for later use
        self.pil_image = PILImage.fromarray(cv_image)

def get_class_mean(num_classes, labels):
    for i in range(num_classes):
        # calculate mean x, y position of labels in class:
        class_labels = np.where(labels == i)
        x_mean = np.mean(class_labels[1])
        y_mean = np.mean(class_labels[0])
        print(f"Class {i} mean x: {x_mean}, mean y: {y_mean}")


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
class RealSensePointCalculator(Node):
    def __init__(self, cam_frame_size = [480, 640], focal_scaler = 1.0, depth_scaler = 1.0, image_frame_size = [480, 480]):
        super().__init__('realsense_point_calculator')
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.info_callback, 10)
        self.depth_image = None
        self.depth_time = None
        self.intrinsics = None
        self.cam_frame_size = cam_frame_size
        self.focal_scaler = focal_scaler
        self.depth_scaler = depth_scaler
        self.x_offset = self.cam_frame_size[1] // 2 - image_frame_size[1] // 2
        self.y_offset = self.cam_frame_size[0] // 2 - image_frame_size[0] // 2

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_time = self.get_clock().now().nanoseconds * 1e-9

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
            self.destroy_subscription(self.info_sub)

    def info_received(self, del_time=0.5):
        if self.depth_image is None or self.intrinsics is None or self.get_clock().now().nanoseconds * 1e-9 - self.depth_time > del_time:
            return False
        else:
            return True

    def calculate_point(self, pixel_y, pixel_x, ):
        pixel_x += self.x_offset
        pixel_y += self.y_offset
        
        depth_pixel_y = int((pixel_y - self.cam_frame_size[0] // 2)/self.depth_scaler + self.cam_frame_size[0] // 2)
        depth_pixel_x = int((pixel_x - self.cam_frame_size[1] // 2)/self.depth_scaler + self.cam_frame_size[1] // 2)
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


# point cloud feature hash map
pcfm = {}
def update_pcfm(feat, point, camera_position):
    # update pcfm with feat and point
    # feat: feature vector of object
    # point: 3D point of object
    # pcfm: point cloud feature hash map
    pcfm[point] = feat
