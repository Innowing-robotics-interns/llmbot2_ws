import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from interfaces.srv import SemanticQuery
from geometry_msgs.msg import Point

from .utils import *
from .image_processors import *
from interfaces.msg import ColorDepthTrans

import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import threading

class SemanticPoint:
    def __init__(self, feat, label):
        '''
        Args:
            feat (1D torch tensor): The feature of the point.
            label (str): The label of the point.
        '''
        self.feat = feat
        self.label = label
        
class SemanticMapCore():
    def __init__(self, 
    image_semantic_extractor, 
    max_depth_threshold=2e3, 
    erase_point_depth_threshold=40000, 
    erase_point_depth_tolerance=100,
    round_points_to=0.05,
    z_axis_lower_bound=0.5,
    z_axis_upper_bound=2.0):
        self.pil_image = None
        self.depth = None
        self.trans = None

        self.max_depth_threshold = max_depth_threshold # in mm

        # if the robot is moving fast, trans and depth might not be synchronous,
        # so record previous and current trans and depth (also pil_image, since why not)
        # so that we can compare the two frames and decide whether to erase the point or not
        self.prev_depth = None
        self.prev_trans = None
        self.prev_pil_image = None
        self.erase_point_depth_threshold = erase_point_depth_threshold # in mm
        # self.erase_point_trans_threshold = 3

        self.image_semantic_extractor = image_semantic_extractor
        self.semantic_point_cloud = {} # dict in the format of (x,y,z):SemanticPoint
        self.rscalc = RealSensePointCalculator()

        image_width = 640
        image_height = 480

        # The tolerance of the point to be erased in mm
        # if back_project_depth < current_view_depth - tolerance, the point is erased
        self.erase_point_depth_tolerance = 100

        # round points in x, y, z to precision of "round_points_to" value
        self.round_points_to = round_points_to

        self.z_axis_lower_bound = z_axis_lower_bound
        self.z_axis_upper_bound = z_axis_upper_bound
    
    def update_depth(self, depth_image):
        self.prev_depth = self.depth
        self.depth = depth_image
        # let all depth greater than 2 be 2
        self.depth[self.depth > self.max_depth_threshold] = self.max_depth_threshold
        self.rscalc.update_depth(self.depth)
    def update_pil_image(self, pil_image):
        self.prev_pil_image = self.pil_image
        self.pil_image = pil_image
    def update_trans(self, trans):
        self.prev_trans = self.trans
        self.trans = trans
    def update_info(self, depth_image, pil_image, trans):
        self.update_depth(depth_image)
        self.update_pil_image(pil_image)
        self.update_trans(trans)

    def get_feat_pixel_label(self, **kwargs):
        feat_list, pixel_list, label_list = self.image_semantic_extractor.get_feat_pixel_labels(self.pil_image, **kwargs)
        return feat_list, pixel_list, label_list

    def two_frame_difference_high(self):
        # calculate the difference between the current depth and the previous depth
        # calculate the difference between the current trans and the previous trans
        # if both difference are greater than a threshold, return True, meaning should erase
        if self.prev_depth is None or self.prev_trans is None or self.prev_pil_image is None:
            return False
        else:
            depth_diff = self.depth - self.prev_depth
            depth_diff = np.abs(depth_diff)
            depth_diff = np.mean(depth_diff)

            if depth_diff > self.erase_point_depth_threshold:
                return True
            else:
                return False
    
    def transform_to_points(self, pixel_list):
        point_list = []
        for i in range(len(pixel_list)):
            # Check whether the depth of the pixel is too large
            if self.depth[pixel_list[i][1], pixel_list[i][0]] >= self.max_depth_threshold:
                point_transformed = None
            else:
                point = self.rscalc.calculate_point(pixel_y=pixel_list[i][1],
                                                    pixel_x=pixel_list[i][0])
                point_transformed = self.rscalc.transform_point(point=point,
                                                                    trans=self.trans[:3],
                                                                    rot=self.trans[3:])
                if point_transformed[2] < self.z_axis_lower_bound or point_transformed[2] > self.z_axis_upper_bound:
                    point_transformed = None
            point_list.append(point_transformed)
        return point_list
    
    def inverse_transform_to_pixels(self, point_list):
        pixel_list_inverse = []
        for i in range(len(point_list)):
            point_inverse = self.rscalc.inverse_transform(point=point_list[i],
                                                        trans=self.trans[:3],
                                                        rot=self.trans[3:])
            pixel_y, pixel_x, depth = self.rscalc.backproj_point_to_pixel_depth(point=point_inverse)
            pixel_list_inverse.append((pixel_x, pixel_y))
        return pixel_list_inverse
    
    def should_erase_point(self, point_world_frame):
        # convert point in world frame to camera frame
        if point_world_frame is None:
            raise Exception('Point is None')
        point = self.rscalc.inverse_transform(point=point_world_frame,
                                            trans=self.trans[:3],
                                            rot=self.trans[3:])
        if point is None:
            raise Exception('Point is None 2')
        pixel_y, pixel_x, depth = self.rscalc.backproj_point_to_pixel_depth(point=point)

        if pixel_x == None or pixel_y == None or depth == None:
            return True
        if pixel_x < 0 or pixel_x >= 640 or pixel_y < 0 or pixel_y >= 480:
            return False
        # remember to convert depth from mm to m
        if self.depth[pixel_y, pixel_x] is None:
            return True
        if depth < self.depth[pixel_y, pixel_x] - self.erase_point_depth_tolerance:
            return True
        return False
    
    def erase_old_points(self):
        if self.two_frame_difference_high():
            return
        list_of_points = list(self.semantic_point_cloud.keys())
        points_to_delete = []
        for point in list_of_points:
            if self.should_erase_point(point):
                points_to_delete.append(point)
        for point in points_to_delete:
            del self.semantic_point_cloud[point]
        return
    
    def round_points(self,point_list):
        for i in range(len(point_list)):
            if point_list[i] is None:
                continue
            point_list[i] = (round(point_list[i][0]/self.round_points_to)*self.round_points_to,
                            round(point_list[i][1]/self.round_points_to)*self.round_points_to,
                            round(point_list[i][2]/self.round_points_to)*self.round_points_to)
        return point_list
    
    def update_semantic_point_cloud(self, points_transformed, feature_list, label_list):
        if len(points_transformed) != len(feature_list):
            raise Exception('Length of points_transformed and feature_list is not the same')
        if label_list is None:
            points_transformed = self.round_points(points_transformed)
            for i in range(len(points_transformed)):
                if points_transformed[i] is None:
                    continue
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], None)
        else:
            if len(points_transformed)!= len(label_list):
                raise Exception('Length of points_transformed and label_list is not the same')
            points_transformed = self.round_points(points_transformed)
            for i in range(len(points_transformed)):
                if points_transformed[i] is None:
                    continue
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], label_list[i])
    
    def similarity_search(self, similarity_threshold, text):
        '''
        Search for points in the semantic map that are similar to the given point.
        Args:
            similarity_threshold (float): The threshold for angular similarity between two features, in radians
        '''
        text_feat = lseg_model.encode_text(text)
        threshold = np.cos(similarity_threshold)
        list_of_SemPoints = list(self.semantic_point_cloud.values())
        list_of_points = list(self.semantic_point_cloud.keys())
        found_labels = []
        found_points = []
        found_similarities = []
        for i in range(len(list_of_SemPoints)):
            similarity = list_of_SemPoints[i].feat @ text_feat.t()
            similarity = similarity.cpu().detach().numpy().astype(np.float32).item()
            if similarity > threshold:
                found_labels.append(list_of_SemPoints[i].label)
                found_points.append(list_of_points[i])
                found_similarities.append(similarity)
        return found_similarities, found_points, found_labels

class SemanticMapNode(Node):
    def __init__(self, map_core=None):
        super().__init__('semantic_map_node')
        self.map_core = map_core or SemanticMapCore(image_semantic_extractor=YOLO_LSeg_ImageProcessor())

        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
        self.topic_pil_image = None
        self.topic_depth = None
        self.topic_trans = None

        self.publisher = self.create_publisher(PointCloud, 'point_cloud', 10)

        self.update_interval = 0.01
        self.map_lock = threading.Lock()
        self.create_timer(self.update_interval, self.map_update)

    def publish_input_point_cloud(self, point_list):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "map"
        for point in point_list:
            msg_point = Point32()
            msg_point.x = point[0]
            msg_point.y = point[1]
            msg_point.z = point[2]
            point_cloud_msg.points.append(msg_point)
        self.publisher.publish(point_cloud_msg)
    
    def callback(self, msg):
        if msg.rotation_x == 0.0 and msg.rotation_y == 0.0 and msg.rotation_z == 0.0 and msg.rotation_w == 0.0:
            return
        image_height = msg.height
        image_width = msg.width
        color_image = np.frombuffer(msg.data_color, dtype=np.uint8).reshape(image_height, image_width, 3)
        depth_image = np.frombuffer(msg.data_depth, dtype=np.uint16).reshape(image_height, image_width)
        with self.map_lock:
            self.topic_pil_image = Image.fromarray(color_image)
            self.topic_depth = depth_image
            self.topic_trans = np.array([msg.translation_x, msg.translation_y, msg.translation_z, msg.rotation_x, msg.rotation_y, msg.rotation_z, msg.rotation_w])
    
    def map_update(self):
        with self.map_lock:
            if self.topic_pil_image is not None and self.topic_depth is not None and self.topic_trans is not None:
                self.map_core.update_info(self.topic_depth, self.topic_pil_image, self.topic_trans)
                self.map_core.erase_old_points()

                feat_list, pixel_list, label_list = self.map_core.get_feat_pixel_label()
                points_transformed = self.map_core.transform_to_points(pixel_list)
                self.map_core.update_semantic_point_cloud(points_transformed, feat_list, label_list)

                self.publish_input_point_cloud(self.map_core.semantic_point_cloud.keys())

                self.topic_pil_image = None
                self.topic_depth = None
                self.topic_trans = None

                # print(self.map_core.similarity_search(3.14/12, 'keyboard'))

class SemanticMapService(Node):
    def __init__(self, map_core=None):
        super().__init__('semantic_map_node')
        self.query_service = self.create_service(
            SemanticQuery,
            'semantic_query',
            self.handle_semantic_query,
            callback_group=ReentrantCallbackGroup()
        )

        self.map_core = map_core or SemanticMapCore(image_semantic_extractor=YOLO_LSeg_ImageProcessor())

        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
        self.topic_pil_image = None
        self.topic_depth = None
        self.topic_trans = None

        self.publisher = self.create_publisher(PointCloud, 'point_cloud', 10)

        self.update_interval = 0.01
        self.map_rwlock = threading.RLock()
        self.data_lock = threading.Lock()
        self.create_timer(self.update_interval, self.map_update)

    def publish_input_point_cloud(self, point_list):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "map"
        for point in point_list:
            msg_point = Point32()
            msg_point.x = point[0]
            msg_point.y = point[1]
            msg_point.z = point[2]
            point_cloud_msg.points.append(msg_point)
        self.publisher.publish(point_cloud_msg)
    
    def callback(self, msg):
        if msg.rotation_x == 0.0 and msg.rotation_y == 0.0 and msg.rotation_z == 0.0 and msg.rotation_w == 0.0:
            return
        image_height = msg.height
        image_width = msg.width
        color_image = np.frombuffer(msg.data_color, dtype=np.uint8).reshape(image_height, image_width, 3)
        depth_image = np.frombuffer(msg.data_depth, dtype=np.uint16).reshape(image_height, image_width)
        with self.data_lock:
            self.topic_pil_image = Image.fromarray(color_image)
            self.topic_depth = depth_image
            self.topic_trans = np.array([msg.translation_x, msg.translation_y, msg.translation_z, msg.rotation_x, msg.rotation_y, msg.rotation_z, msg.rotation_w])
    
    def map_update(self):
        with self.map_rwlock:
            if self.topic_pil_image is not None and self.topic_depth is not None and self.topic_trans is not None:
                self.map_core.update_info(self.topic_depth, self.topic_pil_image, self.topic_trans)
                self.map_core.erase_old_points()

                feat_list, pixel_list, label_list = self.map_core.get_feat_pixel_label()
                points_transformed = self.map_core.transform_to_points(pixel_list)
                rclpy.logging.get_logger('sem_map_node').info(f'points: {points_transformed}')
                self.map_core.update_semantic_point_cloud(points_transformed, feat_list, label_list)

                self.publish_input_point_cloud(self.map_core.semantic_point_cloud.keys())

                self.topic_pil_image = None
                self.topic_depth = None
                self.topic_trans = None

    def handle_semantic_query(self, request, response):
        with self.map_rwlock:
            rclpy.logging.get_logger('sem_map_node').info(f'handle_semantic_query: {request.object_name}')
            found_similarities, found_points, found_labels = self.map_core.similarity_search(request.similarity_threshold_rad, request.object_name)
            response.similarities = found_similarities
            point_list = []
            for point_tuple in found_points:
                new_point = Point()
                new_point.x = point_tuple[0]
                new_point.y = point_tuple[1]
                new_point.z = point_tuple[2]
                point_list.append(new_point)
            response.points = point_list
            response.labels = found_labels
            rclpy.logging.get_logger('sem_map_node').info(f'found points: {found_points}')
        return response