import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from interfaces.srv import SemanticQuery
from geometry_msgs.msg import Point

from .utils import *
from .image_processors import *
from .params import *
from interfaces.msg import ColorDepthTrans
from interfaces.msg import StringArray

import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import threading
import uuid
import pickle
import time

def load_semantic_map_core(map_config, image_processor_selection):
    map_read_name = map_config['map_read_name']
    map_save_name = map_config['map_save_name']
    map_save_interval = map_config['map_save_interval']
    max_depth_threshold = map_config['map_depth_threshold']
    erase_point_depth_threshold = map_config['erase_point_depth_threshold']
    erase_point_depth_tolerance = map_config['erase_point_depth_tolerance']
    round_points_to = map_config['round_points_to']
    z_axis_lower_bound = map_config['z_axis_lower_bound']
    z_axis_upper_bound = map_config['z_axis_upper_bound']
    exception_point_xy_radius = map_config['exception_point_xy_radius']
    image_processor_conf_threshold = map_config['image_processor_conf_threshold']
    depth_moving_average_window = map_config['depth_moving_average_window']
    erase_points = map_config['erase_points']
    transform_difference_moving_window = map_config['transform_difference_moving_window']
    translation_difference_threshold = map_config['translation_difference_threshold']
    rotation_difference_threshold = map_config['rotation_difference_threshold']
    # 打印加载语义地图核心的相关配置信息
    print(f"Loading semantic map core with the following configuration:\n"
          f"Map read name: {map_read_name}\n"
          f"Map save name: {map_save_name}\n"
          f"Map save interval: {map_save_interval}\n"
          f"Max depth threshold: {max_depth_threshold}\n"
          f"Erase point depth threshold: {erase_point_depth_threshold}\n"
          f"Erase point depth tolerance: {erase_point_depth_tolerance}\n"
          f"Round points to: {round_points_to}\n"
          f"Z-axis lower bound: {z_axis_lower_bound}\n"
          f"Z-axis upper bound: {z_axis_upper_bound}\n"
          f"exception_point_xy_radius: {exception_point_xy_radius}\n"
          f"Image processor confidence threshold: {image_processor_conf_threshold}\n"
          f"Depth moving average window: {depth_moving_average_window}\n"
          f"Erase points: {erase_points}\n"
          f"Transform difference moving window: {transform_difference_moving_window}\n"
          f"Translation difference threshold: {translation_difference_threshold}"
          f"Rotation difference threshold: {rotation_difference_threshold}")
    return SemanticMapCore(
        image_semantic_extractor=create_processor(image_processor_selection),
        map_read_name=map_read_name,
        map_save_name=map_save_name,
        map_save_interval=map_save_interval,
        max_depth_threshold=max_depth_threshold,
        erase_point_depth_threshold=erase_point_depth_threshold,
        erase_point_depth_tolerance=erase_point_depth_tolerance,
        round_points_to=round_points_to,
        z_axis_lower_bound=z_axis_lower_bound,
        z_axis_upper_bound=z_axis_upper_bound,
        exception_point_xy_radius=exception_point_xy_radius,
        image_processor_conf_threshold=image_processor_conf_threshold,
        depth_moving_average_window=depth_moving_average_window,
        erase_points=erase_points,
        transform_difference_moving_window=transform_difference_moving_window,
        translation_difference_threshold=translation_difference_threshold,
        rotation_difference_threshold=rotation_difference_threshold
    )

class SemanticPoint:
    def __init__(self, feat, label, conf=None):
        '''
        Args:
            feat (1D torch tensor): The feature of the point.
            label (str): The label of the point.
        '''
        self.feat = feat
        self.label = label
        self.conf = conf
        
class SemanticMapCore():
    def __init__(self, 
    image_semantic_extractor, 
    map_read_name=None,
    map_save_name=None,
    map_save_interval=30,
    max_depth_threshold=2e3, 
    erase_point_depth_threshold=40000, 
    erase_point_depth_tolerance=100,
    round_points_to=0.05,
    z_axis_lower_bound=0.5,
    z_axis_upper_bound=2.0,
    exception_point_xy_radius=12.0,
    image_processor_conf_threshold=0.5,
    depth_moving_average_window=5,
    erase_points=True,
    transform_difference_moving_window=6,
    translation_difference_threshold=0.03,
    rotation_difference_threshold=0.05):

        if transform_difference_moving_window < 2:
            self.transform_difference_used = False
        else:
            self.transform_difference_used = True
        
        self.transform_moving_window = transform_difference_moving_window
        self.transform_moving = np.zeros((transform_difference_moving_window, 7))
        self.depth_window = np.zeros((transform_difference_moving_window, 480, 640))
        self.color_window = np.zeros((transform_difference_moving_window, 480, 640, 3))

        self.transform_moving[-1] = np.array([0, 0, 0, 0, 0, 0, 1]) # init transform to identity
        self.transform_moving_index = 0
        self.transform_moving_window_count = 0

        self.transform_difference_window = np.zeros((transform_difference_moving_window-1, 2))
        self.difference_update_index = 0

        self.half_transform_difference_window = int((transform_difference_moving_window-1)/2)

        self.angle_difference_sum_threshold = rotation_difference_threshold * (self.transform_moving_window - 1)
        self.translation_distance_sum_threshold = translation_difference_threshold * (self.transform_moving_window - 1)
        self.translation_difference_threshold = translation_difference_threshold * 2
        self.rotation_difference_threshold = rotation_difference_threshold * 2
        
        self.erase_points = erase_points

        self.depth_moving_average_window = depth_moving_average_window
        self.depth_moving_average = np.zeros((depth_moving_average_window, 480, 640))
        self.depth_moving_average_index = 0
        self.depth_moving_average_sum = 0
        self.depth_moving_average_count = 0

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

        if image_semantic_extractor is None:
            raise ValueError('image_semantic_extractor is None')
        self.image_semantic_extractor = image_semantic_extractor
        self.image_semantic_extractor.conf_threshold = image_processor_conf_threshold

        self.label_used = image_semantic_extractor.label_used

        self.semantic_point_cloud = {} # dict in the format of (x,y,z):SemanticPoint
        if (map_read_name is not None) and (map_read_name != ""):
            map_read_name = sem_map_path + map_read_name + '.pkl'
            with open(map_read_name, 'rb') as f:
                self.semantic_point_cloud = pickle.load(f)
        self.map_save_name = self.generate_new_map_name(map_save_name)
        self.map_save_interval = map_save_interval # in rounds of updates

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

        self.exception_point_xy_radius = exception_point_xy_radius
    
    def two_transform_difference(self, trans1, trans2):
        trans1_linear = trans1[:3]
        trans2_linear = trans2[:3]
        norm_distance = np.linalg.norm(trans1_linear - trans2_linear)
        trans1_quat = trans1[3:]
        trans2_quat = trans2[3:]
        trans1_quat = R.from_quat(trans1_quat)
        trans2_quat = R.from_quat(trans2_quat)
        trans1_quat = trans1_quat.as_euler('xyz', degrees=False)
        trans2_quat = trans2_quat.as_euler('xyz', degrees=False)
        angle_difference = np.linalg.norm(trans1_quat - trans2_quat)
        return norm_distance, angle_difference
    
    def update_transform_difference_window(self, trans, color_image, depth_image):
        # update the moving average of the transform difference
        # first 3 of trans is linear transformation, last 4 is quaternion
        self.transform_moving[self.transform_moving_index] = trans
        self.color_window[self.transform_moving_index] = color_image
        self.depth_window[self.transform_moving_index] = depth_image

        difference = self.two_transform_difference(self.transform_moving[self.transform_moving_index-1], self.transform_moving[self.transform_moving_index])
        self.transform_difference_window[self.difference_update_index] = difference
        print(self.transform_difference_window)

        self.difference_update_index += 1
        if self.difference_update_index >= self.transform_moving_window-1:
            self.difference_update_index = 0
        if self.transform_moving_window_count < self.transform_moving_window:
            self.transform_moving_window_count += 1

        self.transform_moving_index += 1
        if self.transform_moving_index >= self.transform_moving_window:
            self.transform_moving_index = 0
        return difference
    
    def update_frame_at_middle(self):
        # sum the difference in transform up, if they are smaller than threshold, update
        if self.transform_moving_window_count < self.transform_moving_window:
            return False
        for distance_difference, angle_difference in self.transform_difference_window:
            if distance_difference == 0 and angle_difference == 0:
                return False
            if distance_difference > self.translation_difference_threshold or angle_difference > self.rotation_difference_threshold:
                return False
        distance_difference_sum, angle_difference_sum = np.sum(self.transform_difference_window, axis=0)
        if (distance_difference_sum < self.translation_distance_sum_threshold) and (angle_difference_sum < self.angle_difference_sum_threshold):
            middle_frame_index = self.half_transform_difference_window + self.transform_moving_index
            middle_frame_index = middle_frame_index % self.transform_moving_window
            self.update_info(self.depth_window[middle_frame_index], self.color_window[middle_frame_index], self.transform_moving[middle_frame_index])
            return True
        else:
            return False
    
    def update_depth_moving_average(self, depth_image):
        # update the moving average depth image
        self.depth_moving_average[self.depth_moving_average_index] = depth_image
        self.depth_moving_average_index += 1
        if self.depth_moving_average_index >= self.depth_moving_average_window:
            self.depth_moving_average_index = 0
        if self.depth_moving_average_count < self.depth_moving_average_window:
            self.depth_moving_average_count += 1
        self.depth_moving_average_sum = np.sum(self.depth_moving_average[:self.depth_moving_average_count], axis=0)
        return self.depth_moving_average_sum / self.depth_moving_average_count
    
    def point_too_far(self, point):
        radius = np.sqrt(point[0]**2 + point[1]**2)
        if radius > self.exception_point_xy_radius:
            return True
        else:
            return False
    
    def generate_new_map_name(self, map_save_name):
        now = time.localtime()
        if (map_save_name is not None) and (map_save_name != ""):
            return sem_map_path + map_save_name + time.strftime("%Y_%m_%d_%H_%M_%S", now) + '.pkl'
        else:
            return sem_map_path + self.image_semantic_extractor.name + time.strftime("%Y_%m_%d_%H_%M_%S", now) + '.pkl'
    
    def save_map(self):
        with open(self.map_save_name, 'wb') as f:
            pickle.dump(self.semantic_point_cloud, f)
        print('Map saved to {}'.format(self.map_save_name))

    def update_depth(self, depth_image):
        self.prev_depth = self.depth
        # self.depth = depth_image
        self.depth = self.update_depth_moving_average(depth_image)
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
        if self.prev_trans is not None:
            transform_difference, angle_difference = self.two_transform_difference(self.trans, self.prev_trans)
            print(f"Transform difference: {transform_difference}, Angle difference: {angle_difference}")
            print(f"Transform difference: {transform_difference}, Angle difference: {angle_difference}")
            print(f"Transform difference: {transform_difference}, Angle difference: {angle_difference}")
            print(f"Transform difference: {transform_difference}, Angle difference: {angle_difference}")
            print(f"Transform difference: {transform_difference}, Angle difference: {angle_difference}")

    def get_feat_pixel_label_conf(self, **kwargs):
        feat_list, pixel_list, label_list, conf_list = self.image_semantic_extractor.get_feat_pixel_label_confs(self.pil_image, **kwargs)
        return feat_list, pixel_list, label_list, conf_list

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
        if self.erase_points == False:
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
    
    def update_semantic_point_cloud(self, points_transformed, feature_list, label_list, conf_list):
        if len(points_transformed) != len(feature_list):
            raise Exception('Length of points_transformed and feature_list is not the same')
        if label_list is None:
            points_transformed = self.round_points(points_transformed)
            for i in range(len(points_transformed)):
                if (points_transformed[i] is None):
                    continue
                if self.point_too_far(points_transformed[i]):
                    print(f"Point {points_transformed[i]} is out of bound {self.exception_point_xy_radius} m from origin")
                    raise Exception('Point is too far')
                    continue
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], None, None)
        else:
            if len(points_transformed)!= len(label_list):
                raise Exception('Length of points_transformed and label_list is not the same')
            points_transformed = self.round_points(points_transformed)
            for i in range(len(points_transformed)):
                if (points_transformed[i] is None):
                    continue
                if self.point_too_far(points_transformed[i]):
                    print(f"Point {points_transformed[i]} is out of bound {self.exception_point_xy_radius} m from origin")
                    raise Exception('Point is too far')
                    continue
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], label_list[i], conf_list[i])
    
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
        found_confs = []
        found_points = []
        found_similarities = []
        if self.label_used:
            for i in range(len(list_of_SemPoints)):
                similarity = list_of_SemPoints[i].feat @ text_feat.t()
                similarity = similarity.cpu().detach().numpy().astype(np.float32).item()
                if similarity > threshold:
                    found_labels.append(list_of_SemPoints[i].label)
                    found_confs.append(list_of_SemPoints[i].conf.cpu().detach().numpy().astype(np.float32).item())
                    found_points.append(list_of_points[i])
                    found_similarities.append(similarity)
        else:
            for i in range(len(list_of_SemPoints)):
                similarity = list_of_SemPoints[i].feat @ text_feat.t()
                similarity = similarity.cpu().detach().numpy().astype(np.float32).item()
                if similarity > threshold:
                    found_points.append(list_of_points[i])
                    found_similarities.append(similarity)
        return found_similarities, found_points, found_labels, found_confs

class SemanticMapNode(Node):
    def __init__(self, map_core=None):
        super().__init__(f'semantic_map_node_{uuid.uuid4().hex[:4]}')
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

                feat_list, pixel_list, label_list, conf_list = self.map_core.get_feat_pixel_label_conf()
                points_transformed = self.map_core.transform_to_points(pixel_list)
                self.map_core.update_semantic_point_cloud(points_transformed, feat_list, label_list, conf_list)

                self.publish_input_point_cloud(self.map_core.semantic_point_cloud.keys())

                self.topic_pil_image = None
                self.topic_depth = None
                self.topic_trans = None

                # print(self.map_core.similarity_search(3.14/12, 'keyboard'))

class SemanticMapService(Node):
    def __init__(self, image_processor_selection):
        super().__init__('sem_map_service_'+image_processor_selection)

        config = read_config("config_"+image_processor_selection)
        map_config = config['map_config']
        self.map_core = load_semantic_map_core(map_config, image_processor_selection)

        # Other variables
        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
        self.topic_pil_image = None
        self.topic_depth = None
        self.topic_trans = None

        self.publisher_mapped_points = self.create_publisher(PointCloud, 'sem_points_'+image_processor_selection, 10)
        self.publisher_found_points = self.create_publisher(PointCloud, 'sem_points_found_'+image_processor_selection, 10)

        self.label_used = self.map_core.image_semantic_extractor.label_used
        if self.map_core.image_semantic_extractor.label_used:
            self.publisher_labels = self.create_publisher(StringArray,'existing_labels_'+image_processor_selection, 10)

        self.query_service = self.create_service(
            SemanticQuery,
            'semantic_query_'+image_processor_selection,
            self.handle_semantic_query,
            callback_group=ReentrantCallbackGroup()
        )

        self.map_rwlock = threading.RLock()
        self.data_lock = threading.Lock()

        self.save_counter = 0
        self.update_interval = 0.01
        self.create_timer(self.update_interval, self.map_update)
    
    def publish_labels(self):
        if self.map_core.image_semantic_extractor.label_used:
            msg = StringArray()
            semantic_points = list(self.map_core.semantic_point_cloud.values())
            if len(semantic_points) != 0:
                if semantic_points[0].label is None:
                    raise Exception('Label is None')
                label_list = [semantic_point.label for semantic_point in semantic_points]
                msg.string_array = list(set(label_list))
                self.publisher_labels.publish(msg)

    def publish_input_point_cloud(self, point_list, publisher_selected):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "map"
        for point in point_list:
            msg_point = Point32()
            msg_point.x = point[0]
            msg_point.y = point[1]
            msg_point.z = point[2]
            point_cloud_msg.points.append(msg_point)
        publisher_selected.publish(point_cloud_msg)
    
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
                if self.map_core.transform_difference_used:
                    difference = self.map_core.update_transform_difference_window(self.topic_trans, self.topic_pil_image, self.topic_depth)
                    update = self.map_core.update_frame_at_middle()
                    if update == False:
                        self.topic_pil_image = None
                        self.topic_depth = None
                        self.topic_trans = None
                        return
                    else:
                        self.get_logger().info(f'Update map with transform difference: {difference}')
                        self.get_logger().info(f'Update map with transform difference: {difference}')
                        self.get_logger().info(f'Update map with transform difference: {difference}')
                        self.get_logger().info(f'Update map with transform difference: {difference}')
                        self.get_logger().info(f'Update map with transform difference: {difference}')
                else:
                    self.map_core.update_info(self.topic_depth, self.topic_pil_image, self.topic_trans)

                self.map_core.erase_old_points()

                feat_list, pixel_list, label_list, conf_list = self.map_core.get_feat_pixel_label_conf()
                points_transformed = self.map_core.transform_to_points(pixel_list)
                rclpy.logging.get_logger('sem_map_service').info(f'points: {points_transformed}')
                self.map_core.update_semantic_point_cloud(points_transformed, feat_list, label_list, conf_list)

                self.publish_input_point_cloud(self.map_core.semantic_point_cloud.keys(), self.publisher_mapped_points)

                self.topic_pil_image = None
                self.topic_depth = None
                self.topic_trans = None
                
                self.save_counter += 1
                if self.save_counter % self.map_core.map_save_interval == 0:
                    self.map_core.save_map()
                    rclpy.logging.get_logger('sem_map_service').info(f'save map')
                    self.save_counter = 0
                    if self.label_used:
                        self.publish_labels()

    def handle_semantic_query(self, request, response):
        rclpy.logging.get_logger('sem_map_service').info(f'handle_semantic_query: {request.object_name}')
        with self.map_rwlock:
            found_similarities, found_points, found_labels, found_confs = self.map_core.similarity_search(request.similarity_threshold_rad, request.object_name)
        print("found")
        print("found")
        print("found")
        print(found_labels)
        print(found_confs)
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
        response.confs = found_confs
        rclpy.logging.get_logger('sem_map_service').info(f'found points: {found_points}')

        self.publish_input_point_cloud(found_points, self.publisher_found_points)

        return response