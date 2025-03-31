# from .image_processors import *
from .utils import *
from interfaces.msg import ColorDepthTrans
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node

from PIL import Image


# for testing
# import matplotlib.pyplot as plt

class SemanticPoint:
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

class SemanticMapMaintainer(Node):
    def __init__(self, image_semantic_extractor, 
    max_depth_threshold=2e3, 
    erase_point_depth_threshold=40000, 
    erase_point_depth_tolerance=100,
    round_points_to=0.05):
        super().__init__('semantic_map_maintainer')

        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
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
    
    def get_feat_pixel_label(self, **kwargs):
        feat_list, pixel_list, label_list = self.image_semantic_extractor.get_feat_pixel_labels(self.pil_image, **kwargs)
        return feat_list, pixel_list, label_list

    def callback(self, msg):
        # convert msg to numpy array
        if msg.rotation_x == 0.0 and msg.rotation_y == 0.0 and msg.rotation_z == 0.0 and msg.rotation_w == 0.0:
            return
        
        self.prev_depth = self.depth
        self.prev_trans = self.trans
        self.prev_pil_image = self.pil_image

        image_height = msg.height
        image_width = msg.width
        color_image = np.frombuffer(msg.data_color, dtype=np.uint8).reshape(image_height, image_width, 3)
        depth_image = np.frombuffer(msg.data_depth, dtype=np.uint16).reshape(image_height, image_width)
        self.pil_image = Image.fromarray(color_image)
        self.depth = depth_image
        self.trans = np.array([msg.translation_x, msg.translation_y, msg.translation_z, msg.rotation_x, msg.rotation_y, msg.rotation_z, msg.rotation_w])

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
    
    def update_depth(self):
        # let all depth greater than 2 be 2
        self.depth[self.depth > self.max_depth_threshold] = self.max_depth_threshold
        self.rscalc.update_depth(self.depth)

    def transform_to_points(self, pixel_list):
        point_list = []
        for i in range(len(pixel_list)):
            point = self.rscalc.calculate_point(pixel_y=pixel_list[i][1],
                                                pixel_x=pixel_list[i][0])
            point_transformed = self.rscalc.transform_to_point(point=point,
                                                            trans=self.trans[:3],
                                                            rot=self.trans[3:])
            point_list.append(point_transformed)
        return point_list
    
    def inverse_transform_to_pixels(self, point_list):
        pixel_list_inverse = []
        for i in range(len(point_list)):
            point_inverse = self.rscalc.inverse_transform_to_pixel(point=point_list[i],
                                                                    trans=self.trans[:3],
                                                                    rot=self.trans[3:])
            pixel_y, pixel_x, depth = self.rscalc.backproj_point_to_pixel_depth(point=point_inverse)
            pixel_list_inverse.append((pixel_x, pixel_y))
        return pixel_list_inverse
    
    def should_erase_point(self, point_world_frame):
        # convert point in world frame to camera frame
        if point_world_frame is None:
            raise Exception('Point is None')
        point = self.rscalc.inverse_transform_to_pixel(point=point_world_frame,
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
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], None)
        else:
            if len(points_transformed)!= len(label_list):
                raise Exception('Length of points_transformed and label_list is not the same')
            points_transformed = self.round_points(points_transformed)
            for i in range(len(points_transformed)):
                self.semantic_point_cloud[points_transformed[i]] = SemanticPoint(feature_list[i], label_list[i])
        

