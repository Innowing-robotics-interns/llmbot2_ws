# from .image_processors import *
from .utils import *
from interfaces.msg import ColorDepthTrans
import rclpy
import numpy as np

from rclpy.node import Node

from PIL import Image

# for testing
# import matplotlib.pyplot as plt

class SemanticPoint:
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

class SemanticMapMaintainer(Node):
    def __init__(self, image_semantic_extractor):
        super().__init__('semantic_map_maintainer')

        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
        self.pil_image = None
        self.depth = None
        self.trans = None

        self.image_semantic_extractor = image_semantic_extractor
        self.semantic_point_cloud = {} # dict in the format of (x,y,z):SemanticPoint
        self.rscalc = RealSensePointCalculator()
    
    def get_feat_pixel(self, **kwargs):
        feat_list, pixel_list, label_list = self.image_semantic_extractor.get_feat_pixel_labels(self.pil_image, **kwargs)
        return feat_list, pixel_list, label_list

    def callback(self, msg):
        # convert msg to numpy array
        if msg.rotation_x == 0.0 and msg.rotation_y == 0.0 and msg.rotation_z == 0.0 and msg.rotation_w == 0.0:
            return

        image_height = msg.height
        image_width = msg.width
        color_image = np.frombuffer(msg.data_color, dtype=np.uint8).reshape(image_height, image_width, 3)
        depth_image = np.frombuffer(msg.data_depth, dtype=np.uint16).reshape(image_height, image_width)
        self.pil_image = Image.fromarray(color_image)
        self.depth = depth_image
        # display the image size
        # self.get_logger().info('Image size: %d x %d' % (image_width, image_height))
        # self.get_logger().info('Depth image size: %d x %d' % (depth_image.shape[1], depth_image.shape[0]))
        # # display the image
        # plt.imshow(self.pil_image)
        # plt.show()

        self.trans = [msg.translation_x, msg.translation_y, msg.translation_z, msg.rotation_x, msg.rotation_y, msg.rotation_z, msg.rotation_w]
    
    def update_depth(self):
        self.rscalc.update_depth(self.depth)

    def transform_points(self, pixel_list):
        point_list = []
        for i in range(len(pixel_list)):
            point = self.rscalc.calculate_point_with_offset(pixel_y=pixel_list[i][1],
                                                            pixel_x=pixel_list[i][0],
                                                            offset_x=0,
                                                            offset_y=0)
            # point = self.rscalc.calculate_point_with_offset(pixel_y=pixel_list[i][1],
            #                                                 pixel_x=pixel_list[i][0],
            #                                                 offset_x=self.image_semantic_extractor.offset_x,
            #                                                 offset_y=self.image_semantic_extractor.offset_y)
            point_transformed = self.rscalc.transform_point(point=point,
                                                            trans=self.trans[:3],
                                                            rot=self.trans[3:])
            point_list.append(point_transformed)
        return point_list
    
    def back_transform_point_clouds(self):
        point_cloud = []
        for key in self.semantic_point_cloud.keys():
            point = self.rscalc.inverse_transform_point(point=key,
                                                        trans=self.trans[:3],
                                                        rot=self.trans[3:])
            point_cloud.append(point)
        return point_cloud, semantic_point_cloud.keys()
    
    def erase_overseen_points(self, point_cloud, semantic_point_cloud_keys):
        for i in range(len(point_cloud)):
            point = point_cloud[i]
            pixel_y, pixel_x, depth = self.rscalc.backproj_point_to_pixel_depth(point=point)
            if self.depth[pixel_y, pixel_x] is None:
                continue
            if depth < self.depth[pixel_y, pixel_x]:
                del self.semantic_point_cloud[semantic_point_cloud_keys[i]]
    

