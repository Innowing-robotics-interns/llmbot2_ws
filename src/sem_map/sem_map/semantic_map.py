from .image_processors import *
from .utils import RealSensePointCalculator

class SemanticPoint:
    def __init__(self, label):
        self.label = label

class SemanticMap:
    def __init__(self, image_semantic_extractor):
        self.image_semantic_extractor = image_semantic_extractor
        self.semantic_point_cloud = {} # dict in the format of (x,y,z):SemanticPoint
        self.rscalc = RealSensePointCalculator()
    
    def update_depth(self, depth_img):
        self.rscalc.update_depth(depth_img)
    
    def add_semantic_point(self, image, **kwargs):
        feat_pixel_pair = self.image_semantic_extractor.get_feat_pixel_pair(image, **kwargs)


    

