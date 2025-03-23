from .image_processors import *

class SemanticPoint:
    def __init__(self, yolo_object_name, yolo_object_conf):
        self.yolo_object_name = yolo_object_name
        self.yolo_object_conf = yolo_object_conf
        self.clip_feature = None
        self.tap_caption = None

class SemanticMap:
    def __init__(self, image_semantic_extractor):
        self.image_semantic_extractor = image_semantic_extractor
        self.semantic_point_cloud = {} # dict in the format of (x,y,z):SemanticPoint
