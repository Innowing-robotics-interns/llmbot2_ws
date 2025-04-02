from .params import *

# class LSegFeatImageProcessor
from .load_lseg import model as lseg_model
import torchvision.transforms as transforms
import torch
import numpy as np
from cuml.decomposition import PCA as cuPCA
from cuml.cluster import KMeans as cuKMeans
import cupy
from skimage.measure import label

# class YOLO_LSeg_ImageProcessor
from ultralytics import YOLO

'''
Each ImageProcessor should have the following methods:
- get_feat_pixel_labels(pil_image)
return: 
  - feat_list: list of features, in format: [feat1, feat2,...], each feat is a tensor
  - pixel_list: list of pixels, in format: [(x,y), (x,y), ...]
  - label_list: list of labels, in format: [label1, label2,...], can be None

and the following member variables:
self.name
self.label_used
self.image_offset_x
self.image_offset_y
'''

def create_processor(image_processr_name, **kwargs):
    if image_processr_name == 'lseg_feat':
        return LSegFeatImageProcessor(**kwargs)
    elif image_processr_name == 'yolo_lseg':
        return YOLO_LSeg_ImageProcessor(**kwargs)
    else:
        raise ValueError('Unknown image processor name: {}'.format(image_processr_name))

def encode_text(text):
    return lseg_model.encode_text(text)

def similarity(text_feat, features):
    similarities = []
    for feat in features:
        sim = feat.half() @ text_feat.t()
        similarities.append(sim)
    similarities = torch.cat(similarities)
    similarities = similarities.cpu().detach().numpy()
    return similarities

def max_sim_feature_index(list_of_features, text_feat):
    similarities = similarity(text_feat, list_of_features)
    return np.argmax(similarities)

class LSegFeatImageProcessor:
    def __init__(self, model=lseg_model):
        self.name = 'lseg_feat'

        self.label_used = False

        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    480
                ),  # Resize the shorter side to 480 while maintaining aspect ratio
                transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
                transforms.ToTensor(),  # Convert to tensor
            ]
        )
        self.current_features = None
        self.current_image = None
        # self.image_original_size = [640, 480]
        self.image_offset_x = 80
        self.image_offset_y = 0
    
    def update_current_image(self, pil_image):
        self.current_image = self.transform(pil_image)

    def update_feature(self):
        with torch.no_grad():
            feat = self.model(self.current_image.unsqueeze(0).cuda())
        self.current_features = feat.half()

    def relabel_connected_components(self, class_image, n_classes=10):
        '''relabelling connected components in the clustered image.

        Args:
            class_image (numpy.ndarray): The clustered image.
            n_classes (int): The number of classes in the clustered image.

        Returns:
            numpy.ndarray: The relabeled image.
        '''
        relabeled_image = np.zeros_like(class_image)

        for class_label in range(n_classes):
            mask = class_image == class_label
            labeled_mask, num_features = label(mask, connectivity=2, return_num=True)
            relabeled_mask = labeled_mask + (relabeled_image.max() + 1) * (labeled_mask > 0)
            relabeled_image += relabeled_mask
        return relabeled_image

    def PCA_cuda(self, feats, n_components=20):
        '''
        Applies PCA to reduce the dimensionality of the feature map on CUDA.

        Args:
            feats (torch.Tensor): The input feature map.
            n_components (int): The number of principal components to keep.

        Returns:
            numpy.ndarray: The transformed feature map with reduced dimensionality.
        '''
        pca = cuPCA(n_components=n_components)
        feat_map = feats.flatten(start_dim=0, end_dim=1)
        features = pca.fit_transform(feat_map)
        return features

    def Cluster_cuda(self, features, n_clusters=10):
        '''
        Applies K-Means clustering to the feature map on CUDA.

        Args:
            features (numpy.ndarray): The input feature map with reduced dimensionality.
            n_clusters (int): The number of clusters to form.
            H (int): The height of the original image.
            W (int): The width of the original image.

        Returns:
            numpy.ndarray: The clustered image.
        '''
        kmeans = cuKMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(features)
        clustered_image = labels.reshape(480, 480)
        clustered_image = cupy.asnumpy(clustered_image)
        clustered_image = self.relabel_connected_components(clustered_image, n_classes=n_clusters)
        return clustered_image

    def Cluster_cuda_wo_relabel(self, features, n_clusters=10):
        '''
        Applies K-Means clustering to the feature map on CUDA.

        Args:
            features (numpy.ndarray): The input feature map with reduced dimensionality.
            n_clusters (int): The number of clusters to form.
            H (int): The height of the original image.
            W (int): The width of the original image.

        Returns:
            numpy.ndarray: The clustered image.
        '''
        kmeans = cuKMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(features)
        clustered_image = labels.reshape(480, 480)
        clustered_image = cupy.asnumpy(clustered_image)
        # clustered_image = self.relabel_connected_components(clustered_image, n_classes=n_clusters)
        return clustered_image

    def obtain_key_pixels(self, feat, clustered_image, pixels_percent=0.3, rule_out_threshold=500):
        '''Obtain key pixels from the clustered image based on the feature map.

        Args:
            feat (torch.Tensor): The input feature map.
            clustered_image (numpy.ndarray): The clustered image.
            n_pixels (int): The number of key pixels to obtain.
            rule_out_threshold (int): The threshold to rule out classes with fewer pixels.

        Returns:
            feat_list: The list of key features in the format of [feat_1, feat_2, ...].
            pixel_list: The list of key pixels in the format of [[pixel_x, pixel_y]...]].
        '''
        pixels_percent = pixels_percent % 1
        num_class = clustered_image.max() + 1
        # feat_pixel_pair = []
        feat_list = []
        pixel_list = []
        for i in range(num_class):
            class_feat = feat[clustered_image == i]
            if len(class_feat) == 0:
                continue
            feat_mean = class_feat.mean(dim=0)
            # class_pixel format: [[class_pixel_x, ...], [class_pixel_y, ...]]
            class_pixel = np.where(clustered_image == i)
            if len(class_pixel[0]) < rule_out_threshold:
                continue
            feat_list.append(feat_mean)
            pixel_list.append([int(class_pixel[1].mean()+self.image_offset_x), int(class_pixel[0].mean()+self.image_offset_y)])
        return feat_list, pixel_list
    
    def get_clustered_map(self, image, num_clusters=36):
        self.update_current_image(image)
        self.update_feature()
        features = self.PCA_cuda(self.current_features)
        clustered_image = self.Cluster_cuda(features, n_clusters=num_clusters)
        return clustered_image

    def get_clustered_map_wo_relabel(self, image, num_clusters=36):
        self.update_current_image(image)
        self.update_feature()
        features = self.PCA_cuda(self.current_features)
        clustered_image = self.Cluster_cuda_wo_relabel(features, n_clusters=num_clusters)
        return clustered_image
    
    def get_feat_pixel_labels(self, image, n_pca_components=20, n_clusters=36, pixels_percent=0.3, rule_out_threshold=500):
        self.update_current_image(image)
        self.update_feature()
        features = self.PCA_cuda(self.current_features, n_components=n_pca_components)
        clustered_image = self.Cluster_cuda(features, n_clusters=36)
        feat_list, pixel_list = self.obtain_key_pixels(self.current_features, clustered_image, pixels_percent=pixels_percent, rule_out_threshold=rule_out_threshold)
        return feat_list, pixel_list, None

class YOLO_LSeg_ImageProcessor:
    def __init__(self, model_path="/home/fyp/llmbot2_ws/src/sem_map/model/yolo11x.pt"):
        self.name = 'yolo_lseg'

        self.label_used = True
        self.model = YOLO(model_path)

        self.image_offset_x = 0
        self.image_offset_y = 0
    
    def get_label(self, image):
        results = self.model(image)
        xywh = results[0].boxes.xywh
        names = [results[0].names[cls.item()] for cls in results[0].boxes.cls.int()]
        confs = results[0].boxes.conf
        return xywh, names, confs
    
    def get_feat_pixel_labels(self, image):
        xywh, names, confs = self.get_label(image)
        # feat_pixel_pair = []
        feat_list = []
        pixel_list = []
        xywh = xywh.cpu().detach().numpy().tolist()
        for i in range(len(xywh)):
            x, y, w, h = xywh[i]
            name = names[i]
            # feat_pixel_pair.append((encode_text(name)[0], (y, x)))
            feat_list.append(encode_text(name)[0])
            pixel_list.append((int(x), int(y)))
        return feat_list, pixel_list, names
    


