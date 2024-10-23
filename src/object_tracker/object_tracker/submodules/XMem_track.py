import torch

if torch.cuda.is_available():
  print('Using GPU')
  device = 'cuda'

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
import numpy as np

torch.set_grad_enabled(False)

class XMemTrack:
    def __init__(self, frame_size=[720, 1280]):
        # default configuration
        self.config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10000,
        }
        self.device = 'cuda'
        self.network = XMem(self.config, '/home/fyp/XMem/saves/XMem.pth').eval().to(device)
        self.frame_size = frame_size

    def initialize(self, num_objects,mask,frame):
        torch.cuda.empty_cache()
        processor = InferenceCore(self.network, config=self.config)
        processor.set_all_labels(range(1, num_objects+1)) # consecutive labels

        frame_torch, _ = image_to_torch(frame, device=self.device)
        mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
        prediction = processor.step(frame_torch, mask_torch[1:])
        prediction = torch_prob_to_numpy_mask(prediction)
        print("initialized")
        return processor, prediction

    def track(self, processor, frame):
        # need the following line for the while loop
        # with torch.cuda.amp.autocast(enabled=True):
        frame_torch, _ = image_to_torch(frame, device=self.device)
        prediction = processor.step(frame_torch)
        prediction = torch_prob_to_numpy_mask(prediction)
        return processor, prediction
    
    def center_of_mask(self, prediction):
        binary_mask = prediction > 0
        if binary_mask.sum() == 0:
            print("No valid pixels found in the mask.")
            return None
        coords = np.column_stack(np.where(binary_mask))
        center = coords.mean(axis=0)
        center_pixel = tuple(center.astype(int))
        if center_pixel[0] < 0 or center_pixel[1] < 0 or center_pixel[0] > self.frame_size[0] or center_pixel[1] > self.frame_size[1]:
            print("Center pixel out of frame.")
            print(center_pixel)
            # write binary_mask to a file
            print(binary_mask.shape)
            cv2.imwrite("~/mask.jpg", binary_mask.astype(np.uint8)*255)
            return None
        return center_pixel