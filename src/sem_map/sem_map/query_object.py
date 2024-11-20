import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import torch
import time

# Before running this, add lang-seg folder to your python path
from .LoadLSeg import *

from .map_utils import PointCloudManager

import pickle

def similarity(text, features):
    with torch.no_grad():
        text_feat = model.encode_text(text)
    similarities = []
    for feat in features:
        similarities.append(feat @ text_feat.t())
    similarities = torch.cat(similarities)
    similarities = similarities.cpu().detach().numpy()
    return similarities

def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"
    with open(map_path, 'rb') as f:
        pcfm = pickle.load(f)

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    pc_manager = PointCloudManager(topic_name='/pointcloud')
    search_manager = PointCloudManager(topic_name='/pc_search')
    executor.add_node(pc_manager)
    executor.add_node(search_manager)

    threshold = 0.95

    try:
        while rclpy.ok():
            pc_manager.publish_point_cloud(pcfm.keys())
            text = input("Enter text: ")
            similarities = similarity(text, list(pcfm.values()))
            # find all index where similarity is greater than threshold
            idx = torch.where(similarity > threshold)[0]
            points = np.array(list(pcfm.keys()))[idx]
            search_manager.publish_point_cloud(points)
            executor.spin_once(timeout_sec=0.5)
            time.sleep(0.1)
    except KeyboardInterrupt:
        rclpy.logging.get_logger('query_object').info("KeyboardInterrupt")
    finally:
        rclpy.logging.get_logger('query_object').info("Shutting down...")
        pc_manager.destroy_node()
        search_manager.destroy_node()
        executor.shutdown()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
