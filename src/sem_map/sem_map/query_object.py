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
        similarities.append(feat.half() @ text_feat.t())
    similarities = torch.cat(similarities)
    similarities = similarities.cpu().detach().numpy()
    return similarities

def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"
    with open(map_path, 'rb') as f:
        pcfm = pickle.load(f)
    print("pcfm loaded:")
    print("length:", len(pcfm))

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    pc_manager = PointCloudManager(topic_name='/pointcloud')
    search_manager = PointCloudManager(topic_name='/pc_search')
    max_manager = PointCloudManager(topic_name='/max_search')
    executor.add_node(pc_manager)
    executor.add_node(search_manager)

    threshold = 0.89
    print("\033[H\033[J", end="")
    print("threshold similarity:", threshold)

    try:
        while rclpy.ok():
            print("publishing background", len(list(pcfm.keys())), "points")
            pc_manager.publish_point_cloud(list(pcfm.keys()))
            text = input("Enter New Query: ")
            pc_manager.publish_point_cloud(list(pcfm.keys()))

            similarities = similarity(text, list(pcfm.values()))
            pc_manager.publish_point_cloud(list(pcfm.keys()))

            # find all index where similarity is greater than threshold
            idx = np.where(similarities > threshold)[0]
            points = np.array(list(pcfm.keys()))[idx]
            # print("points shape", points.shape)
            search_manager.publish_point_cloud(points)
            max_manager.publish_point_cloud([list(pcfm.keys())[np.argmax(similarities)]])
            executor.spin_once(timeout_sec=0.5)

            print("\033[H\033[J", end="")
            print("Query:", text)
            print()
            print("Found", len(points), "points")
            print("Max Similarity Point:", [list(pcfm.keys())[np.argmax(similarities)]])
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
