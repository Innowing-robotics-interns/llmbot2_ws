    
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import cv2
import rclpy.logging
import rclpy.logging
import torch
import time

# Before running this, add lang-seg folder to your python path
from .LoadLSeg import *
import torchvision.transforms as transforms
from test_lseg_zs import *

from sklearn.cluster import KMeans

from .map_utils import *
import sys
    
def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"

    rscalc = RealSensePointCalculator()
    socket_receiver = SocketReceiver()

    rclpy.init(args=args)
    pcfm_main = PointCloudFeatureMap()
    executor = MultiThreadedExecutor()
    executor.add_node(pcfm_main)
    
    transform = transforms.Compose([
        transforms.Resize(480),           # Resize the shorter side to 480 while maintaining aspect ratio
        transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
        transforms.ToTensor()            # Convert to tensor
    ])

    # print("\033[H\033[J", end="")
    try:
        socket_receiver.socket_connect()

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            # pcfm_main.listen_tf()
            image_tensor = transform(image_subscriber.pil_image)
            with torch.no_grad():
                feat = model(image_tensor.unsqueeze(0).cuda())
            cluster_image = PCA_and_Cluster(feat)
            key_pixels = obtain_key_pixels(feat, cluster_image)
            key_points = rscalc.obtain_key_points(key_pixels)
            update_done = pcfm_main.update_pcfm(key_points, pcfm_threshold=1000, drop_range=0.5, drop_ratio=0.2)
            print(f"pcfm point count: {len(pcfm_main.pcfm)}")
            map_saved = pcfm_main.save_pcfm(map_path)
            if map_saved:
                print("Map saved at", pcfm_main.curr_time)
            # convert to display type
            cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
            cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
            cv2.imshow('Cluster Image', cluster_image_uint8)
            cv2.waitKey(1)
            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught")
        # sys.stdout.flush()
    finally:
        print("Shutting down")
        # sys.stdout.flush()
        socket_receiver.server_socket.close()
        cv2.destroyAllWindows()
        executor.shutdown()
        pcfm_main.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    print("enter main")
    main()