    
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import cv2
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
    rclpy.init(args=args)
    print("init")
    image_subscriber = ImageSubscriber()
    rscalc = RealSensePointCalculator()
    pcfm_main = PointCloudFeatureMap()
    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(rscalc)
    executor.add_node(pcfm_main)

    print("load test setting")
    # following line for testing, remember to remove
    pcfm_main.camera_to_world = TransformStamped()
    pcfm_main.camera_to_world.transform.translation.x = 0.0
    pcfm_main.camera_to_world.transform.translation.y = 0.0
    pcfm_main.camera_to_world.transform.translation.z = 0.0
    pcfm_main.camera_to_world.transform.rotation.x = 0.0
    pcfm_main.camera_to_world.transform.rotation.y = 0.0
    pcfm_main.camera_to_world.transform.rotation.z = 0.0
    pcfm_main.camera_to_world.transform.rotation.w = 1.0
    print("camera_to_world is None", pcfm_main.camera_to_world is None)
    
    transform = transforms.Compose([
        transforms.Resize(480),           # Resize the shorter side to 480 while maintaining aspect ratio
        transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
        transforms.ToTensor()            # Convert to tensor
    ])

    # Clear terminal
    print("clear terminal")
    print("\033[H\033[J", end="")
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            # pcfm_main.listen_tf()
            if image_subscriber.pil_image is not None and rscalc.info_received():
                image_tensor = transform(image_subscriber.pil_image)
                with torch.no_grad():
                    feat = model(image_tensor.unsqueeze(0).cuda())
                cluster_image = PCA_and_Cluster(feat)
                key_pixels = obtain_key_pixels(feat, cluster_image)
                key_points = rscalc.obtain_key_points(key_pixels)
                update_done = pcfm_main.update_pcfm(key_points, pcfm_threshold=1000, drop_range=0.5, drop_ratio=0.2)
                pcfm_main.save_pcfm("/home/fyp/llmbot2/src/sem_map/sem_map/pcfm.pkl")
                # convert to display type
                cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
                cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
                cv2.imshow('Cluster Image', cluster_image_uint8)
                cv2.waitKey(1)
            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught")
        sys.stdout.flush()
    finally:
        print("Shutting down")
        sys.stdout.flush()
        cv2.destroyAllWindows()
        executor.shutdown()
        image_subscriber.destroy_node()
        rscalc.destroy_node()
        pcfm_main.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    print("enter main")
    main()