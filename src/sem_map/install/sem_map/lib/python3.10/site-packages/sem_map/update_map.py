    
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

from .utils import *
    
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rscalc = RealSensePointCalculator()
    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(rscalc)
    
    transform = transforms.Compose([
        transforms.Resize(480),           # Resize the shorter side to 480 while maintaining aspect ratio
        transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
        transforms.ToTensor()            # Convert to tensor
    ])

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if image_subscriber.pil_image is not None and rscalc.info_received():
                image_tensor = transform(image_subscriber.pil_image)
                with torch.no_grad():
                    feat = model(image_tensor.unsqueeze(0).cuda())
                cluster_image = PCA_and_Cluster(feat)

                key_pixels = obtain_key_pixels(feat, cluster_image)
                key_points = rscalc.obtain_key_points(key_pixels)
                update_pcfm(key_points)
                print(len(pcfm.keys()))
                # convert to display type
                cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
                cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
                # display cluster image of shape (480, 480) with 10 classes (0-9)
                cv2.imshow('Cluster Image', cluster_image_uint8)
                cv2.waitKey(1)  # Add this to allow OpenCV to process window events

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down")
        cv2.destroyAllWindows()
        executor.shutdown()
        image_subscriber.destroy_node()
        rscalc.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()