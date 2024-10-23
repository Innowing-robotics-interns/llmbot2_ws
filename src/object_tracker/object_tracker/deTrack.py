
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time
from .submodules.owl_vit import OWLViT, ImageSubscriber, TextSubscriber
from .submodules.sam import SAM
from .submodules.XMem_track import XMemTrack
from .submodules import calc_point
import numpy as np

    

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    text_subscriber = TextSubscriber()

    # Initialize processor and model once
    owlvit = OWLViT()
    sam = SAM()
    xmem = XMemTrack()
    point_calculator = calc_point.RealSensePointCalculator()

    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(text_subscriber)
    executor.add_node(point_calculator)

    print("\033[H\033[J", end="")
    print("Waiting a query to start tracking")
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if text_subscriber.entering and image_subscriber.pil_image is not None:
                print("\033[H\033[J", end="")
                text_subscriber.entering = False
                Label = text_subscriber.text
                print(f"Detecting {Label}...")
                text = f"a photo of a {Label}"
                image = image_subscriber.pil_image
                results = owlvit.predict(image, text_subscriber.text)
                box, score, label = owlvit.get_best_box(results)
                if box is None:
                    print("No object detected\n")
                    print("Waiting a query to start tracking")
                else:
                    print(f"Object detected with score {score}")
                    print("Initializing tracker...")
                    mask = sam.predict_from_boxes(image_subscriber.cv_image, np.array(box))
                    processor, prediction = xmem.initialize(1, mask, image_subscriber.cv_image)
                    print("Tracking...")
                    with torch.cuda.amp.autocast(enabled=True):
                        while rclpy.ok() and not text_subscriber.entering:
                            executor.spin_once(timeout_sec=0.01)
                            pixel = xmem.center_of_mask(prediction)
                            if pixel:
                                point = point_calculator.calculate_point(pixel[0], pixel[1])
                                point_calculator.get_logger().info(f"3D point: {point}")
                                if point:
                                    if point[0] < 0.01 and point[1] < 0.01 and point[2] < 0.01:
                                        print("Object is too close")
                                    else:
                                        point_calculator.publish_transform(point)
                            processor, prediction = xmem.track(processor, image_subscriber.cv_image)
                            cv2.imshow("mask", prediction*200)
                            cv2.waitKey(80)
                            time.sleep(0.05)

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        image_subscriber.destroy_node()
        text_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()