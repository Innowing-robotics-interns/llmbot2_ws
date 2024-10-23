import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import torch
from groundingdino.util.inference import load_model, predict, annotate, load_image
import groundingdino.datasets.transforms as T
import cv2
import time

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Change this to your image topic
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.pil_image = None
        self.cv_image = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rgb_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        self.cv_image = rgb_image  # Store OpenCV image for later use
        self.pil_image = PILImage.fromarray(cv_image)

class TextSubscriber(Node):
    def __init__(self):
        super().__init__('text_subscriber')
        self.subscription = self.create_subscription(
            String,
            'text_topic',  # Change this to your text topic
            self.text_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.text = None
        self.entering = False

    def text_callback(self, msg):
        self.text = msg.data
        self.entering = True
        print(self.text)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    text_subscriber = TextSubscriber()

    # Initialize processor and model once
    model = load_model("~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/fyp/weights/groundingdino_swint_ogc.pth")
    IMAGE_PATH = "/home/fyp/image.jpg"
    TEXT_PROMPT = "chair . person . dog ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    print("load done")


    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(text_subscriber)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if text_subscriber.entering and image_subscriber.pil_image is not None:
                text_subscriber.entering = False
                cv2.imwrite("/home/fyp/image.jpg", image_subscriber.cv_image)
                TEXT_PROMPT = text_subscriber.text

                image_source, image = load_image(IMAGE_PATH)

                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
                print(boxes)
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

                cv2.imshow("Detected Objects", annotated_frame)
                cv2.waitKey(500)
                # cv2.destroyAllWindows()

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