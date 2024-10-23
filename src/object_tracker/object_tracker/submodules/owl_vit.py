import rclpy
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
        # cv2.imshow("Image Window", self.cv_image)
        # cv2.waitKey(100)  # Wait for a key press for 1 millisecond

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

class OWLViT:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    def predict(self, image, text):
        target_sizes = torch.tensor([image.size[::-1]])
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        threshold = 0.02
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
        return results
    
    def get_best_box(self, results):
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        # for box, score, label in zip(boxes, scores, labels):
        #      box = [round(i, 2) for i in box.tolist()]
        #      print(f"Detected {Label} with confidence {round(score.item(), 3)} at location {box}")
        if len(scores) > 0:
            max_score_index = scores.argmax()
            box = boxes[max_score_index]
            score = scores[max_score_index]
            label = labels[max_score_index]
            box = [int(i) for i in box.tolist()]
            return box, score, label
        return None, None, None
    

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    text_subscriber = TextSubscriber()

    # Initialize processor and model once
    owlvit = OWLViT()

    executor = MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(text_subscriber)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if text_subscriber.entering and image_subscriber.pil_image is not None:
                text_subscriber.entering = False
                Label = text_subscriber.text
                text = f"a photo of a {Label}"

                image = image_subscriber.pil_image

                results = owlvit.predict(image, text_subscriber.text)

                box, score, label = owlvit.get_best_box(results)
                if box is not None:
                    x_min, y_min, x_max, y_max = [int(coord) for coord in box]
                    cv2.rectangle(image_subscriber.cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image_subscriber.cv_image, f"{Label}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the image with bounding boxes
                cv2.imshow("Detected Objects", image_subscriber.cv_image)
                cv2.waitKey(500)
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