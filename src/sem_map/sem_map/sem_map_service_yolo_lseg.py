import rclpy
from .semantic_map import *
from .utils import *
import time

image_processor_selection = 'yolo_lseg'

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    sms = SemanticMapService(image_processor_selection)
    executor.add_node(sms)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sms.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
