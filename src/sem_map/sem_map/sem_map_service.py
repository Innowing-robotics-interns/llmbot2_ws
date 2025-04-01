import rclpy
from .semantic_map import *

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    sms = SemanticMapService()
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
