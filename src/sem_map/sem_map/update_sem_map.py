import rclpy
from rclpy.node import Node

from .semantic_map import SemanticMapMaintainer
from .image_processors import *


def main(args=None):
    rclpy.init(args=args)
    lseg_processor = LSegFeatImageProcessor()
    smm = SemanticMapMaintainer(image_semantic_extractor=lseg_processor)
    executor = rclpy.executors.MultiThreadedExecutor()

    # executor add nodes
    executor.add_node(smm)

    try:
        while rclpy.ok():
            if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
                smm.sem_map.add_semantic_point(image=smm.pil_image, trans=smm.trans)
                smm.sem_map.update_depth(smm.depth)
                smm.pil_image = None
    except KeyboardInterrupt:
        pass
    finally:
        smm.destroy_node()
        executor.shutdown()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
