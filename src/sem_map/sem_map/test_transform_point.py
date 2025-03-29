import rclpy
from rclpy.node import Node
from .semantic_map import SemanticMapMaintainer
from .utils import *
# from .image_processors import *
import time

def main(args=None):
    rclpy.init(args=args)
    lseg_processor = None
    smm = SemanticMapMaintainer(image_semantic_extractor=lseg_processor)
    pcp = PointCloudPublisher()
    # executor = rclpy.executors.SingleThreadedExecutor()
    # use multithread executor instead
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(smm)
    executor.add_node(pcp)

    # create pixel list of all pixels in the image of size (640, 480)
    pixel_list = []
    for i in range(640):
        for j in range(480):
            if i % 50 != 0 or j % 50 != 0:
                continue
            pixel_list.append((i, j))

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=1.0)
            if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
                smm.update_depth()
                smm.pil_image = None
                points_transformed = smm.transform_points(pixel_list)
                pcp.update_all_points(points_transformed)
                pcp.publish_point_cloud()
                # display how many points are in the point cloud
                print('Number of points in the point cloud: %d' % len(points_transformed))
                print('First point: %f, %f, %f' % (points_transformed[0][0], points_transformed[0][1], points_transformed[0][2]))
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        smm.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
