import rclpy
from rclpy.node import Node
from .semantic_map import *
from .utils import *
# from .image_processors import *
import time

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    smn = SemanticMapService()
    executor.add_node(smn)

    # lseg_processor = LSegFeatImageProcessor()
    # yolo_processor = YOLO_LSeg_ImageProcessor()
    # smm = SemanticMapMaintainer(image_semantic_extractor=yolo_processor)
    # pcp = PointCloudPublisher()
    # executor = rclpy.executors.SingleThreadedExecutor()
    # use multithread executor instead
    # executor.add_node(smm)
    # executor.add_node(pcp)

    # test erase point (without semantic)
    # create pixel list of all pixels in the image of size (640, 480)
    # pixel_list = []
    # for i in range(640):
    #     for j in range(480):
    #         if i % 50 != 0 or j % 50 != 0:
    #             continue
    #         pixel_list.append((i, j))

    try:
        executor.spin()
        # test building semnatic map with SemanticMapNode
        # test building semantic map
        # while rclpy.ok():
        #     executor.spin_once(timeout_sec=1.0)
        #     if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
        #         # get average depth
        #         sum_depth = 0
        #         for i in range(640):
        #             for j in range(480):
        #                 if smm.depth[j, i] is not None:
        #                     sum_depth += smm.depth[j, i]
        #         avg_depth = sum_depth / (640 * 480)
        #         print("avg_depth: ", avg_depth)
        #         smm.update_depth()
        #         smm.erase_old_points()
        #         feat_list, pixel_list, label_list = smm.get_feat_pixel_label()
        #         print("info")
        #         print(type(feat_list))
        #         print(type(pixel_list))
        #         print(pixel_list)
        #         print(label_list)

        #         points_transformed = smm.transform_to_points(pixel_list)
        #         smm.update_semantic_point_cloud(points_transformed, feat_list, label_list)
        #         pcp.publish_input_point_cloud(smm.semantic_point_cloud.keys())

        #         smm.pil_image = None
        #     time.sleep(0.1)
        #######################################################
        # test erase point (without semantic)
        # while rclpy.ok():
        #     executor.spin_once(timeout_sec=1.0)
        #     if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
        #         smm.update_depth()

        #         if smm.two_frame_difference_high()==False:
        #             should_erase = []
        #             list_points = list(pcp.all_points.keys())
        #             for i in range(len(list_points)):
        #                 if smm.should_erase_point(list_points[i]):
        #                     should_erase.append(list(list_points[i]))
        #             for i in range(len(should_erase)):
        #                 pcp.all_points.pop(tuple(should_erase[i]), None)

        #         smm.pil_image = None
        #         points_transformed = smm.transform_to_points(pixel_list)
        #         pcp.update_all_points(points_transformed)
        #         pcp.publish_point_cloud()
        #         print(smm.two_frame_difference_high())
        #         # display how many points are in the point cloud
        #         # print('Number of points in the point cloud: %d' % len(points_transformed))
        #         # print('First point: %f, %f, %f' % (points_transformed[0][0], points_transformed[0][1], points_transformed[0][2]))
        #     time.sleep(0.1)
        #######################################################
        # Test transforming the pixels and depths to the points
        # can view in rviz2
        # while rclpy.ok():
        #     executor.spin_once(timeout_sec=1.0)
        #     if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
        #         smm.update_depth()
        #         smm.pil_image = None
        #         points_transformed = smm.transform_points(pixel_list)
        #         pcp.update_all_points(points_transformed)
        #         pcp.publish_point_cloud()
        #         # display how many points are in the point cloud
        #         print('Number of points in the point cloud: %d' % len(points_transformed))
        #         print('First point: %f, %f, %f' % (points_transformed[0][0], points_transformed[0][1], points_transformed[0][2]))

    except KeyboardInterrupt:
        pass
    finally:
        # smm.destroy_node()
        smn.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
