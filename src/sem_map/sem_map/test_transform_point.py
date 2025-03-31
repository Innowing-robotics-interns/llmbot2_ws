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

                points_cam = smm.transform_pixels_to_camera_frame_points(pixel_list)
                points_world = smm.transform_camera_frame_points_to_world_frame_points(points_cam)
                points_pixel = smm.transform_world_frame_points_to_camera_frame_points(points_world)
                pixel_depth_list = smm.transform_camera_frame_points_to_pixel_depth(points_pixel)
                if len(pixel_list)!=len(pixel_depth_list):
                    print("Error: pixel list and pixel cam inverse have different length")
                else:
                    print("1")
                    for i in range(len(pixel_list)):
                        if pixel_depth_list[i][2]==None:
                            print('Error: Depth None')
                            print('Error: %f, %f' % (pixel_list[i][0], pixel_list[i][1]))
                            print('Error: %f, %f' % (pixel_depth_list[i][0], pixel_depth_list[i][1]))
                        if pixel_list[i][0]!=pixel_depth_list[i][0]:
                            print('Error: %f, %f' % (pixel_list[i][0], pixel_list[i][1]))
                            print('Error: %f, %f, %f' % (pixel_depth_list[i][0], pixel_depth_list[i][1], pixel_depth_list[i][2]))
                        if pixel_list[i][1]!=pixel_depth_list[i][1]:
                            print('Error: %f, %f' % (pixel_list[i][0], pixel_list[i][1]))
                            print('Error: %f, %f, %f' % (pixel_depth_list[i][0], pixel_depth_list[i][1], pixel_depth_list[i][2]))
                        if pixel_depth_list[i][2]!=smm.depth[pixel_list[i][1]][pixel_list[i][0]]:
                            print('Error: %f, %f' % (pixel_list[i][0], pixel_list[i][1]))
                            print('Error: %f, %f, %f' % (pixel_depth_list[i][0], pixel_depth_list[i][1], pixel_depth_list[i][2]))

                # for i in range(len(points_cam)):
                #     if np.linalg.norm(np.array(points_pixel[i])-np.array(points_cam[i]))>0.01:
                #         print('Error: %f, %f, %f' % (points_pixel[i][0], points_pixel[i][1], points_pixel[i][2]))
                #         print('Error: %f, %f, %f' % (points_cam[i][0], points_cam[i][1], points_cam[i][2]))
                # for i in range(len(points_cam_inverse)):
                #     if np.linalg.norm(np.array(points_cam_inverse[i])-np.array(points_cam[i]))>0.01:
                #         print('Error: %f, %f, %f' % (points_cam_inverse[i][0], points_cam_inverse[i][1], points_cam_inverse[i][2]))
            else:
                print("No image or depth or trans")

            time.sleep(0.1)
        #####################################################
        # while rclpy.ok():
        #     executor.spin_once(timeout_sec=1.0)
        #     if smm.pil_image is not None and smm.depth is not None and smm.trans is not None:
        #         smm.update_depth()
        #         should_erase = []
        #         list_points = list(pcp.all_points.keys())
        #         for i in range(len(list_points)):
        #             if smm.should_erase_point(list_points[i]):
        #                 should_erase.append(list(list_points[i]))
        #         for i in range(len(should_erase)):
        #             pcp.all_points.pop(tuple(should_erase[i]), None)

        #         smm.pil_image = None
        #         points_transformed = smm.transform_points(pixel_list)
        #         pcp.update_all_points(points_transformed)
        #         pcp.publish_point_cloud()
        #         # display how many points are in the point cloud
        #         print('Number of points in the point cloud: %d' % len(points_transformed))
        #         print('First point: %f, %f, %f' % (points_transformed[0][0], points_transformed[0][1], points_transformed[0][2]))
        #     time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        smm.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
