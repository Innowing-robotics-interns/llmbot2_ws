    
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import cv2
import rclpy.logging
import rclpy.logging
import torch
import time

# Before running this, add lang-seg folder to your python path
# from .LoadLSeg import *
# import torchvision.transforms as transforms
# from test_lseg_zs import *

from sklearn.cluster import KMeans

from .map_utils import *
import sys
    
def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"

    rscalc = RealSensePointCalculator()
    socket_receiver = SocketReceiver()
    pcfm_main = PointCloudFeatureMap()

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    pc_manager = PointCloudManager()
    executor.add_node(pc_manager)

    
    # transform = transforms.Compose([
    #     transforms.Resize(480),           # Resize the shorter side to 480 while maintaining aspect ratio
    #     transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
    #     transforms.ToTensor()            # Convert to tensor
    # ])

    # print("\033[H\033[J", end="")
    try:
        socket_receiver.socket_connect()
        rclpy.logging.get_logger('update_map').info("Ready to connect to socket client.")

        # Wait until all messages are received
        while rclpy.ok() and socket_receiver.pil_image is None or socket_receiver.depth is None or socket_receiver.tf is None or socket_receiver.info is None:
            socket_receiver.send_handshake('info')
            socket_receiver.get_info()
            socket_receiver.send_handshake('trans')
            socket_receiver.get_trans()
            socket_receiver.send_handshake('color')
            socket_receiver.get_color()
            socket_receiver.send_handshake('depth')
            socket_receiver.get_depth()

        while rclpy.ok():
            socket_receiver.send_handshake('info')
            socket_receiver.get_info()
            socket_receiver.send_handshake('trans')
            socket_receiver.get_trans()
            socket_receiver.send_handshake('color')
            socket_receiver.get_color()
            socket_receiver.send_handshake('depth')
            socket_receiver.get_depth()

            if True:
                rclpy.logging.get_logger('update_map').info(f"Received transformation data: {socket_receiver.tf}")

                rscalc.update_depth(socket_receiver.depth)
                rscalc.update_intr(socket_receiver.info)
                # display depth image
                # shape of depth image is (480, 640)
                # cv2.imshow('Depth Image', socket_receiver.depth)
                # cv2.waitKey(1)

                point_dict = {}
                for x in range(640):
                    for y in range(480):
                        if x % 10 == 0 and y % 10 == 0:
                            point_dict[tuple(rscalc.calculate_point(y, x))] = socket_receiver.depth[y, x]
                pc_manager.publish_point_cloud(point_dict)

                # image_tensor = transform(socket_receiver.pil_image)
                # with torch.no_grad():
                #     feat = model(image_tensor.unsqueeze(0).cuda())
                # cluster_image = PCA_and_Cluster(feat)
                # key_pixels = obtain_key_pixels(feat, cluster_image)
                # rscalc.update_depth(socket_receiver.depth)
                # key_points = rscalc.obtain_key_points(key_pixels)
                # # print all keypoints

                # pcfm_main.update_pcfm(key_points,tf_array=socket_receiver.tf, 
                #                       pcfm_threshold=1000, drop_range=0.5, drop_ratio=0.2)
                # rclpy.logging.get_logger('update_map').info(f"pcfm point count: {len(pcfm_main.pcfm)}")
                # pc_manager.publish_point_cloud(pcfm_main.pcfm)

                # # map_saved = pcfm_main.save_pcfm(map_path)
                # # if map_saved:
                # #     print("Map saved at", pcfm_main.curr_time)
                # # convert to display type
                # cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
                # cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
                # cv2.imshow('Cluster Image', cluster_image_uint8)
                # cv2.waitKey(1)

                ###| Test Display \###
                # if socket_receiver.color is not None and socket_receiver.color.size > 0:
                #     rclpy.logging.get_logger('update_map').info(str(socket_receiver.color.shape))
                #     cv2.imshow('Color Image', cv2.cvtColor(socket_receiver.color, cv2.COLOR_RGB2BGR))
                #     cv2.waitKey(1)
                # else:
                #     rclpy.logging.get_logger('update_map').info("Received empty or invalid image.")
                ###\ Test Display |###

            time.sleep(0.1)

    except KeyboardInterrupt:
        rclpy.logging.get_logger('update_map').info("KeyboardInterrupt")
        # sys.stdout.flush()
    finally:
        rclpy.logging.get_logger('update_map').info("Shutting down...")
        # sys.stdout.flush()
        socket_receiver.server_socket.close()
        pc_manager.destroy_node()
        executor.shutdown()
        cv2.destroyAllWindows()
        rclpy.shutdown()



if __name__ == '__main__':
    print("enter main")
    main()