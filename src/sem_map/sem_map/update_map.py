    
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import cv2
import rclpy.logging
import rclpy.logging
import torch
import time

# Before running this, add lang-seg folder to your python path
from .LoadLSeg import *
import torchvision.transforms as transforms
from test_lseg_zs import *

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

    
    transform = transforms.Compose([
        transforms.Resize(480),           # Resize the shorter side to 480 while maintaining aspect ratio
        transforms.CenterCrop((480, 480)),  # Crop the center to 480x480
        transforms.ToTensor()            # Convert to tensor
    ])

    # print("\033[H\033[J", end="")
    try:
        rclpy.logging.get_logger('update_map').info("Ready to connect to socket client.")
        socket_receiver.socket_connect()

        # Wait until all messages are received
        while rclpy.ok() and socket_receiver.pil_image is None or socket_receiver.depth is None or socket_receiver.translation is None or socket_receiver.info is None:
            socket_receiver.send_handshake('info')
            socket_receiver.get_info()
            socket_receiver.send_handshake('trans')
            socket_receiver.get_trans()
            socket_receiver.send_handshake('color')
            socket_receiver.get_color()
            socket_receiver.send_handshake('depth')
            socket_receiver.get_depth()

        save_every = 10
        c = 0
        while rclpy.ok():
            rclpy.logging.get_logger('update_map').info(f"Sending info hs...")
            socket_receiver.send_handshake('info')
            rclpy.logging.get_logger('update_map').info(f"Waiting for info...")
            socket_receiver.get_info()
            rclpy.logging.get_logger('update_map').info(f"Sending trans hs...")
            socket_receiver.send_handshake('trans')
            rclpy.logging.get_logger('update_map').info(f"Waiting for trans...")
            socket_receiver.get_trans()
            rclpy.logging.get_logger('update_map').info(f"Sending color hs...")
            socket_receiver.send_handshake('color')
            rclpy.logging.get_logger('update_map').info(f"Waiting for color...")
            socket_receiver.get_color()
            rclpy.logging.get_logger('update_map').info(f"Sending depth hs...")
            socket_receiver.send_handshake('depth')
            rclpy.logging.get_logger('update_map').info(f"Waiting for depth...")
            socket_receiver.get_depth()


            if True:
                rclpy.logging.get_logger('update_map').info(f"Received tf. trans: {socket_receiver.translation}; rot: {socket_receiver.rotation}")
                ax, ay, az = tf_transformations.euler_from_quaternion(socket_receiver.rotation)
                rclpy.logging.get_logger('update_map').info(f"Received angle: {az}")

                rscalc.update_depth(socket_receiver.depth)
                rscalc.update_intr(socket_receiver.info)

                rclpy.logging.get_logger('update_map').info(f"Received depth: {socket_receiver.depth.shape}")
                # point_dict = {}
                # for x in range(640):
                #     for y in range(480):
                #         if x % 50 == 0 and y % 50 == 0:
                #             point_dict[tuple(rscalc.calculate_point(y, x))] = 0
                # pc_manager.publish_transformed_point_cloud(socket_receiver.translation, socket_receiver.rotation, point_dict.keys())

                image_tensor = transform(socket_receiver.pil_image)
                with torch.no_grad():
                    feat = model(image_tensor.unsqueeze(0).cuda())
                cluster_image = PCA_and_Cluster(feat.half())
                key_pixels = obtain_key_pixels(feat, cluster_image)
                rscalc.update_depth(socket_receiver.depth)
                key_points = rscalc.obtain_key_points(key_pixels)

                pcfm_main.update_pcfm(key_points, socket_receiver.translation, socket_receiver.rotation, 
                                      pcfm_threshold=1000, drop_range=0.5, drop_ratio=0.2)

                rclpy.logging.get_logger('update_map').info(f"pcfm point count: {len(pcfm_main.pcfm)}")
                pc_manager.publish_point_cloud(pcfm_main.pcfm.keys())

                # cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
                # cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
                # cv2.imshow('Cluster Image', cluster_image_uint8)
                # cv2.waitKey(1)

                c += 1
                if c % save_every == 0:
                    map_saved = pcfm_main.save_pcfm(map_path)
                    if map_saved:
                        rclpy.logging.get_logger('update_map').info(f"Map saved at {pcfm_main.curr_time}")
                    c = 0

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