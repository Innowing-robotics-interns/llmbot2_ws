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


from .map_utils import *


def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    sfpc = ServerFeaturePointCloudMap()
    pc_manager = PointCloudManager()
    executor.add_node(pc_manager)

    # print("\033[H\033[J", end="")
    try:
        rclpy.logging.get_logger("update_map").info(
            "Ready to connect to socket client."
        )
        sfpc.init_socket(port_num=5555)
        sfpc.set_model(model)

        save_every = 10
        c = 0
        while rclpy.ok():
            sfpc.receive_data_and_update()

            rclpy.logging.get_logger("update_map").info(
                f"Received tf. trans: {sfpc.trans[:3]}; rot: {sfpc.trans[3:]}"
            )
            ax, ay, az = tf_transformations.euler_from_quaternion(
                sfpc.trans[3:]
            )
            rclpy.logging.get_logger("update_map").info(f"Received angle: {az}")

            rclpy.logging.get_logger("update_map").info(
                f"Received depth: {sfpc.depth.shape}"
            )

            pc_manager.publish_point_cloud(sfpc.fpc.keys())

            # cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
            # cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
            # cv2.imshow('Cluster Image', cluster_image_uint8)
            # cv2.waitKey(1)

            c += 1
            if c % save_every == 0:
                sfpc.save_fpc(map_path)
                c = 0
            time.sleep(0.1)

    except KeyboardInterrupt:
        rclpy.logging.get_logger("update_map").info("KeyboardInterrupt")
        # sys.stdout.flush()
    finally:
        rclpy.logging.get_logger("update_map").info("Shutting down...")
        # sys.stdout.flush()
        sfpc.socket_receiver.server_socket.close()
        pc_manager.destroy_node()
        executor.shutdown()
        # cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    print("enter main")
    main()
