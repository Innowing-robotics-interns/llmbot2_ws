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
        sfpc.socket_receiver.socket_connect()

        # Wait until all messages are received
        while (
            rclpy.ok()
            and sfpc.socket_receiver.pil_image is None
            or sfpc.socket_receiver.depth is None
            or sfpc.socket_receiver.translation is None
            or sfpc.socket_receiver.info is None
        ):
            sfpc.handshake_receive_data()

        save_every = 10
        c = 0
        while rclpy.ok():
            sfpc.handshake_receive_data()

            if True:
                rclpy.logging.get_logger("update_map").info(
                    f"Received tf. trans: {sfpc.socket_receiver.tras[:3]}; rot: {sfpc.socket_receiver.tras[3:]}"
                )
                ax, ay, az = tf_transformations.euler_from_quaternion(
                    sfpc.socket_receiver.tras[3:]
                )
                rclpy.logging.get_logger("update_map").info(f"Received angle: {az}")

                sfpc.rscalc.update_depth(sfpc.socket_receiver.depth)
                sfpc.rscalc.update_intr(sfpc.socket_receiver.info)

                rclpy.logging.get_logger("update_map").info(
                    f"Received depth: {sfpc.socket_receiver.depth.shape}"
                )

                sfpc.set_model(model)
                sfpc.update_feature()

                sfpc.feat_to_points()

                pcfm_main.update_pcfm(
                    key_points,
                    socket_receiver.translation,
                    socket_receiver.rotation,
                    pcfm_threshold=1000,
                    drop_range=0.5,
                    drop_ratio=0.2,
                )

                rclpy.logging.get_logger("update_map").info(
                    f"pcfm point count: {len(pcfm_main.pcfm)}"
                )
                pc_manager.publish_point_cloud(pcfm_main.pcfm.keys())

                # cluster_image_normalized = cv2.normalize(cluster_image, None, 0, 255, cv2.NORM_MINMAX)
                # cluster_image_uint8 = cluster_image_normalized.astype(np.uint8)
                # cv2.imshow('Cluster Image', cluster_image_uint8)
                # cv2.waitKey(1)

                c += 1
                if c % save_every == 0:
                    map_saved = pcfm_main.save_pcfm(map_path)
                    if map_saved:
                        rclpy.logging.get_logger("update_map").info(
                            f"Map saved at {pcfm_main.curr_time}"
                        )
                    c = 0

            time.sleep(0.1)

    except KeyboardInterrupt:
        rclpy.logging.get_logger("update_map").info("KeyboardInterrupt")
        # sys.stdout.flush()
    finally:
        rclpy.logging.get_logger("update_map").info("Shutting down...")
        # sys.stdout.flush()
        socket_receiver.server_socket.close()
        pc_manager.destroy_node()
        executor.shutdown()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    print("enter main")
    main()
