import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
import rclpy.logging
import rclpy.logging
import time

# Before running this, add lang-seg folder to your python path
from .LoadLSeg import *
from test_lseg_zs import *

from .map_utils import *

def main(args=None):
    map_path = "/home/fyp/llmbot2_ws/src/sem_map/sem_map/pcfm.pkl"

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    sfpc = ServerFeaturePointCloudMap()
    pc_manager = PointCloudManager()
    text_query = TextQueryReceiver(sfpc=sfpc)
    executor.add_node(pc_manager)

    # print("\033[H\033[J", end="")
    try:
        rclpy.logging.get_logger("update_map").info(
            "Ready to connect to text query socket client."
        )
        text_query.start_listening(port_num=6000)
        rclpy.logging.get_logger("update_map").info(
            "Ready to connect to image sender socket client."
        )
        sfpc.init_socket(port_num=5555)
        sfpc.set_model(model)

        save_every = 10
        c = 0
        while rclpy.ok():
            sfpc.receive_data_and_update()

            pc_manager.publish_point_cloud(sfpc.fpc.keys())

            c += 1
            if c % save_every == 0:
                sfpc.save_fpc(map_path)
                c = 0
            time.sleep(0.1)

    except KeyboardInterrupt:
        rclpy.logging.get_logger("update_map").info("KeyboardInterrupt")
    finally:
        text_query.stop_listening()
        rclpy.logging.get_logger("update_map").info("Shutting down...")
        sfpc.socket_receiver.server_socket.close()
        pc_manager.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    print("enter main")
    main()
