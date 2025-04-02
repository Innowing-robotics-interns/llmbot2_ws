import rclpy
from .semantic_map import *
from .utils import *
import time
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

image_processor_selection = 'lseg_feat'

class SemanticMapService(Node):
    def __init__(self):
        super().__init__('sem_map_service_'+image_processor_selection)

        config = read_config("config_"+image_processor_selection)
        map_config = config['map_config']
        self.map_core = load_semantic_map_core(map_config, image_processor_selection)

        # Other variables
        self.subscriber = self.create_subscription(ColorDepthTrans, 'color_depth_trans', self.callback, 10)
        self.topic_pil_image = None
        self.topic_depth = None
        self.topic_trans = None

        self.publisher_mapped_points = self.create_publisher(PointCloud, 'sem_points_'+image_processor_selection, 10)
        self.publisher_found_points = self.create_publisher(PointCloud, 'sem_points_found_'+image_processor_selection, 10)

        self.query_service = self.create_service(
            SemanticQuery,
            'semantic_query_'+image_processor_selection,
            self.handle_semantic_query,
            callback_group=ReentrantCallbackGroup()
        )

        self.map_rwlock = threading.RLock()
        self.data_lock = threading.Lock()

        self.save_counter = 0
        self.update_interval = 0.01
        self.create_timer(self.update_interval, self.map_update)

    def publish_input_point_cloud(self, point_list, publisher_selected):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "map"
        for point in point_list:
            msg_point = Point32()
            msg_point.x = point[0]
            msg_point.y = point[1]
            msg_point.z = point[2]
            point_cloud_msg.points.append(msg_point)
        publisher_selected.publish(point_cloud_msg)
    
    def callback(self, msg):
        if msg.rotation_x == 0.0 and msg.rotation_y == 0.0 and msg.rotation_z == 0.0 and msg.rotation_w == 0.0:
            return
        image_height = msg.height
        image_width = msg.width
        color_image = np.frombuffer(msg.data_color, dtype=np.uint8).reshape(image_height, image_width, 3)
        depth_image = np.frombuffer(msg.data_depth, dtype=np.uint16).reshape(image_height, image_width)
        with self.data_lock:
            self.topic_pil_image = Image.fromarray(color_image)
            self.topic_depth = depth_image
            self.topic_trans = np.array([msg.translation_x, msg.translation_y, msg.translation_z, msg.rotation_x, msg.rotation_y, msg.rotation_z, msg.rotation_w])
    
    def map_update(self):
        with self.map_rwlock:
            if self.topic_pil_image is not None and self.topic_depth is not None and self.topic_trans is not None:
                self.map_core.update_info(self.topic_depth, self.topic_pil_image, self.topic_trans)
                self.map_core.erase_old_points()

                feat_list, pixel_list, label_list = self.map_core.get_feat_pixel_label()
                points_transformed = self.map_core.transform_to_points(pixel_list)
                rclpy.logging.get_logger('sem_map_service').info(f'points: {points_transformed}')
                self.map_core.update_semantic_point_cloud(points_transformed, feat_list, label_list)

                self.publish_input_point_cloud(self.map_core.semantic_point_cloud.keys(), self.publisher_mapped_points)

                self.topic_pil_image = None
                self.topic_depth = None
                self.topic_trans = None
                
                self.save_counter += 1
                if self.save_counter % self.map_core.map_save_interval == 0:
                    self.map_core.save_map()
                    rclpy.logging.get_logger('sem_map_service').info(f'save map')
                    self.save_counter = 0

    def handle_semantic_query(self, request, response):
        rclpy.logging.get_logger('sem_map_service').info(f'handle_semantic_query: {request.object_name}')
        with self.map_rwlock:
            found_similarities, found_points, found_labels = self.map_core.similarity_search(request.similarity_threshold_rad, request.object_name)
        response.similarities = found_similarities
        point_list = []
        for point_tuple in found_points:
            new_point = Point()
            new_point.x = point_tuple[0]
            new_point.y = point_tuple[1]
            new_point.z = point_tuple[2]
            point_list.append(new_point)

        response.points = point_list
        response.labels = found_labels
        rclpy.logging.get_logger('sem_map_service').info(f'found points: {found_points}')

        self.publish_input_point_cloud(found_points, self.publisher_found_points)

        return response

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
