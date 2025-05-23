import rclpy
import rclpy.serialization
from rclpy.node import Node
from interfaces.srv import SemanticQuery
import time
from .utils import *
from rclpy.executors import SingleThreadedExecutor
import struct
from interfaces.msg import ObjectSem
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import subprocess

class SemanticQueryClient(Node):
    def __init__(self):
        super().__init__('semantic_query_handler')

        config_file = read_config("config_query_socket_handler")
        self.port_num = config_file['socket_connection']['port_num']
        self.sem_map_clients = {}
        self.socket_receiver = SocketReceiver()
        self.get_logger().info(f"Semantic Query Client waiting for connection at port num {self.port_num}")
        self.socket_receiver.socket_connect(port_num=self.port_num)

        self.publisher_found_points = self.create_publisher(PointCloud, 'sem_points_found_client', 10)
        
        # 发现所有匹配的服务
        self.service_names = []
        self.discover_services()
        self.number_of_server = len(self.sem_map_clients.keys())
    
    def trigger_services_and_send(self, object_name, similarity):
        self.discover_services()
        results = self.trigger_all_queries(object_name, similarity)
        self.socket_receiver.conn.sendall(struct.pack('<L', self.number_of_server))
        list_service_name = list(results.keys())

        for i in range(self.number_of_server):
            message = ObjectSem()
            message.service_name = list_service_name[i]
            message.object_name = object_name
            message.similarity_threshold_rad = similarity
            message.labels = results[message.service_name]['labels']
            message.confs = results[message.service_name]['confs']
            message.points = results[message.service_name]['points']
            message.similarities = results[message.service_name]['similarities']
            serialized_message = rclpy.serialization.serialize_message(message)
            data_size = len(serialized_message)
            print(data_size)
            self.socket_receiver.conn.sendall(struct.pack('<L', data_size)+serialized_message)
            print("send")
            self.wait_handshake("send_next")

        point_cloud_msg = PointCloud()
        point_cloud_msg.header.frame_id = "map"
        for i in range(self.number_of_server):
            for j in range(len(results[list_service_name[i]]['points'])):
                point = Point32()
                point.x = results[list_service_name[i]]['points'][j].x
                point.y = results[list_service_name[i]]['points'][j].y
                point.z = results[list_service_name[i]]['points'][j].z
                point_cloud_msg.points.append(point)
        self.publisher_found_points.publish(point_cloud_msg)
    
    def wait_similarity(self):
        data = struct.unpack('d', self.socket_receiver.conn.recv(8))
        print(data)
        data = data[0]
        if not data:
            self.get_logger().error("Connection closed before receiving full data")
        print(data)
        self.get_logger().info(f"Received similarity: {data}")
        return data
    
    def wait_object_name(self):
        data_size = struct.unpack('<L', self.socket_receiver.conn.recv(4))[0]
        data = b""
        if data_size == 0:
            self.get_logger().info(f"received empty object name")
        while len(data) < data_size:
            packet = self.socket_receiver.conn.recv(data_size - len(data))
            if not packet:
                self.get_logger().error("Connection closed before receiving full data")
                break
            data += packet
        object_name = data.decode()
        self.get_logger().info(f"Received object name: {object_name}")
        return object_name

    def wait_handshake(self, handshake_msg="handshake"):
        self.get_logger().info(f"Waiting for handshake")
        handshake = self.socket_receiver.conn.recv(1024).decode()
        if handshake != handshake_msg:
            self.get_logger().error("Handshake failed")
            self.get_logger().error(f"Received: {handshake}, expect: {handshake_msg}")
            return False
        else:
            self.get_logger().info(f"Handshake {handshake_msg} successful")
            return True

    def send_confirmation_handshake(self, message="received"):
        self.socket_receiver.conn.sendall(message.encode())
        self.get_logger().info(f"Confirmation {message} Send")
        
    def discover_services(self):
        self.get_logger().info("Discovering services...")

        existing_clients = list(self.sem_map_clients.keys())
        self.get_logger().info(f"Existing clients: {existing_clients}")
        self.get_logger().info(f"Removing all clients ...")
        for service_name in existing_clients:
            self.destroy_client(self.sem_map_clients[service_name])
            del self.sem_map_clients[service_name]
            self.get_logger().warning(f'Removed client: {service_name}')

        try:
            # 执行ros2 service list命令并捕获输出
            result = subprocess.run(['ros2', 'service', 'list'], 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                # self.get_logger().info("当前运行的服务:\n" + result.stdout)
                self.service_names = [
                    name.strip() 
                    for name in result.stdout.splitlines() 
                    if (
                        name.strip().startswith('/semantic_query') and
                        '/' not in name.strip()[len('/semantic_query'):]
                    )
                ]
                self.get_logger().info(f"ros2 service list的semantic map: {self.service_names}")
            else:
                self.get_logger().error("获取服务列表失败: " + result.stderr)
                
        except Exception as e:
            self.get_logger().error("执行命令时出错: " + str(e))

            self.service_names = [name for name, _ in self.get_service_names_and_types()
                            if (name.startswith('/semantic_query') and 
                                '/' not in name[len('/semantic_query'):])]
            self.get_logger().info(f"当前的semantic map: {self.service_names}")

        # 为每个服务创建客户端
        for service_name in self.service_names:
            if service_name in self.sem_map_clients:
                self.get_logger().info(f"client {service_name} already exist")
                continue
            self.sem_map_clients[service_name] = self.create_client(
                SemanticQuery, 
                service_name
            )
            self.get_logger().info(f'Found service: {service_name}')
        self.get_logger().info(f"after adding: client list: {self.sem_map_clients.keys()}")
    
    def query_service(self, service_name, object_name, threshold):
        client = self.sem_map_clients.get(service_name)
        if not client:
            self.get_logger().warn(f'Service {service_name} not found!')
            return None
        self.get_logger().info(f'Calling service {service_name} with object name: {object_name} and threshold: {threshold}')

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {service_name} not available!')
            return None

        self.get_logger().info(f"client {service_name} is available")
        self.get_logger().info(f'Calling service {service_name} with object name: {object_name} and threshold: {threshold}')

        req = SemanticQuery.Request()
        req.object_name = object_name
        req.similarity_threshold_rad = threshold
        
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        try:
            self.get_logger().info(f'Service {service_name} called successfully')
            self.get_logger().info(f"Response: {future.result()}")
            return future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            return None
    
    def trigger_all_queries(self, object_name, threshold):
        results = {}
        for service_name in self.service_names:
            response = self.query_service(service_name, object_name, threshold)
            if response:
                results[service_name] = {
                    'similarities': response.similarities,
                    'points': response.points,
                    'labels': response.labels,
                    'confs': response.confs
                }
        return results

def main():
    rclpy.init()
    query_client = SemanticQueryClient()
    executor = SingleThreadedExecutor()
    executor.add_node(query_client)
    print("start")
    try:
        while rclpy.ok():
            print("Waiting for query ... ")
            query_client.wait_handshake("query")
            query_client.send_confirmation_handshake("send_object")
            object_name = query_client.wait_object_name()
            query_client.send_confirmation_handshake("send_sim")
            similarity_threshold = query_client.wait_similarity()
            query_client.trigger_services_and_send(object_name, similarity_threshold)
            query_client.send_confirmation_handshake("done")
    except KeyboardInterrupt:
        pass
    finally:
        query_client.socket_receiver.socket_close()
        time.sleep(5)
        query_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()