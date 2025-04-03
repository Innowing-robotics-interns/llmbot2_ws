import rclpy
import rclpy.serialization
from rclpy.node import Node
from interfaces.srv import SemanticQuery
import time
from .utils import *
from rclpy.executors import SingleThreadedExecutor
import struct
from interfaces.msg import ObjectSem

class SemanticQueryClient(Node):
    def __init__(self):
        super().__init__('semantic_query_handler')

        config_file = read_config("config_query_socket_handler")
        self.port_num = config_file['socket_connection']['port_num']

        self.sem_map_clients = {}

        self.socket_receiver = SocketReceiver()
        self.get_logger().info(f"Semantic Query Client waiting for connection at port num {self.port_num}")
        self.socket_receiver.socket_connect(port_num=self.port_num)
        
        # 发现所有匹配的服务
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
            message.points = results[message.service_name]['points']
            message.similarities = results[message.service_name]['similarities']
            serialized_message = rclpy.serialization.serialize_message(message)
            data_size = len(serialized_message)
            print(data_size)
            self.socket_receiver.conn.sendall(struct.pack('<L', data_size)+serialized_message)
            print("send")
            self.wait_handshake("send_next")
    
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
        """自动发现所有/semantic_query开头的服务"""
        service_names = [name for name, _ in self.get_service_names_and_types()
                        if (name.startswith('/semantic_query') and 
                            '/' not in name[len('/semantic_query'):])]
        print(service_names)
        
        list_of_client = self.sem_map_clients.keys()
        for client in list_of_client:
            if client not in service_names:
                self.sem_map_clients.pop(client)
                self.get_logger().info(f"delete client: {client}")

        # 为每个服务创建客户端
        for service_name in service_names:
            if service_name in self.sem_map_clients:
                self.get_logger().info(f"client {service_name} already exist")
                continue
            self.sem_map_clients[service_name] = self.create_client(
                SemanticQuery, 
                service_name
            )
            self.get_logger().info(f'Found service: {service_name}')
    
    def query_service(self, service_name, object_name, threshold):
        """向指定服务发送请求"""
        client = self.sem_map_clients.get(service_name)
        if not client:
            self.get_logger().warn(f'Service {service_name} not found!')
            return None
            
        req = SemanticQuery.Request()
        req.object_name = object_name
        req.similarity_threshold_rad = threshold
        
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        try:
            return future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            return None
    
    def trigger_all_queries(self, object_name, threshold):
        """触发所有已发现服务的查询"""
        results = {}
        for service_name in self.sem_map_clients:
            response = self.query_service(service_name, object_name, threshold)
            if response:
                results[service_name] = {
                    'similarities': response.similarities,
                    'points': response.points,
                    'labels': response.labels
                }
        return results

# 使用示例
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
        query_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()