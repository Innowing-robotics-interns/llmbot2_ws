import rclpy
from rclpy.node import Node
import subprocess
import time

class ServiceLister(Node):
    def __init__(self):
        super().__init__('service_lister')
        # 设置定时器，每2秒执行一次list_services
        self.service_names = []
        self.timer = self.create_timer(2.0, self.list_services)
        
    def list_services(self):
        try:
            # 执行ros2 service list命令并捕获输出
            result = subprocess.run(['ros2', 'service', 'list'], 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                self.get_logger().info("当前运行的服务:\n" + result.stdout)
                # 修改后的服务名称过滤逻辑
                self.service_names = [
                    name.strip() 
                    for name in result.stdout.splitlines() 
                    if (
                        name.strip().startswith('/semantic_query') and
                        '/' not in name.strip()[len('/semantic_query'):]
                    )
                ]
                self.get_logger().info(f"当前的semantic map: {self.service_names}")
            else:
                self.get_logger().error("获取服务列表失败: " + result.stderr)
                
        except Exception as e:
            self.get_logger().error("执行命令时出错: " + str(e))

def main(args=None):
    rclpy.init(args=args)
    node = ServiceLister()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
