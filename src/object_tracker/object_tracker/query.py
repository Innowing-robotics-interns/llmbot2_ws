import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TextPublisher(Node):
    def __init__(self):
        super().__init__('text_publisher')
        self.publisher_ = self.create_publisher(String, 'text_topic', 10)
        timer_period = 0.5  # seconds (adjust as needed)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.input_text = ""

    def timer_callback(self):
        msg = String()
        print("\033[H\033[J", end="")
        print(f"Inputed Query: {self.input_text}\n")
        self.input_text = input("Enter new query: ")
        msg.data = f"{self.input_text}"
        self.publisher_.publish(msg)
        # self.get_logger().info(f"Publishing: '{msg.data}'")

def main(args=None):
    rclpy.init(args=args)
    text_publisher = TextPublisher()
    try:
        rclpy.spin(text_publisher)
    except KeyboardInterrupt:
        pass
    text_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()