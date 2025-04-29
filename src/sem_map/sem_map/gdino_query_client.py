import rclpy
from rclpy.node import Node
import socket
import pickle
from .utils import read_config
import numpy as np
from PIL import Image

from groundingdino.util.inference import load_model, predict, annotate, load_image_wo_PIL
import groundingdino.datasets.transforms as T

class GDinoQueryClient(Node):
    def __init__(self):
        super().__init__('image_transform_client')

        self.image_width = 640
        self.image_height = 480
        self.gd_model = load_model("~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/fyp/weights/groundingdino_swint_ogc.pth")
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.30
        print("load done")

        config_file = read_config("config_gdino_query_client")
        self.port_num = config_file['socket_connection']['port_num']
        self.connect_to = config_file['socket_connection']['connect_to']

        # Socket client setup
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.connect_to, self.port_num))
        self.get_logger().info("Connected to server.")

        self.is_connected = True

        # Timer to periodically receive and process data
        self.timer = self.create_timer(0.1, self.run)
    
    def run(self):
        if not self.is_connected:
            return
        try:
            # Receive data from the server
            color_image_msg, query = self.receive_data()
            print("received data")
            if color_image_msg is not None and query is not None:
                print("not none")
                # Process the image and send back the response
                xy, phrases, logits = self.process_image(color_image_msg, query)
                print("processed data")
                self.send_response(xy, phrases, logits)
                print("sent data")
            else:
                print("none")
        except Exception as e:
            self.get_logger().error(f"Error in run loop: {e}")
            self.cleanup_connection()

    def receive_data(self):
        if not self.is_connected:
            return
        try:
            # Receive size of the byte stream
            size_data = self.client_socket.recv(4)
            if not size_data:
                self.get_logger().warning("Server disconnected.")
                self.client_socket.close()
                return

            size = int.from_bytes(size_data, 'big')

            # Receive the byte stream
            byte_stream = b""
            while len(byte_stream) < size:
                packet = self.client_socket.recv(size - len(byte_stream))
                if not packet:
                    self.get_logger().warning("Server disconnected.")
                    self.client_socket.close()
                    return
                byte_stream += packet
            # Deserialize the data
            data = pickle.loads(byte_stream)

            # Separate images and transform
            color_image_msg = data.get("color_image")
            query = data.get("query")
            self.get_logger().info(f"Received Query: {query}")
        except ConnectionResetError as e:
            self.get_logger().error(f"Connection reset: {e}")
            self.cleanup_connection()
        except Exception as e:
            self.get_logger().error(f"Error receiving data: {e}")
            self.cleanup_connection()
        return color_image_msg, query
    
    def process_image(self, color_image_msg, query):
        TEXT_PROMPT = f"{query} ."
        image = color_image_msg.data
        image = np.frombuffer(image, dtype=np.uint8).reshape(self.image_height, self.image_width, 3)
        print(type(image))
        image = Image.fromarray(image)
        print(type(image))
        image_source, image = load_image_wo_PIL(image)
        boxes, logits, phrases = predict(
            model=self.gd_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD
        )
        xy = []
        boxes = boxes.cpu().detach().numpy().tolist()
        for cx, cy, w, h in boxes:
            xy.append((int(cx*self.image_width), int(cy*self.image_height)))
        print(xy)
        print(phrases)
        print(logits)
        return xy, phrases, logits
    
    def send_response(self, xy, phrases, logits):
        response = {
            "xy": xy,
            "phrases": phrases,
            "logits": logits.cpu().detach().numpy().tolist()
        }
        try:
            serialized_response = pickle.dumps(response)
            size = len(serialized_response)
            size_data = size.to_bytes(4, 'big')
            self.client_socket.sendall(size_data + serialized_response)
        except Exception as e:
            self.get_logger().error(f"Error sending response: {e}")
            self.cleanup_connection()

    def cleanup_connection(self):
        if self.is_connected:
            self.is_connected = False
            self.timer.cancel()  # Stop the timer
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except Exception as e:
                self.get_logger().warning(f"Error closing socket: {e}")
    def destroy_node(self):
        if self.is_connected:
            self.cleanup_connection()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GDinoQueryClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
