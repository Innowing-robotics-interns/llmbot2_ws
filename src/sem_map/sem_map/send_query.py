import socket
import struct
import numpy as np

class SocketSender:
    def __init__(self, host='fyp', port=6000):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send(self, query, angle=3.1415/12):
        self.socket.sendall(struct.pack("<f", angle))
        self.socket.sendall(query.encode())
        try:
            num_points = struct.unpack("<L", self.socket.recv(4))[0]
            point_list = []
            if num_points == 0:
                print("No points found.")
                return
            for i in range(num_points):
                point = struct.unpack("<3f", self.socket.recv(12))
                point_list.append(point)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return np.array(point_list)

    def __del__(self):
        self.socket.close()

if __name__ == "__main__":
    sender = SocketSender(host='fyp', port=6000)
    while True:
        try:
            angle = float(input("Enter the angle: "))
        except ValueError:
            print("Invalid angle.")
            continue
        query = input("Enter your query: ")
        print(sender.send(query, angle))
