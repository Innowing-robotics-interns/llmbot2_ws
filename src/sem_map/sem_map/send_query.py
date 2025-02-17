import socket
import struct
import numpy as np

class SocketSender:
    def __init__(self, host='fyp', port=6000):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send(self, query):
        self.socket.sendall(query.encode())
        data = self.socket.recv(12)
        return np.array(struct.unpack("<3f", data))

    def __del__(self):
        self.socket.close()

if __name__ == "__main__":
    sender = SocketSender(host='fyp', port=6000)
    while True:
        query = input("Enter your query: ")
        print(sender.send(query))
