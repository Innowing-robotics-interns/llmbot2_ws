import socket
import struct
import numpy as np
from sklearn.cluster import DBSCAN

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
                return num_points, None
            for i in range(num_points):
                point = struct.unpack("<3f", self.socket.recv(12))
                point_list.append(point)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return num_points, np.array(point_list)

    def looking_for_some_points(self, query, n_lower=10, n_upper=40, step=0.2, loop_max=20):
        prev = 0 # 0 means no previous angle, 1 means previous is to high, -1 means previous is too low
        for _ in range(loop_max):
            num_points, points = self.send(query, angle)
            if num_points < n_lower:
                if prev != 1:
                    angle += step 
                else:
                    step /= 2
                    angle += step 
                prev = -1
            elif num_points > n_upper:
                if prev != -1:
                    angle -= step 
                else:
                    step /= 2
                    angle -= step 
                prev = 1
            else:
                return angle, num_points, points
        return angle, num_points, points
    
    def dbscan(self, points, eps=0.5, min_samples=5):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        clustering.labels_
        label = np.bincount(clustering.labels_).argmax()
        return np.mean(points[clustering.labels_ == label])

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
        angle, num_points, points = sender.looking_for_some_points(query)
        print(f"Angle: {angle}, Number of points: {num_points}, Points: {points}")
        print(f"Mean of common cluster: {sender.dbscan(points)}")
