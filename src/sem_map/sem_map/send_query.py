import socket

class SocketSender():
    def __init__(self, host='localhost', port=6000):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
    def send(self, query):
        self.s.sendall(query.encode())
        response = self.s.recv(1024).decode()
        return response

if __name__ == "__main__":
    sender = SocketSender(host='fyp', port=6000)
    while True:
        query = input("Enter your query: ")
        print(sender.send(query))
