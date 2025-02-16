import socket

def send_query(query, host='localhost', port=5555):
    '''
    Sends a text query to the TextQueryReceiver and prints the response.

    Args:
        query (str): The text query to send.
        host (str): The hostname of the TextQueryReceiver server.
        port (int): The port number of the TextQueryReceiver server.
    '''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(query.encode())
        response = s.recv(1024).decode()
        print(f"Response: {response}")

if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        send_query(query, host='fyp', port=5555)
