import socketserver
import threading
import queue

queues = []
queues_mtx = threading.Lock();

class MyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(" print('client connected from {} to {}".format(self.client_address[0], threading.current_thread()))
        queueel = queue.Queue()
        with queues_mtx:
            queues.append(queueel)
        while True:
            data = ((str(queueel.get())) + "\n").encode()
            print("sending data: ", data)
            self.request.sendall(data)

def send_pose(pose):
    with queues_mtx:
        for queueel in queues:
            try:
                queueel.put(pose, False)
            except Exception as e:
                print(e)

def run_server():
    HOST, PORT = "0.0.0.0", 5070

    server = socketserver.ThreadingTCPServer((HOST, PORT), MyHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)
    return server