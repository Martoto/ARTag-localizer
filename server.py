import socket
import threading
import queue

HOSTPORT = ("localhost", 5070)

def send_pose(pose):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = (str(pose) + "\n").encode()
    #print("sending data: ", data)
    sock.sendto(data, HOSTPORT)

def run_server():
    return None

