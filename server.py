import socket
import threading
import queue

HOSTPORTS = [("localhost", 5070), ("192.168.101.3", 5070)]


def send_pose(pose):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = (str(pose) + "\n").encode()
    #print("sending data: ", data)
    for hostport in HOSTPORTS:
        sock.sendto(data, hostport)

def run_server():
    return None

