import socket
import sys
import threading


def opfileread():
    Fileman = open('YOLO_OUTPUT', 'r')  # opening yolo_output with a read command
    out = Fileman.read()  # reading yolo_output
    z = out.split("'")  # splitting the string into a list based on " ' " separator
    x = 1  # initiating a count variable to iterate the list
    while x < len(z):  # a while loop to iterate through the list and print odd values
        print(z[x])
        x += 2


def socket_client_detection():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 10000)
    sock.bind(server_address)
    try:
        # Receive response from the server
        data, server = sock.recvfrom(4096)
        detection_data = str(data)
        z = detection_data.split("//")  # splitting the string into a list based on " ' " separator
        detection = z[0]
        point_cloud = z[1]
        point_cloud_split = point_cloud.split(",")
        print(detection, "\n", point_cloud)
        x = 0  # initiating a count variable to iterate the list
        while x < len(point_cloud_split):  # a while loop to iterate through the list and print odd values
            print(point_cloud_split[x])
            x += 1
    finally:
        sock.close()

def socket_client_detected_memory():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 20000)
    sock.bind(server_address)
    try:
        # Receive response from the server
        data, server = sock.recvfrom(4096)
        detection_data = str(data)
        z = detection_data.split("]")  # splitting the string into a list based on " ' " separator
        x = 0  # initiating a count variable to iterate the list
        while x < len(detection_data):  # a while loop to iterate through the list and print odd values
            print(detection_data[x])
            x += 1
    finally:
        sock.close()

while True:  # a while loop to iterate through different user inputs
    txt = input("Would you like to see?")  # Taking user input
    if txt == "y":
        socket_client_detection()

    if txt == "yy":
        socket_client_detected_memory()
