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


def opfileread_full():
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 10000)
    sock.bind(server_address)
    try:
        # Receive response from the server
        data, server = sock.recvfrom(4096)
        detection_data = str(data)
        zx = detection_data.split("//")
        z = zx[0].split("'")  # splitting the string into a list based on " ' " separator
        x = 1  # initiating a count variable to iterate the list
        while x < len(z):  # a while loop to iterate through the list and print odd values
            print(z[x])
            x += 2
        z = zx[1].split(",")  # splitting the string into a list based on " ' " separator
        x = 0  # initiating a count variable to iterate the list
        while x < len(z):  # a while loop to iterate through the list and print odd values
            print(z[x])
            x += 1
    finally:
        sock.close()


while True:  # a while loop to iterate through different user inputs
    txt = input("Would you like to see?")  # Taking user input
    if txt == "y":
        opfileread()

    if txt == "yy":
        opfileread_full()
