import socket

def socket_server_control(camera):
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)  # SOCK_DGRAM stands for datagram which is UDP's method of data transmission
    server_address = ('localhost', 30000)  # the local host address and port number. If you run into an error try changing,
    # as the port might be occupied by some other process
    message = camera # the detected data separated by a separator '//' from the point
    # cloud location data of x, y, z value of the bounding box for
    # location estimation of the object in the environment

    try:
        # Send data in a try block to prevent the presence of an error
        sent = sock.sendto(message.encode(), server_address)
    finally:
        sock.close()  # once the data has been sent, the socket can be closed

txt = input("Would you like to see?")  # Taking user input
while True:  # a while loop to iterate through different user inputs
    if txt == "y":
        camera = "live"
        socket_server_control(camera)

    elif txt == "n":
        camera = "black"
        socket_server_control(camera)