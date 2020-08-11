import socket

def opfileread():
    Fileman = open('YOLO_OUTPUT', 'r')  # opening yolo_output with a read command
    out = Fileman.read()  # reading yolo_output
    z = out.split("'")  # splitting the string into a list based on " ' " separator
    x = 1  # initiating a count variable to iterate the list
    while x < len(z):  # a while loop to iterate through the list and print odd values
        print(z[x])
        x += 2


def op_socket(x):
    msgfromclient = "Hello UDP Server"
    bytestoSend = str.encode(msgfromclient)
    serverAddressPort = ("127.0.0.1",12345)
    bufferSize = 1024

    Client = socket.socket(family=socket.AF_INET, type= socket.SOCK_DGRAM)
    Client.sendto(bytestoSend, serverAddressPort)
    msgfromserver = Client.recvfrom(bufferSize)

    msg = "Message from server {} ".format(msgfromserver)
    print(msg)


while True:  # a while loop to iterate through different user inputs
    txt = input("Would you like to see?")  # Taking user input
    if txt == "y":
        opfileread()

    if txt == "yy":
        x = "Show me"
        opsockets(x)
