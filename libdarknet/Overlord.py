def opfileread():
    Fileman = open('YOLO_OUTPUT', 'r')  # opening yolo_output with a read command
    out = Fileman.read()                # reading yolo_output
    z = out.split("'")                  # splitting the string into a list based on " ' " separator
    x = 1                               # initiating a count variable to iterate the list
    while x < len(z):                   # a while loop to iterate through the list and print odd values
        print(z[x])
        x += 2


while True:                                     #a while loop to iterate through different user inputs
    txt = input("Would you like to see?")       #Taking user input
    if txt == "y":
        opfileread()
