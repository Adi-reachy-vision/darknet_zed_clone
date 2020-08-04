from sys import argv
import darknet_zed
import multiprocessing


def images():
    darknet_zed.main(argv)      #calls upon the main function of darknet_zed


def printop(detections, count):    #prints output for the detected objects (once)
    i = 0
    for _ in detections:
        detect: list = detections[i]  # a list of detected objects from darknet_zed as a string holding label, positional and class info

        for __ in detect:
            det = detect[0]             # a new variable to hold just the label information

        i += 1                          #an iterative value to move the cursor to the
        print(det)                      # next detected obejct and it's information in the detect list variable


processes = []                          # a list for holding process information for parallel processing

z = multiprocessing.Process(target=images)          #intention to start darknet_zed file to ensure image processing runs in the background irrespective of printing values
z.start()                                           #starting darknet_zed
processes.append(z)                                 #attaching darknet_zed to a list of processes that can simultaneously run

zx = multiprocessing.Process(target=printop)        #printing list of detected objects
zx.start()
processes.append(zx)

"""zx1 = multiprocessing.Process(target=print("Hello"))
zx1.start()
processes.append(zx1)
"""
