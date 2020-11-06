import io
import math
import random
import socket
import statistics
import time

#from skimage.measure import compare_ssim
import numpy as np
import cv2
from PIL import Image, ImageChops


def socket_server_status(detections, point_cloud_data):
    '''The Detection data is transmitted via a socket connection to OverLord.py via TCP or UDP protocol,
    to be viewed by the primary user of the program. The import function was compromised between darknet_zed & Overlord,
    meaning the text - input command from Overlord was initiated every time darknet_zed ran and the sequential structure
    execution in python meant that the detection algorithm would only start once the input was given to print detected function.
    The socket method kept both code independent from each other, allowing us to detect the objects and at the same time
    see it in a seperate window. Parallel Processing libraries such as threading and multiprocessing were tried but this
    method gave a suitable outcome.

    UDP was chosen over TCP due to it's independence from the necessity of a listening client, thus irrespective of
    whether or not, there was a listening client the server keeps on transmitting data and the client only prints out
     the real-time data once the user needs it otherwise, the data can get overwritten.'''

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)  # SOCK_DGRAM stands for datagram which is UDP's method of data transmission
    server_address = ('localhost',
                      10000)  # the local host address and port number. If you run into an error try changing,
                                # as the port might be occupied by some other process
    message = detections + "//" + point_cloud_data  # the detected data separated by a separator '//' from the point
                                                    # cloud location data of x, y, z value of the bounding box for
                                                    #location estimation of the object in the environment

    try:
        # Send data in a try block to prevent the presence of an error
        sent = sock.sendto(message.encode(), server_address)
    finally:
        sock.close()  # once the data has been sent, the socket can be closed


def socket_server_detected_objects(detections):
    '''The detected objects list is transmitted via a socket connection to OverLord.py via TCP or UDP protocol,
    to be viewed by the primary user of the program. The import function was compromised between darknet_zed & Overlord,
    meaning the text - input command from Overlord was initiated every time darknet_zed ran and the sequential structure
    execution in python meant that the detection algorithm would only start once the input was given to print detected function.
    The socket method kept both code independent from ech other, allowing us to detect the objects and at the same time
    see it in a seperate window. Parallel Processing libraries such as threading and multiprocessing were tried but this
    method gave a suitable outcome.

    UDP was chosen over TCP due to it's independence from the necessity of a listning client, thus irrespective of
    whther or not, there was a listening client the server keeps on transmitting data and the client only prints out
     the real-time data once the user needs it otherwise, the data can get overwritten'''

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)  # SOCK_DGRAM stands for datagram which is UDP's method of data transmission
    server_address = ('localhost',
                      20000)  # the local host address and port number. If you run into an error try changing, as the port might be occupied by some other process
    message = detections  # the detected data separated by a separator '//' from the point cloud location data of x, y, z value of the bounding box for location estimation of the object in the environment

    try:
        # Send data in a try block to prevent the presence of an error
        sent = sock.sendto(message.encode(), server_address)
    finally:
        sock.close()  # once the data has been sent, the socket can be closed


def opfileprint(detection):
    '''An alternate method of displaying data to the socket method. They are both similar in their mechanism, with
    this method printing the detected data in a file called "YOLO_OUTPUT.txt". The easier to implement method out of the 2
    is kept as an alternate if needed. The Recieveing end needs to just read the file and print out the data. The recieving end
    is present in Overlord.py'''

    Fileman = open('YOLO_OUTPUT', 'w')  # creating and opening file in the write configuration
    for i in detection:
        Fileman.write(i)  # writing the detection into the file
        # Fileman.write('\n')
    Fileman.close()


def image_segmentation_depth(y_extent, x_extent, y_coord, x_coord, depth, image, median_max, avg_median, grasp_y_delay,
                             shallow_cluster_y):
    '''Perform image segmentation based on depth information received by ZED camera's point cloud data.
    The depth of the pixels in the bounding box are added to a flattened array and after averaging their depth,
    the mean depth of the object is received and the depth of the pixels is once again compared to analyse if the
     target pixel is greater or lesser than the mean depth of the object. If it is lesser than the mean depth, the pixel
     is filled with red colour.

     Grasping points - an additional method which may be used in conjuncture to the image segmentation mask is to identify,
     the ideal grasping points of the object with respect to the object geometry. The present primary method to acheive,
     grasping points on the object is done by storing all the y-axis pixels which are lesser than the median depth and
     then finding the median values from the pixel to locate the center of the object in terms of y-axis values. Once,
     the central y-axis coordinates have been narrowed down, it's respective x-axis values are at the edge of the
     segmentation mask are highlighted using a green circle marker.'''
    median_depth = []  # initialising an array to store the depth of all pixels in the bounding box
    grasp_array_y = []
    grasp_array_x = []
    min_object_depth = []
    shallow_grasp_x = []
    height = 0
    for i in range(y_extent):  # element by element multiplication of the height of the bounding box
        y_val = y_coord + (i - 1)
        height += 1
        for j in range(x_extent):  # element by element multiplication of the width of the bounding box
            x_val = x_coord + (j - 1)
            # print(x_val,j)
            try:
                calc_depth = depth[y_val, x_val]
                calc_depth = math.sqrt((calc_depth[0] * calc_depth[0]) + (calc_depth[1] * calc_depth[1]) + (
                        calc_depth[2] * calc_depth[2])) * 10  # calculating euclidian distance of a pixel
                median_depth.append(calc_depth)  # storing the scaled up depth in a flattened array for avergaing

            except:
                pass

    shallow_depth = min(median_depth)
    median = float(round(np.median(median_depth), 2))  # calculating median depth using numpy median method
    median_large = median_average(median, median_max, avg_median)  # an averaging function to stablise the depth over
    # multiple frames by finding median depth over multiple
    # iterations of stored average distances of the object in the bounding box

    try:
        y_grasp = grasp_y_delay[len(grasp_y_delay) - 1]  # the y-axis coordinates which happen to be lesser than the
        # median depth of the object in the previous frame
        y_shallow = shallow_cluster_y[len(shallow_cluster_y) - 1]
    except:
        y_grasp = y_coord  # in case of an error return the y_coord value
        y_shallow = y_coord

    for i in range(y_extent):  # element by element multiplication of the height of the bounding box
        y_val = y_coord + (i - 1)
        for j in range(x_extent):  # element by element multiplication of the width of the bounding box
            x_val = x_coord + (j - 1)
            # print(x_val,j)
            try:  # encapsulating the depth calculation in a try - catch block to prevent value errors
                calc_depth = depth[
                    y_val, x_val]  # storing x,y,z values from individual pixels for comparing with the threshold value
                calc_depth = math.sqrt(
                    (calc_depth[0] * calc_depth[0]) + (calc_depth[1] * calc_depth[1]) + (
                            calc_depth[2] * calc_depth[
                        2])) * 10  # calculating euclidian distance of a pixel

                if calc_depth < median_large:  # comparing the pixel distance from the threshold
                    # print("True")
                    if calc_depth == shallow_depth:
                        shallow_cluster_y.append(y_val)
                    min_object_depth.append(calc_depth)
                    if y_val not in grasp_array_y:
                        grasp_array_y.append(y_val)  # store the y-axis coordinate which is lesser than the depth

                    if y_val == y_grasp:  # change the colour of the pixels to highlight the ideal location for grasping the object
                        # image[y_val, x_val] = (0, 255, 0, 0)
                        grasp_array_x.append(x_val)  # store the x-axis coordinate which is lesser than the depth
                    if y_val == y_shallow:
                        shallow_grasp_x.append(x_val)
                    else:
                        image[y_val, x_val] = (0, 0, 255, 0)  # highlighting the pixels whose depth is greater than
                    # the median depth of the bounding box with a red colour
            except:
                pass
    y_grasp = grasp_array_y[(len(grasp_array_y) - 1)]
    # print(len(grasp_array_x))
    if len(grasp_array_x) > 0:
        x_pt1 = int(grasp_array_x[0])  # the 1st x-axis coordinates which happen to be lesser than the median depth of
        # the object in the previous frame
        x_pt2 = int(grasp_array_x[len(grasp_array_x) - 1])  # the 2nd x-axis coordinates which happen to be lesser than
        # the median depth of the object in the previous frame
        height = statistics.median(height)
        #height = height / 2  # height of the bounding box
        y_grasp = int(y_grasp - height)  # the ideal point to grab the object in the y-axis
        cv2.circle(image, (x_pt1, y_grasp), 20, (0, 255, 0, 0), -1)  # forming a circle around the first point
        cv2.circle(image, (x_pt2, y_grasp), 20, (0, 255, 0, 0), -1)  # forming a circle around the second point

    if len(shallow_grasp_x) > 0:
        x_pt1 = int(shallow_grasp_x[0])  # the 1st x-axis coordinates which happen to be lesser than the median depth of
        # the object in the previous frame
        x_pt2 = int(
            shallow_grasp_x[len(shallow_grasp_x) - 1])  # the 2nd x-axis coordinates which happen to be lesser than
        # the median depth of the object in the previous frame
        height = height / 2  # height of the bounding box
        y_grasp = int(y_grasp - height)  # the ideal point to grab the object in the y-axis
        cv2.circle(image, (x_pt1, y_shallow), 20, (0, 0, 255, 0), -1)  # forming a circle around the first point
        cv2.circle(image, (x_pt2, y_shallow), 20, (0, 0, 255, 0), -1)  # forming a circle around the second point

    grasp_y_delay.append(
        y_grasp)  # appending the value of y_grasp into the array to ensure it can be recalled in the next frame
    grasp_method(median_large, min_object_depth)
    return image  # returning a modified image which has depth based segmentation mask
    # marked on the image in the area of the bounding box


def median_average(median, median_max, median_array):
    '''Averaging function to stablise the depth obtained from 'image_segmentation_depth'. The idea behind this was
    behind the assumption that depth estimate of one frame was accurate but the depth over multiple frames would allow
    a better understanding of the depth of the object which provides the program with it's segmentation mask.'''
    median_max.append(median)
    # print(len(median_max))
    while len(median_max) % 10 == 0:  # averaging function over 10 frames
        median_large = np.mean(median_max)  # provide a mean value for the median values stored in the median_max array
        median_max.clear()  # once the averaging function is performed, the median_max array is cleared.
        median_large = float(round(median_large, 3))  # rounding the median average to 3 decimal points
        median_array.append(median_large)  # storing the average avergae median in a new array to ensure,
        # the value is stored over multiple frames and doesnt get
        # reinitialised every time, the function is called
        # print("Values after comparison: ", median_large, median, len(median_max))
        if abs(median_large - median) > 2.00:  # if the median average has moved more than 2 metres from the
            # median of the present frame than it means, the object has moved
            # the avergaing function over multiple feames has to be restarted.
            median_array.clear()
        break

    try:
        return median_array[len(median_array) - 1]  # return the average function if present
    except:
        return median  # if the average values aren't present return the median of the
        # past frame which was being analysed


def grasp_method(median_large, min_object_depth):
    '''The grasping method which relies on object geometry gives us information in the width of the object between the extremities
    of the grasping points. If they are greater than the width of the curved hand, then it will be uncomfortable for
    grasping the object in the present pose and thus, will have to be changed, to be more accessible by the hand for
    a better grip.'''
    min_object = round(min(min_object_depth), 3)
    final = median_large - min_object
    if final < 0.7:
        method = "Grasp"
    else:
        method = "Pinch"

    #print(method)


def image_segmentation_colour(image, mask, y_coord, y_extent, x_coord, x_extent, thresh, bounds):
    '''An alternate image segmentation method which provides a black and white segmentaiton mask based on the presence
    of colour pixels in the designated bounding box area.The center pixel as designated by yolo is marked as the pixel
    of interest whose colour value is taken as the colour of interest and after adding a threshold as padding to keep it within
    a range, the presence of colour in those BGR values is marked in balck & white as a masking method based on colour.

    Depth based segmentation is the primary method but colour based segmentation is also useful in some situations,
    personally it works phenomenally well, in darker colour regions.'''
    cropped_image = image[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)] #cropping the image to the size of the bounding box to reduce confusion by the inRange function in detecting the dominant colour
    try:
        color = image[bounds[0], bounds[1]] #finding the colour value of the center pixel
    except:
        color = 0 #in case we can't find return a 0 value
    cropped_image = cv2.cvtColor(cropped_image,
                                 cv2.COLOR_BGRA2BGR)  # cropping the image to the size of the bounding box
    blue = color[0]  # storing blue value in bgr
    green = color[1]  # storing green value in bgr
    red = color[2]  # storing red value in bgr
    try:  # encapsulating the function in a try block to prevent the program from crashing if the right pixel is not found
        lower = np.array([(int(blue) - thresh), (int(green) - thresh), (int(red) - thresh)],
                         dtype="uint8")  # setting the lower barrier of the inRange function
        upper = np.array([(int(blue) + thresh), (int(green) + thresh), (int(red) + thresh)],
                         dtype="uint8")  # setting the upper barrier of the inRange function
        masked = cv2.inRange(cropped_image, lower,
                             upper)  # determining the presence of the colour in the bounding box and returning it as a 8-bit integer mask

        mask[y_coord:(y_coord + y_extent),
        x_coord:(x_coord + x_extent)] = masked  # pasting the bounding box mask into the bigger mask image
    except:
        mask = mask  # returning the original mask (a blank black-white image) in case the try block fails to run
    return mask


def color_test(image):
    '''The color recognition of objects needed to be refined using custom determined values. The initial testing method involved passing
    boundaries for the colours in the RGB matrix from images on the internet which had a different colour composition than
    from the ones, the camera was seeing. Thus, it had to be altered using a testing method like this function, where the colour
    values of BGRA pixel was manually entered and it was determined to be the best range for the colour composition.'''
    # to test the robustness of the color recognition method and can be deleted later if needed
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # the Alpha Channel is stripped from the initial image
    lower = np.array([1, 90, 200], dtype="uint8")  # the lower boundary for the inRange function is set
    upper = np.array([58, 155, 255], dtype="uint8")  # the upper boundary for the inRange function is set
    # find the colors within the specified boundaries and apply
    try:  # a try block to prevent the program from crashing if the method doesnt run smoothly
        mask = cv2.inRange(image, lower,
                           upper)  # returning a mask with the colour values highlighted as white pixels and blank as the ones that don't
    except:
        pass  # do nothing if the method doesn't work and continue executing the function.

    return mask


def get_color(image):
    '''The inRange functions return values based on  the hard ranges as set in the inRange function. Every colour has a certain BGR value, and
    giving those values to the inRange function helps us determine if the area in the bounding box has the colour. As there
    are multiple colours we are looking the 8-bit mask which takes the output is summed up and the highest value is sent as the
    dominannt colour in the bounding box area.'''

    img = cv2.cvtColor(image,
                       cv2.COLOR_BGRA2BGR)  # converting to a BGR image as the inrange works in only a 3 channel image not a 4 channel

    boundaries = [([17, 15, 100], [50, 56, 234]),  # red
                  ([36, 31, 4], [180, 88, 60]),  # blue
                  ([45, 76, 20], [85, 235, 180]),  # green
                  ([20, 150, 210], [90, 255, 255]),  # yellow
                  ([103, 86, 65], [145, 133, 128]),  # black
                  ([1, 1, 1], [45, 40, 40])]  # white
    count = 0
    for (lower, upper) in boundaries:
        try:
            count += 1
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            # find the colors within the specified boundaries and apply
            mask = cv2.inRange(img, lower, upper)  # the inRange function which returns the mask value
            if count == 1:
                red_sum = int(np.sum(mask))  # separating the values into separate masks based on color
            elif count == 2:  # adding the mask values and storing them into variables, depending on the iteration of the loop
                blue_sum = int(np.sum(mask))
            elif count == 3:
                green_sum = int(np.sum(mask))
            elif count == 4:
                yellow_sum = int(np.sum(mask))
            elif count == 5:
                white_sum = int(np.sum(mask))
            elif count == 6:
                black_sum = int(np.sum(mask))

        except:
            pass

    # placing the sum in an array to be sorted

    color_arrays = [blue_sum, green_sum, red_sum, yellow_sum, white_sum, black_sum]
    color_arrays = np.sort(color_arrays)  # sorting an array to present the most dominant colour as the output
    print_color = color_arrays[len(color_arrays) - 1]  # the values are then sorted in an array ascending order and
    # returning the last value (the largest) as the most dominant
    # colour in the bounding box
    if print_color == blue_sum:  # determing the colour, the highest value is equal to and returning that value in a string
        object_color = "Blue"
    elif print_color == green_sum:
        object_color = "Green"
    elif print_color == red_sum:
        object_color = "Red"
    elif print_color == yellow_sum:
        object_color = "Yellow"
    elif print_color == white_sum:
        object_color = "White"
    elif print_color == black_sum:
        object_color = "Black"

    return object_color  # returning the highest mask value as the colour


def get_positional_data(camera_pose, py_translation):
    ''' The positional tracking and rotational data generated by the IMU, gyroscope sensors in the camera are returned
    here as tx, ty, tz.
    The positional tracking information relies on a variety of factors such as depth data (Ultra and Quality modes give
    more reliable information but affect fps, however, Performance mode is faster but the positional information isn't
    reliable.)

    The positional tracking information is critical to updating the positional information of detected objects as it
    moves out of the frame.
    The rotational information can be used in pose estimation. '''

    rotation = camera_pose.get_rotation_vector()  # The rotation information from the gyroscopic sensors
    rx = round(rotation[0], 2)  # The rotational information of the x-axis
    ry = round(rotation[1], 2)  # The rotational information of the y-axis
    rz = round(rotation[2], 2)  # The rotational information of the z-axis

    translation = camera_pose.get_translation(py_translation)  # The translational information from the IMU sensors
    tx = round(translation.get()[0], 3)  # The translational information of the x-axis
    ty = round(translation.get()[1], 3)  # The translational information of the y-axis
    tz = round(translation.get()[2], 3)  # The translational information of the z-axis

    return tx, ty, tz, rx, ry, rz


def positional_buffer_CAMERAframe(camera_pose, py_translation, positional_buffer_array, rotational_buffer_array, rotation_timer):
    tx, ty, tz, rx, ry, rz = get_positional_data(camera_pose, py_translation)
    '''The translational and rotational data given by the camera may have some errors due to the speed of the movement 
    or via other means (repeated values before the camera frame values normalise to 0,0,0). 
     This function stores values between 2 consecutive zero values and releases them once, the CAMERA frame stablises.
     This allows, the values to be updated only once, rather than a repeated update.'''
    x_sum = y_sum = z_sum = 0.0
    rx_sum = ry_sum = rz_sum = 0.0
    time_start = None
    if tx != 0.0 or ty != 0.0 or tz != 0.0:  # A trigger to start storing values for translation to be processed later
        pos = [tx, ty, tz]
        if pos not in positional_buffer_array:  # storing unique values in an array
            positional_buffer_array.append(pos)

    if rx != 0.0 or ry != 0.0 or rz != 0.0:  # A trigger to start storing values for rotation to be processed later
        rot = [rx, ry, rz]
        if rot not in rotational_buffer_array:  # storing unique values in an array
            rotational_buffer_array.append(rot)

    if len(positional_buffer_array) > 0:  # if the translational array is not None
        print("Trans True")
        if abs(tx) == 0.0 and abs(ty) == 0.0 and abs(tz) == 0.0:  # if the values have stablised to reach a zero value
            # print("Condition zero trans")
            # print("greater : ", positional_buffer_array)
            for x, y, z in positional_buffer_array:  # take the individual x,y,z values and in the list and add them
                x_sum += x
                y_sum += y
                z_sum += z
            positional_buffer_array.clear()
            print("mod : ", x_sum, y_sum, z_sum)

    if len(rotational_buffer_array) > 0:  # if the rotational array is not None
        print("Rot True")
        if abs(rx) == 0.0 and abs(ry) == 0.0 and abs(rz) == 0.0:  # if the values have stablised to reach a zero value
            print("Condition zero rot")
            # print("greater : ", rotational_buffer_array)
            for x, y, z in rotational_buffer_array:  # take the individual x,y,z values and in the list and add them
                rx_sum += x
                ry_sum += y
                rz_sum += z
            rotational_buffer_array.clear()
            print("rot mod : ", rx_sum, ry_sum, rz_sum)
            if 2.0 > ry_sum >= 1.4 or -1.4 > ry_sum > - 2.0:
                positional_buffer_array.clear()

    tx, ty, tz = x_sum, y_sum, z_sum  # the x,y,z values from the translation array are transmitted only once, otherwise it is 0
    rx, ry, rz = rx_sum, ry_sum, rz_sum  # the x,y,z values from the rotation array are transmitted only once, otherwise it is 0
    return tx, ty, tz, rx, ry, rz  # returning translation and rotation array

def rotational_update(detected_objects, rx, ry, rz):
    '''When the camera rotates, its coordinate system changes and this update needs to be reflected in the memory component
    of the deteccted objects as well. The new coordinate system has a new x,y,z value where the previous coordinate system's
    values now have to be updated with the new coordinate system's values.'''
    rotation = False
    detected_objects_rot = []
    i = 0
    length = len(detected_objects)
    if -1.4 > ry > - 2.0:  # generally a y- axis value of -1.5 means it has turned to the left on a steady plane (y - axis elevation hasn't changed)
        rotation = True
        detected_objects_rot.clear()
        for detectedx in detected_objects:  # iterating through the detected objects list contents
            # print(detected_objects, i)
            id_new = detectedx[0]  # storing the id of the list object that is about to be changed
            label_new = detectedx[1]  # storing the list of the list object that is about to be changed
            xvalue = round(detectedx[2], 3)  # storing the x - axis of the list object that is about to be changed
            yvalue = round(detectedx[3], 3)  # storing the y-axis of the list object that is about to be changed
            zvalue = round(detectedx[4], 3)  # storing the z-axis of the list object that is about to be changed
            if detectedx[6] == "no rotation":
                detected_o = [id_new, label_new, -(zvalue), (yvalue),
                              (xvalue), "change", "rotated left"]
            elif detectedx[6] == "rotated right":
                detected_o = [id_new, label_new, - (zvalue), (yvalue),
                              (xvalue), "change", "no rotation"]
            elif detectedx[6] == "rotated left":
                detected_o = [id_new, label_new, -(zvalue), (yvalue),
                              (xvalue), "change", "rotated left"]
            detected_objects_rot.append(
                detected_o)  # the updated entry of detected object with new positional information
            i += 1
        # detected_objects.clear()

    elif 2.0 > ry > 1.2:  # generally a y- axis value of 1.5 means it has turned to the right on a steady plane (y - axis elevation hasn't changed)
        rotation = True
        detected_objects_rot.clear()
        for detectedx in detected_objects:
            # print(detectedx, i)
            id_new = detectedx[0]  # storing the id of the list object that is about to be changed
            label_new = detectedx[1]  # storing the list of the list object that is about to be changed
            xvalue = round(detectedx[2], 3)  # storing the x - axis of the list object that is about to be changed
            yvalue = round(detectedx[3], 3)  # storing the y-axis of the list object that is about to be changed
            zvalue = round(detectedx[4], 3)  # storing the z-axis of the list object that is about to be changed
            if detectedx[6] == "rotated left":
                detected_o = [id_new, label_new, (zvalue), (yvalue),
                              -(xvalue), "change", "no rotation"]
            elif detectedx[6] == "no rotation":
                detected_o = [id_new, label_new, (zvalue), (yvalue),
                              -(xvalue), "change", "rotated right"]
            elif detectedx[6] == "rotated right":
                detected_o = [id_new, label_new, (zvalue), (yvalue),
                              -(xvalue), "change", "inverted"]
            detected_objects_rot.append(
                detected_o)  # the updated entry of detected object with new positional information
            i += 1
        detected_objects.clear()

    elif 4.0 > ry > 2.1 or -2.4 > ry > - 4.0: # generally a y- axis value of 3.0 in either direction means it has turned by a full circle on a steady plane (y - axis elevation hasn't changed)
        rotation = True
        detected_objects_rot.clear()
        for detectedx in detected_objects:
            # print(detectedx, i)
            id_new = detectedx[0]  # storing the id of the list object that is about to be changed
            label_new = detectedx[1]  # storing the list of the list object that is about to be changed
            xvalue = round(detectedx[2], 3)  # storing the x - axis of the list object that is about to be changed
            yvalue = round(detectedx[3], 3)  # storing the y-axis of the list object that is about to be changed
            zvalue = round(detectedx[4], 3)  # storing the z-axis of the list object that is about to be changed
            detected_o = [id_new, label_new, -(xvalue), (yvalue), -(zvalue), "change", "inverted"]
            detected_objects_rot.append(
                detected_o)  # the updated entry of detected object with new positional information
            i += 1
        detected_objects.clear()

    if rotation == False:
        return rotation, detected_objects
    elif rotation == True:
        return rotation, detected_objects_rot


def positional_update(detected_objects, tx, ty, tz):
    '''Adding the translational values to the objects in the detected memory to update their location.'''
    length = len(detected_objects)
    i = 0
    detected_objects_change = []
    while i < length:
        detectedx = detected_objects[i - 1]
        # print(detectedx, i)
        id_new = detectedx[0]
        label_new = detectedx[1]
        xvalue = round(detectedx[2], 3)
        yvalue = round(detectedx[3], 3)
        zvalue = round(detectedx[4], 3)
        detected_o = [id_new, label_new, (xvalue - tx), (yvalue - ty),
                      (zvalue - tz), "change", "no rotation"]
        detected_objects_change.append(
            detected_o)  # the updated entry of detected object with new positional information
        i += 1
    detected_objects.clear()
    return detected_objects_change


def positional_update_left(detected_objects, tx, ty, tz):
    '''If the camera has been rotated to the left, it's positional tracking method now has to be changed from it's
    previously used updating method.'''
    length = len(detected_objects)
    i = 0
    detected_objects_rot = []
    while i < length:
        # print(detected_objects, i)
        detectedx = detected_objects[i]
        id_new = detectedx[0]
        label_new = detectedx[1]
        xvalue = round(detectedx[2], 3)
        yvalue = round(detectedx[3], 3)
        zvalue = round(detectedx[4], 3)
        detected_o = [id_new, label_new, (xvalue + tx), (yvalue - ty),
                      (zvalue + tz), "change", "rotated left"]
        detected_objects_rot.append(
            detected_o)  # the updated entry of detected object with new positional information
        i += 1

    return detected_objects_rot


def positional_update_right(detected_objects, tx, ty, tz):
    '''If the camera has been rotated to the right, it's positional tracking method now has to be changed from it's
        previously used updating method.'''
    length = len(detected_objects)
    i = 0
    detected_objects_rot = []
    while i < length:
        # print(detected_objects, i)
        detectedx = detected_objects[i]
        id_new = detectedx[0]
        label_new = detectedx[1]
        xvalue = round(detectedx[2], 3)
        yvalue = round(detectedx[3], 3)
        zvalue = round(detectedx[4], 3)
        detected_o = [id_new, label_new, (xvalue + tx), (yvalue - ty),
                      (zvalue + tz), "change", "rotated right"]
        detected_objects_rot.append(
            detected_o)  # the updated entry of detected object with new positional information
        i += 1

    return detected_objects_rot


def positional_update_inverted(detected_objects, tx, ty, tz):
    '''If the camera has been rotated by 180 degrees, it's positional tracking method now has to be changed from it's
            previously used updating method.'''
    length = len(detected_objects)
    i = 0
    detected_objects_change = []
    while i < length:
        detectedx = detected_objects[i - 1]
        # print(detectedx, i)
        id_new = detectedx[0]
        label_new = detectedx[1]
        xvalue = round(detectedx[2], 3)
        yvalue = round(detectedx[3], 3)
        zvalue = round(detectedx[4], 3)
        detected_o = [id_new, label_new, (xvalue - tx), (yvalue - ty),
                      (zvalue - tz), "change", "inverted"]
        detected_objects_change.append(
            detected_o)  # the updated entry of detected object with new positional information
        i += 1
    detected_objects.clear()
    return detected_objects_change


def get_detected_objects(detected_objects, label, x, y, z, camera_pose, py_translation, cropped_image, existing_labels,
                         positional_buffer_array, rotational_buffer_array, rotation_timer):
    ''' Detected Objects are stored in an array, with verification for uniqueness of the detection being performed by
    location data (it being atleast half a meter in any direction -x, y, z) and image similarity which are stored in
    the directory 'memory_images' with the unique id of the object being stored as the title of the image. The images
     are compared using the functions 'get_ssim' & 'similarity'. '''
    duplicate_detections = []  # an array to store the detection data when the class of object already exists in the list of detected objects
    tx, ty, tz, rx, ry, rz = positional_buffer_CAMERAframe(camera_pose, py_translation,
                                                           positional_buffer_array,
                                                           rotational_buffer_array, rotation_timer)  # transaltional data received from the function
    # tx, ty, tz = get_positional_data(camera_pose,py_translation)
    tx = round(tx, 3)
    ty = round(ty, 3)
    tz = round(tz, 3)
    x = round(x, 3)
    y = round(y, 3)
    z = round(z, 3)
    # print(tx, ty, tz)
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        rotation, detected_objects = rotational_update(detected_objects, rx, ry, rz)
        '''if rotation == False:
            if tx != 0.0 or ty != 0.0 or tz != 0.0:
                detected_objects = positional_update(detected_objects, tx, ty, tz)'''

    if tx != 0.0 or ty != 0.0 or tz != 0.0:
        detected_rot = detected_objects[1]
        if "no rotation" in detected_rot:
            detected_objects = positional_update(detected_objects, tx, ty, tz)
        elif "rotated left" in detected_rot:
            detected_objects = positional_update_left(detected_objects, tx, ty, tz)
        elif "rotated right" in detected_rot:
            detected_objects = positional_update_right(detected_objects, tx, ty, tz)
        elif "inverted" in detected_rot:
            detected_objects = positional_update_inverted(detected_objects, tx, ty, tz)

    if len(detected_objects) >= 0:  # if length of detected objects is greater than or equal to 0
        for detected in detected_objects:  # scrolling through all the entries of the detected objects list
            if detected[1] not in existing_labels:
                existing_labels.append(
                    detected[1])  # storing the labels of the detected objects in the existing_labels array
            if label != detected[1]:  # if condition when the label doesn't match the entry
                pass
            elif label == detected[
                1]:  # if condition when the label of the detected object is similar to label of the entry
                if abs(x - int(detected[2])) % 0.050 == 0 and abs(y - int(detected[3])) % 0.050 == 0 and abs(
                        y - int(detected[4])) % 0.050 == 0:
                    # if the label is same and it is within 5 cm of the previous instance of the detected object
                    pass
                elif abs(x - int(detected[2])) % 0.050 != 0 and abs(y - int(detected[3])) % 0.050 != 0 and abs(
                        z - int(detected[4])) % 0.050 != 0:
                    # if the label is same and it is not within 5 cm of the previous instance of the detected object
                    array_valueid = [label, round(x, 3), round(y, 3),
                                     round(z, 3)]  # creating a template to add into the array
                    if array_valueid not in duplicate_detections:  # if condition when the detected object is not in
                        # detected object list and is thus, stored in array
                        # which shall be verified later. It is done to ensure
                        # that it can be applied when multiple instances of the
                        # same object are present in the memory.

                        duplicate_detections.append(array_valueid)
            '''if abs(tx - float(detected[2])) % float(detected[2]) != 0 or abs(ty - float(detected[3])) % float(
                    detected[3]) != 0 or abs(tz - float(detected[4])) % float(detected[4]) != 0:'''
            # activate in case the camera is constantly moving. The method is implemented to
            # ensure that the list of detected objects updates the location of the
            # detected objects as the camera moves where the tx,ty,tz will change and their difference from the
            # pre-existing values will change leading to a discrepancy that will need to be corrected

    if label not in existing_labels:  # the entries into detected object is appended, with first instances of
        # the class is immediately added into the detected_objects list
        # print(existing_labels)
        id = random.randint(1, 1000000000)
        detected_o = [id, label, round(tx + x, 3), round(ty + y, 3),
                      round(tz + z, 3), "orig", "no rotation"]
        detected_objects.append(detected_o)
        # cv2.imwrite("memory_images/{}.jpg".format(id), cropped_image)  # the cropped image (bounding box) of the detected object is written into a file
    '''else:
        # if the class already exists, then this method verifies if the object has been
        # detected for the first time and if it has, it is appended into detected_objects.
        for entry in duplicate_detections:  #scrolling through the list of new occurences of detected objects in a pre-existing class from detected objects
            #print(duplicate_detections)
            for detected in detected_objects: #scrolling through the index of detected_objects
                exist_value = False     #a boolean function to judge if the same object exists in the list
                if detected[1] == entry[0]: #the labels match
                    if abs(abs(entry[1]) - abs(detected[2])) > 0.1 and abs(abs(entry[2]) - abs(detected[3])) > 0.1 and abs(
                            abs(entry[3]) - abs(detected[4])) > 0.1: #the distances don't match
                        exist_value = True
                        if image_compare_hist(cropped_image, detected[0]) == False: # the objects don't look similar
                            exist_value = True
            if exist_value is True:     #if the object is unique it is appended into the list by giving it a unique ID
                print("True")
                id = random.randint(1, 1000000000)
                detected_o = [id, entry[0], round((entry[1] - tx), 3), round((entry[2] - ty), 3),
                              round((entry[3] - tz), 3), "new", "no rotation"]
                detected_objects.append(detected_o)
                present_image = cv2.imread("present_image.jpg")
                cv2.imwrite("memory_images/{}.jpg".format(id),
                            present_image)  # the cropped image (bounding box) of the detected object is written into a file'''

    return detected_objects


def get_ssim(cropped_image, duplicate_id):
    '''Judging similarity of 2 instances of the object from the same class and judgind their similarity based on
    scikit.measure.structural_similarity. It gave desiable output when 2 images were given which may or may not have been
     similar. It was chosen after comapring it with methods such as
    1. MSE (Mean square error)
    2. Template-matching(checking if the template (cropped instance of an object eg. face of a person in a family
    portrait) exists in the image - it wasn't very reliable when the image was too big and had a lot of information on it)
    3. Feature matching (a popular method to check for similarity of 2 images, where points of similarity are shown.
    However, the method was difficult to interpret in terms of usable data)
     The method was successful when google images or snapped photos of objects were compared but it wasn't helpful
     when the object detection using ZED was performed due to variation in camera image quality.'''
    try:
        cropped_image = cv2.cvtColor(cropped_image,
                                     cv2.COLOR_BGRA2BGR)  # the alpha channel in the cropped image is removed
        mem_image = cv2.imread(
            "memory_images/{}.jpg".format(duplicate_id))  # the cropped image of the object is called for comparison
        # print(cropped_image.shape, mem_image.shape)
        cropped_image = cv2.resize(cropped_image, (mem_image.shape[1], mem_image.shape[
            0]))  # resizing the present image with the stored image as both images need to be of the same size for comparison
        ssim = compare_ssim(cropped_image, mem_image,
                            multichannel=True)  # running an SSIM comparison using the already present function on a coloured image
        # ssim = compare_ssim(cropped_image, mem_image)  # running an SSIM comparison using the already present function on a grayscale image
        ssim = float(round(ssim, 3))  # restricting the output to 3 decimal points
    except:
        ssim = 0.99  # if there is an error return a pre-determined value of 0.99 - meaning images are similar

    return ssim


def image_compare_hist(cropped_image, duplicate_id):
    '''After the shortcoming of the ssim method, this method uses histogram comparison and ImageChops library from PIL
    which accounts for change in pose (rotated or flipped) and is more-reliable even though runs into issues as the object
    detection method primarily uses openCV and thus, the image now has to be first saved (as present_image.jpg using openCV)
    and then called from memory.'''
    error = 95  # error of margin for the comparison method i.e. if the images are 90% similar than it is the same object
    h1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)  # the Alpha channel is removed
    cv2.imwrite("present_image.jpg", h1)  # the present object instance is saved in memory
    h1 = Image.open("present_image.jpg")  # present image is opened for recognition
    #h2 = Image.open("memory_images/{}.jpg".format(duplicate_id))  # the image to be compared is opened for comparison
    try:
        h2 = Image.open("memory_images/{}.jpg".format(duplicate_id))
    except:
        h2 = Image.open("present_image.jpg")
    diff = ImageChops.difference(h1, h2).histogram()  # the histograms of the 2 images are compared
    sq = (value * (i % 256) ** 2 for i, value in
          enumerate(diff))  # a weighting function is applied to provide equal weights to all colours
    sum_squares = sum(sq)
    rms = math.sqrt(sum_squares / float(h1.size[0] * h1.size[1]))  # performing a root mean square

    # Error is an arbitrary value, based on values when
    # comparing 2 rotated images & 2 different images.
    return rms < error