import io
import math
import random
import socket
from skimage.measure import compare_ssim
import numpy as np
import cv2
from PIL import Image, ImageChops


def socket_server_status(detections, point_cloud_data):
    '''The Detection data is transmitted via a socket connection to OverLord.py via TCP or UDP protocol,
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #SOCK_DGRAM stands for datagram which is UDP's method of data transmission
    server_address = ('localhost', 10000)   #the local host address and port number. If you run into an error try changing, as the port might be occupied by some other process
    message = detections + "//" + point_cloud_data  #the detected data separated by a separator '//' from the point cloud location data of x, y, z value of the bounding box for location estimation of the object in the environment

    try:
        # Send data in a try block to prevent the presence of an error
        sent = sock.sendto(message.encode(), server_address)
    finally:
        sock.close()    #once the data has been sent, the socket can be closed


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


def image_segmentation_depth(y_extent, x_extent, y_coord, x_coord, depth, image, median_max, median_array):
    '''Perform image segmentation based on depth information received by ZED camera's point cloud data.
    The depth of the pixels in the bounding box are added to a flattened array and after averaging their depth,
    the mean depth of the object is received and the depth of the pixels is once again compared to analyse if the
     target pixel is greater or lesser than the mean depth of the object. If it is lesser than the mean depth, the pixel
     is filled with red colour.'''
    median_depth = [] #initialising an array to store the depth of all pixels in the bounding box
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
                median_depth = calc_depth     #storing the scaled up depth in a flattened array for avergaing

            except:
                pass

    median = float(round(np.median(median_depth), 2)) #calculating median depth using numpy median method
    median_large = median_average(median, median_max, median_array) # an averaging function to stablise the depth over
                                                                    # multiple frames by finding median depth over multiple
                                                                    #iterations of stored average distances of the object in the bounding box
    # print("delay : ",median_large, median)
    height = height/2
    y_grasp = y_coord + height
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
                    if y_val == y_grasp:
                        image[y_val, x_val] = (0, 255, 0, 0)
                    else:
                        image[y_val, x_val] = (0, 0, 255, 0)

            except:
                pass
    return image    #returning a modified image which has depth based segmentation mask
                    # marked on the image in the area of the bounding box

def median_average(median, median_max, median_array):
    '''Averaging function to stablise the depth obtained from 'image_segmentation_depth'. The idea behind this was
    behind the assumption that depth estimate of one frame was accurate but the depth over multiple frames would allow
    a better understanding of the depth of the object which provides the program with it's segmentation mask.'''
    median_max.append(median)
    # print(len(median_max))
    while len(median_max) % 10 == 0:    #averaging function over 10 frames
        median_large = np.mean(median_max)      #provide a mean value for the median values stored in the median_max array
        median_max.clear()                      #once the averaging function is performed, the median_max array is cleared.
        median_large = float(round(median_large, 3))    #rounding the median average to 3 decimal points
        median_array.append(median_large)               #storing the average avergae median in a new array to ensure,
                                                        # the value is stored over multiple frames and doesnt get
                                                        #reinitialised every time, the function is called
        # print("Values after comparison: ", median_large, median, len(median_max))
        if abs(median_large - median) > 2.00:           #if the median average has moved more than 2 metres from the
                                                        # median of the present frame than it means, the object has moved
                                                        # the avergaing function over multiple feames has to be restarted.
            median_array.clear()
        break

    try:
        return median_array[len(median_array) - 1]      #return the average function if present
    except:
        return median                                   #if the average values aren't present return the median of the
                                                        # past frame which was being analysed


def image_segmentation_colour(cropped_image, color, mask, y_coord, y_extent, x_coord, x_extent, thresh):
    '''An alternate image segmentation method which provides a black and white segmentaiton mask based on the presence
    of colour pixels in the designated bounding box area.The center pixel as designated by yolo is marked as the pixel
    of interest whose colour value is taken as the colour of interest and after adding a threshold as padding to keep it within
    a range, the presence of colour in those BGR values is marked in balck & white as a masking method based on colour.

    Depth based segmentation is the primary method but colour based segmentation is also useful in some situations,
    personally it works phenomenally well, in darker colour regions.'''

    cropped_image = cv2.cvtColor(cropped_image,
                                 cv2.COLOR_BGRA2BGR)  # cropping the image to the size of the bounding box
    blue = color[0]  # storing blue value in bgr
    green = color[1]  # storing green value in bgr
    red = color[2]  # storing red value in bgr
    try:    #encapsulating the function in a try block to prevent the program from crashing if the right pixel is not found
        lower = np.array([(int(blue) - thresh), (int(green) - thresh), (int(red) - thresh)], dtype="uint8") #setting the lower barrier of the inRange function
        upper = np.array([(int(blue) + thresh), (int(green) + thresh), (int(red) + thresh)], dtype="uint8") #setting the upper barrier of the inRange function
        masked = cv2.inRange(cropped_image, lower, upper)   #determining the presence of the colour in the bounding box and returning it as a 8-bit integer mask

        mask[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)] = masked   #pasting the bounding box mask into the bigger mask image
    except:
        mask = mask     #returning the original mask (a blank black-white image) in case the try block fails to run
    return mask


def color_test(image):
    '''The color recognition of objects needed to be refined using custom determined values. The initial testing method involved passing
    boundaries for the colours in the RGB matrix from images on the internet which had a different colour composition than
    from the ones, the camera was seeing. Thus, it had to be altered using a testing method like this function, where the colour
    values of BGRA pixel was manually entered and it was determined to be the best range for the colour composition.'''
    # to test the robustness of the color recognition method and can be deleted later if needed
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) #the Alpha Channel is stripped from the initial image
    lower = np.array([1, 90, 200], dtype="uint8")   #the lower boundary for the inRange function is set
    upper = np.array([58, 155, 255], dtype="uint8") #the upper boundary for the inRange function is set
    # find the colors within the specified boundaries and apply
    try: # a try block to prevent the program from crashing if the method doesnt run smoothly
        mask = cv2.inRange(image, lower, upper)     #returning a mask with the colour values highlighted as white pixels and blank as the ones that don't
    except:
        pass    # do nothing if the method doesn't work and continue executing the function.

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
            mask = cv2.inRange(img, lower, upper) # the inRange function which returns the mask value
            if count == 1:
                red_sum = int(np.sum(mask))  # separating the values into separate masks based on color
            elif count == 2:                #adding the mask values and storing them into variables, depending on the iteration of the loop
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
    print_color = color_arrays[len(color_arrays) - 1]   #the values are then sorted in an array ascending order and
                                                        # returning the last value (the largest) as the most dominant
                                                        # colour in the bounding box
    if print_color == blue_sum:        #determing the colour, the highest value is equal to and returning that value in a string
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
    rotation = camera_pose.get_rotation_vector()
    rx = round(rotation[0], 2)
    ry = round(rotation[1], 2)
    rz = round(rotation[2], 2)

    translation = camera_pose.get_translation(py_translation)
    tx = round(translation.get()[0], 3)
    ty = round(translation.get()[1], 3)
    tz = round(translation.get()[2], 3)

    text_translation = str((tx, ty, tz))
    text_rotation = str((rx, ry, rz))
    return tx, ty, tz


def get_similarity(cropped_image, duplicate_id):
    try:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
        mem_image = cv2.imread("memory_images/{}.jpg".format(duplicate_id))
        # print(cropped_image.shape, mem_image.shape)
        cropped_image = cv2.resize(cropped_image, (mem_image.shape[1], mem_image.shape[0]))
        ssim = compare_ssim(cropped_image, mem_image, multichannel=True)
        ssim = float(round(ssim, 3))
    except:
        ssim = 0.99

    return ssim


def images_are_similar(cropped_image, duplicate_id):
    error = 90
    h1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("present_image.jpg", h1)
    h1 = Image.open("present_image.jpg")
    h2 = Image.open("memory_images/{}.jpg".format(duplicate_id))
    '''try:
        h2 = Image.open("memory_images/{}.jpg".format(duplicate_id))
    except:
        h2 = Image.open("present_image.jpg")'''
    diff = ImageChops.difference(h1, h2).histogram()
    sq = (value * (i % 256) ** 2 for i, value in enumerate(diff))
    sum_squares = sum(sq)
    rms = math.sqrt(sum_squares / float(h1.size[0] * h1.size[1]))

    # Error is an arbitrary value, based on values when
    # comparing 2 rotated images & 2 different images.
    return rms < error


def get_detected_objects(detected_objects, label, x, y, z, camera_pose, py_translation, cropped_image):
    count_object = 0
    new_id = []
    new_id_alt = []
    tx, ty, tz = get_positional_data(camera_pose, py_translation)
    if len(detected_objects) >= 0:

        for detected in detected_objects:
            new_id_alt.append(detected[1])
            if label != detected[1]:
                pass
            elif label == detected[1]:
                if abs(x - int(detected[2])) % 0.050 == 0 and abs(y - int(detected[3])) % 0.050 == 0 and abs(
                        y - int(detected[4])) % 0.050 == 0:
                    array_valueid_alt = [detected[0], detected[2], detected[3], detected[4], label]
                    new_id_alt.append(array_valueid_alt)
                elif abs(x - int(detected[2])) % 0.050 != 0 and abs(y - int(detected[3])) % 0.050 != 0 and abs(
                        z - int(detected[4])) % 0.050 != 0:
                    count_object += 1

                    array_valueid = [label, x, y, z]
                    if array_valueid not in new_id:
                        new_id.append(array_valueid)
                    # print(new_id)
            if abs(tx - float(detected[2])) % float(detected[2]) == 0 or abs(ty - float(detected[3])) % 0 == float(
                    detected[3]) or abs(tz - float(detected[4])) % float(
                detected[4]) == 0:  # activate in case the camera is constantly moving
                pass
            else:
                id_new = detected[0]
                label_new = detected[1]
                detected_objects.remove(detected)
                detected_o = [id_new, label_new, round(x - tx, 3), round(y - ty, 3),
                              round(z + tz, 3), "change"]
                detected_objects.append(detected_o)

    if label not in new_id_alt:
        id = random.randint(1, 1000000000)
        detected_o = [id, label, round(x - tx, 3), round(y - ty, 3),
                      round(z + tz, 3), "orig"]
        detected_objects.append(detected_o)
        cv2.imwrite("memory_images/{}.jpg".format(id), cropped_image)

    else:

        for entry in new_id:
            #print(new_id)
            for detected in detected_objects:
                exist_value = False
                if detected[1] == entry[0]:
                    if (abs(entry[1]) - abs(detected[2])) > 0.5 or (abs(entry[2]) - abs(detected[3])) > 0.5 or (
                            abs(entry[3]) - abs(detected[4])) > 0.5:
                        if images_are_similar(cropped_image,detected[0]) == False:
                            exist_value = True
            if exist_value is True:
                id = random.randint(1, 1000000000)
                detected_o = [id, entry[0], round((entry[1] - tx), 3), round((entry[2] - ty), 3),
                              round((entry[3] - tz), 3), "new"]
                detected_objects.append(detected_o)

    return detected_objects

