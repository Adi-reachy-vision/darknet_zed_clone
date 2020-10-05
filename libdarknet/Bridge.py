import math
import random
import socket
from skimage.measure import compare_ssim
import numpy as np
import cv2


def socket_server_status(
        detections,
        point_cloud_data):  # a socket programme to transmit a certain amount of encrypted data via a TCP or UDP protocol between this program and some other program where the data is needed.
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 10000)
    message = detections + "//" + point_cloud_data

    try:
        # Send data
        sent = sock.sendto(message.encode(), server_address)
    finally:
        sock.close()


def opfileprint(detection):  # printing the detection into a file for the Overlord to read
    Fileman = open('YOLO_OUTPUT', 'w')  # creating and opening file in the write configuration
    for i in detection:
        Fileman.write(i)  # writing the detection into the file
        # Fileman.write('\n')
    Fileman.close()


def median_average(median, median_max, median_array):
    median_max.append(median)
    # print(len(median_max))
    while len(median_max) % 10 == 0:
        median_large = np.mean(median_max)
        median_max.clear()
        median_large = float(round(median_large, 2))
        median_array.append(median_large)
        # print("Values after comparison: ", median_large, median, len(median_max))
        if abs(median_large - median) > 2.00:
            median_array.clear()
        break

    try:
        return median_array[len(median_array) - 1]
    except:
        return median


def get_median_depth(y_extent, x_extent, y_coord, x_coord, depth, image, median_max, median_array):
    median_depth = []

    for i in range(y_extent):  # element by element multiplication of the height of the bounding box
        y_val = y_coord + (i - 1)
        for j in range(x_extent):  # element by element multiplication of the width of the bounding box
            x_val = x_coord + (j - 1)
            # print(x_val,j)
            try:
                calc_depth = depth[y_val, x_val]
                calc_depth = math.sqrt((calc_depth[0] * calc_depth[0]) + (calc_depth[1] * calc_depth[1]) + (
                        calc_depth[2] * calc_depth[2])) * 10  # calculating euclidian distance of a pixel
                median_depth = calc_depth

            except:
                pass

    median = float(round(np.median(median_depth), 2))
    median_large = median_average(median, median_max, median_array)
    # print("delay : ",median_large, median)

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
                    image[y_val, x_val] = (0, 0, 255, 0)

            except:
                pass
    return image


def get_color_segmentation_mask(cropped_image, color, mask, y_coord, y_extent, x_coord, x_extent, thresh):
    cropped_image = cv2.cvtColor(cropped_image,
                                 cv2.COLOR_BGRA2BGR)  # cropping the image to the size of the bounding box
    blue = color[0]  # storing blue value in bgr
    green = color[1]  # storing green value in bgr
    red = color[2]  # storing red value in bgr
    try:
        lower = np.array([(int(blue) - thresh), (int(green) - thresh), (int(red) - thresh)], dtype="uint8")
        upper = np.array([(int(blue) + thresh), (int(green) + thresh), (int(red) + thresh)], dtype="uint8")
        masked = cv2.inRange(cropped_image, lower, upper)

        mask[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)] = masked
    except:
        mask = mask
    return mask


def get_color_all(image):  # to test the robustness of the color recognition method and can be deleted later if needed
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    lower = np.array([1, 90, 200], dtype="uint8")
    upper = np.array([58, 155, 255], dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    try:
        mask = cv2.inRange(image, lower, upper)
    except:
        pass

    return mask


def get_color(image):
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
            # the mask
            mask = cv2.inRange(img, lower, upper)
            if count == 1:
                red_sum = int(np.sum(mask))  # seperating the values into seperate masks based on color
            elif count == 2:
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
    print_color = color_arrays[len(color_arrays) - 1]
    if print_color == blue_sum:
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


def get_similarity(cropped_image, new_id):
    try:

        image_prime = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2GRAY)
        mem_image = cv2.imread("memory_images/{}.jpg".format(new_id), 0)
        # mem_image = cv2.cvtColor(mem_image, cv2.COLOR_BGRA2GRAY)
        image_prime = cv2.resize(image_prime, (mem_image.shape[1], mem_image.shape[0]))
        ssim = compare_ssim(mem_image, image_prime)
        ssim = float(round(ssim, 3))
        return ssim
    except:
        return None



def get_detected_objects(detected_objects, label, bounds, x, y, z, camera_pose, py_translation, cropped_image):
    count_object = 0

    tx, ty, tz = get_positional_data(camera_pose, py_translation)

    if len(detected_objects) >= 0:

        for detected in detected_objects:
            if label != detected[1]:
                pass
            elif label == detected[1]:
                if abs(x - int(detected[2])) % 0.50 == 0 and abs(y - int(detected[3])) % 0.50 == 0 and abs(
                        y - int(detected[4])) % 0.50 == 0:
                    pass
                elif abs(x - int(detected[2])) % 0.50 != 0 and abs(y - int(detected[3])) % 0.50 != 0 and abs(
                        y - int(detected[4])) % 0.50 != 0:
                    count_object += 1
                    '''new_id = detected[0]
                    cv2.imwrite("memory_images/{}.jpg".format(new_id), cropped_image)
                    ssim = get_similarity(cropped_image, new_id)
                    if ssim < 0.5:
                        id = random.randint(1, 1000000000)
                        detected_o = [id, label, round(x - tx, 3), round(y - ty, 3),
                                      round(z + tz, 3)]
                        detected_objects.append(detected_o)
                    print(label, ssim)'''
                '''if abs(tx - float(detected[2])) % float(detected[2]) == 0 or abs(ty - float(detected[3])) % 0 == float(
                        detected[3]) or abs(tz - float(detected[4])) % float(detected[4]) == 0:'''  #activate in case the camera is constantly moving
                if abs(tx - float(detected[2])) == 0 or abs(ty - float(detected[3])) == 0 or abs(
                        tz - float(detected[4])) == 0:  #when the camera is stationery or object is moving
                    pass
                else:
                    id = detected[0]
                    detected_objects.remove(detected)
                    detected_o = [id, label, round(x, 3), round(y, 3),
                                  round(z, 3)]
                    detected_objects.append(detected_o)

    if count_object == 0:
        id = random.randint(1, 1000000000)
        detected_o = [id, label, round(x - tx, 3), round(y - ty, 3),
                      round(z + tz, 3)]
        detected_objects.append(detected_o)
        cv2.imwrite("memory_images/{}.jpg".format(id), cropped_image)

    '''elif count_object == 1:
        detected_o = [label + "/" + str(count_object), int(bounds[0]), int(bounds[1]), round(x, 3), round(y, 3),
                      round(z, 3)]
        detected_objects.append(detected_o)'''

    count_object = 0

    return detected_objects
