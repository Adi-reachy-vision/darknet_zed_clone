#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import multiprocessing
import os
import socketserver
import sys
import threading
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl
import socket

from threading import Thread

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
print("GPU available")
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    # winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")

    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '" + tmp + "' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        print("What is a GPU?")
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll + "`. Trying a GPU run anyway.")
else:
    lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)

    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])

    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """

    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)

    im, arr = array_to_image(custom_image)

    num = c_int(0)

    pnum = pointer(num)

    predict_image(net, im)

    x = time.time()
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    y = time.time()
    z = y - x
    # print("network boxes" + str(z))
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []

    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))

    res = sorted(res, key=lambda x: -x[1])

    free_detections(dets, num)

    return res


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


netMain = None
metaMain = None
altNames = None


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median


def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array


def get_median_depth(y_extent, x_extent, y_coord, x_coord, depth):
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
    median = np.median(median_depth)

    return median


def get_color(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    blue = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    green = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    red = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    white = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    boundaries = [([17, 15, 100], [50, 56, 200]),
                  ([86, 31, 4], [220, 88, 50]),
                  ([25, 146, 190], [62, 174, 250]),
                  ([103, 86, 65], [145, 133, 128])]
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
                blue = mask
            elif count == 2:
                green = mask
            elif count == 3:
                red = mask
            elif count == 4:
                white = mask

        except:
            pass

    blue_count = 0
    green_count = 0
    red_count = 0
    white_count = 0
    
    if blue.any() > green.any():
        blue_count += 1
    if blue.any() > red.any():
        blue_count += 1
    if blue.any() > white.any():
        blue_count += 1

    if green.any() > red.any():
        green_count += 1
    if green.any() > white.any():
        green_count += 1
    if green.any() > blue.any():
        green_count += 1

    if red.any() > white.any():
        red_count += 1
    if red.any() > blue.any():
        red_count += 1
    if red.any() > green.any():
        red_count += 1

    if white.any() > blue.any():
        white_count += 1
    '''if white.any() > green.any:
        white_count += 1'''
    if white.any() > red.any():
        white_count += 1

    color_arrays = [(blue_count, "Blue"), (green_count, "green"), (red_count, "red"), (white_count, "white")]
    final_color_array = np.sort(color_arrays, axis=0)
    final_color = final_color_array[len(final_color_array) - 1]
    final_color = final_color[1]


    return final_color


def main(argv):
    thresh = 0.25
    darknet_path = "../libdarknet/"
    config_path = darknet_path + "cfg/yolov3.cfg"
    weight_path = "yolov3.weights"
    meta_path = "coco.data"
    svo_path = None
    zed_id = 0
    global x
    global y
    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.15
    init.camera_resolution = sl.RESOLUTION.VGA
    init.camera_fps = 60

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Use FILL sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path) + "`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path) + "`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path) + "`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1

    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generate_color(meta_path)

    log.info("Running...")
    processes = []
    key = ''
    while key != 113:  # for 'q' key
        point_cloud_data = ""
        probs = time.time()
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            image = mat.get_data()

            cam.retrieve_measure(
                point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data()
            # Do the detection
            start_time = time.time()  # start time of the loop
            detections = detect(netMain, metaMain, image, thresh)
            # opfileprint(str(detections))

            # Boolean value to ensure, the value is constrained to being sent only once
            # broadcasts the detected objects data from darknet_zed to Overlord
            # Boolean value set to false, once the value has printed already

            bench_time = time.time()  # setting checkpoint for the loop
            mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            # log.info(chr(27) + "[2J" + "**** " + str(len(detections)) + " Results ****")  # printing detected objects

            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                # log.info(pstring)
                #

                bounds = detection[2]
                y_extent = int(bounds[3])
                x_extent = int(bounds[2])
                # Coordinates are around the center
                x_coord = int(bounds[0] - bounds[2] / 2)
                y_coord = int(bounds[1] - bounds[3] / 2)
                # boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
                thickness = 1

                x, y, z = get_object_depth(depth, bounds)
                distance = math.sqrt(x * x + y * y + z * z)
                depth_var = distance * 10  # obtaining a scaled euclidian distance of an anchor point in the bounding box as a threshold
                distance = "{:.2f}".format(distance)
                distance_data = str(label) + ", position from camera x = " + str(round(x, 2)) + " m,  y= " + str(
                    round(y, 2)) + " m,  z= " + str(round(z, 2)) + " m,"

                # print(np.median(median_depth),label)
                if label == "bottle":  # if statement to filter the classes needed for segmentation
                    cropped_image = image[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)]
                    color_string = get_color(
                        cropped_image)  # getting the color output from the color recognition function
                    # print(masked)
                    '''median = get_median_depth(y_extent, x_extent, y_coord, x_coord,
                                              depth)  # getting median depth from the function for establishing depth threshold of the bounding box for segmentation
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
                                if calc_depth < median:  # comparing the pixel distance from the threshold
                                    # print("True")
                                    image[y_val, x_val] = (0, 55, 0, 0)
                            except:
                                pass'''
                    cv2.putText(image, color_string + " " + label + " " + (str(distance) + " m"),
                                (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                2)  # pasting label on top of the segmentation mask

                else:  # j += 1

                    '''cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                                  (x_coord + x_extent + thickness, y_coord + (18 + thickness * 4)),
                                  color_array[detection[3]], -1)'''
                    cv2.putText(image, label + " " + (str(distance) + " m"),  # pasting label on top of detected object
                                (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                                  # pasting bounding box around detected object
                                  (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                                  color_array[detection[3]], int(thickness))
                point_cloud_data += distance_data

            cv2.imshow("ZED", image)
            # cv2.imshow("mask", mask)
            key = cv2.waitKey(5)
            socket_server_status(str(detections), point_cloud_data)

            # log.info("Detection time: {}".format(bench_time - start_time))
            # log.info("Camera FPS: {}".format(1.0 / (time.time() - bench_time)))
            # log.info("Output FPS: {}".format(1.0 / (time.time() - probs)))
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    log.info("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])