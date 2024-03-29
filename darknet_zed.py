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
import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2 as cv2
import pyzed.sl as sl
import Bridge

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



count = 0
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

def get_positional_data(camera_pose, py_translation):
    rotation = camera_pose.get_rotation_vector()  # The rotation information from the gyroscopic sensors
    rx = round(rotation[0], 2)  # The rotational information of the x-axis
    ry = round(rotation[1], 2)  # The rotational information of the y-axis
    rz = round(rotation[2], 2)  # The rotational information of the z-axis

    translation = camera_pose.get_translation(py_translation)  # The translational information from the IMU sensors
    tx = round(translation.get()[0], 3)  # The translational information of the x-axis
    ty = round(translation.get()[1], 3)  # The translational information of the y-axis
    tz = round(translation.get()[2], 3)  # The translational information of the z-axis

    return tx,ty,tz,rx,ry,rz


def main(argv):
    thresh = 0.5
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
        # log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    # ZED parameters

    init = sl.InitParameters(input_t=input_type) #input parameters for ZED
    #init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP #ZED coordinate system
    init.sdk_verbose = True
    init.coordinate_units = sl.UNIT.METER # ZED's distance measurement
    init.depth_minimum_distance = 0.20 # ZED's minimum distance setting
    init.camera_resolution = sl.RESOLUTION.HD720 # Camera resolution
    init.depth_mode = sl.DEPTH_MODE.QUALITY # Depth mode setting between ULTRA, QUALITY, PERFORMANCE
    init.camera_fps = 60 # Camera fps

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Use FILL sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.FILL # Sensing mode
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
    existing_labels = [] # exisitng labels array used in detecetd objects array to store first occurence of an object class
    avg_median = [] # array to store average median of bounding box over a few frames for a specific object
    median_max = [] # array to store median values over the past 'n' number of frames
    detected_objects = [] # array to store all detected unique objects (memory component)
    grasp_y_delay = [] # array to store grasping y_axis coordinates on an object over multiple frames
    shallow_cluster_y = [] # array to store grasping y-axis points based on shallow depth of the object over multiple frames
    positional_buffer_array = [] # array to store positional data over multiple frames
    rotational_buffer_array = [] # array to store rotational data over multiple frames
    x_centroid = []
    x_centroid_marker = [] # array to store black points
    #live_feed = None
    key = ''
    sensor_data = sl.SensorsData() # variable to store sensor data from ZED camera
    transform = sl.Transform() # variable to store transform data from ZED camera
    tracking_params = sl.PositionalTrackingParameters(transform)  # initialises positional tracking
    cam.enable_positional_tracking(tracking_params)  # enables positional tracking
    while key != 113:  # for 'q' key
        point_cloud_data = ""
        probs = time.time()
        err = cam.grab(runtime) # variable to verify if the camera is running successfully
        camera_pose = sl.Pose() # capturing pose data necessary for tracking information
        if err == sl.ERROR_CODE.SUCCESS:
            cam.get_sensors_data(sensor_data, sl.TIME_REFERENCE.IMAGE) # retrieve sensor data into variable sensor_data
            cam.retrieve_image(mat, sl.VIEW.LEFT) # obtain left camera frame and store it in variable - image
            cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)  # point cloud data extracted in variable - depth

            image = mat.get_data() #image to be processed
            py_translation = sl.Translation() # translation data
            depth = point_cloud_mat.get_data()

            start_time = time.time()  # start time of the loop - checkpoint (pre-detection)
            mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))  # a black image, the size of the input camera frame

            tx, ty, tz, rx, ry, rz, stable = Bridge.positional_buffer_CAMERAframe(camera_pose, py_translation, positional_buffer_array,rotational_buffer_array)
            if stable is True:
                # Do the detection when the camera is not moving
                detections = detect(netMain, metaMain, image, thresh) # detections performed and obtained from yolo framework
            else:
                detections.clear()

            bench_time = time.time()  # setting checkpoint for the loop - checkpoint (post detection)

            tracking_state = cam.get_position(camera_pose, sl.REFERENCE_FRAME.LAST)  # initialises a positional tracking sequence to give the distance moved by the camera using camera frame reference

            # log.info(chr(27) + "[2J" + "**** " + str(len(detections)) + " Results ****")  # printing detected objects

            for detection in detections:

                label = detection[0] # label of the object class

                confidence = detection[1] # confidence of the detection

                pstring = label + ": " + str(np.rint(100 * confidence)) + "%" #detection data that can be showed in terminal window

                bounds = detection[2] #bounding box coordinates for the object
                y_extent = int(bounds[3])
                x_extent = int(bounds[2])
                # Coordinates are around the center
                x_coord = int(bounds[0] - bounds[2] / 2)
                y_coord = int(bounds[1] - bounds[3] / 2)
                # boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]

                thickness = 1

                x, y, z = get_object_depth(depth, bounds) # distance estimated by the point cloud data
                distance = math.sqrt(x * x + y * y + z * z)
                distance = "{:.2f}".format(distance) # determining the distance in eucilidian distance format

                distance_data = str(label) + ", position from camera x = " + str(round(x, 2)) + " m,  y= " + str(
                    round(y, 2)) + " m,  z= " + str(round(z, 2)) + " m," # generating a string with distance data to be displayed in the socket

                cropped_image = image[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)] # cropped image for image comparison or colour recognition

                if label == "cup":  # a binding statement to direct colour recognition

                    cropped_image = image[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)]  # cropping image to the size of the object bounding box

                    # mask[y_coord:(y_coord + y_extent), x_coord:(x_coord + x_extent)] = color_test(cropped_image)  # getting the color output from the color recognition function

                    color_string = Bridge.get_color(cropped_image)  # getting colour output from the function as a string

                    thresh_color = 10 # threshold range for colour based segmentation
                    #mask = Bridge.image_segmentation_colour(image, mask, y_coord, y_extent, x_coord, x_extent,thresh_color, bounds)
                    # segmentation based on colour with adaptive threshold range

                    label = color_string + " " + label

                if label == "Blue cup":  # if statement to filter the classes needed for segmentation

                    image = Bridge.image_segmentation_depth(y_extent, x_extent, y_coord, x_coord, depth, image, median_max, avg_median, grasp_y_delay, shallow_cluster_y, x_centroid, x_centroid_marker)
                    # performing image segmentation on the image frame

                    cv2.putText(image, label + " " + (str(distance) + " m"),
                                (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # pasting label on top of the segmentation mask

                else:  # general object detection and labeling

                    cv2.putText(image, label + " " + (str(distance) + " m"),  # pasting label on top of detected object
                                (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),    # pasting bounding box around detected object
                                  (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                                  color_array[detection[3]], int(thickness))

                detected_objects = Bridge.get_detected_objects(detected_objects, label, x, y, z, camera_pose,
                                                               py_translation, cropped_image, existing_labels,
                                                               positional_buffer_array, rotational_buffer_array)
                #   detected objects array which stores all detections in the past in an array

                point_cloud_data += distance_data # adding distance data to be displayed over the socket display in Overlord

            #function to enable control by Overlord_camera_function
            #if you enter 'y' - it shows detection_camera frame
            #if you enter 'n' - it shows a blank image
            '''camera_control = Bridge.socket_client_control()
            if camera_control == "b'live'":
                if live_feed is False:
                    cv2.destroyWindow("mask")
                    live_feed = True
                else:
                    live_feed = True
                cv2.imshow("ZED", image)
            else:
                if live_feed is True:
                    cv2.destroyWindow("ZED")
                    live_feed = False
                else:
                    live_feed = False
                cv2.imshow("mask", mask)'''

            cv2.imshow("ZED", image)
            key = cv2.waitKey(5)
            print(detected_objects)
            Bridge.opfileprint(str(detections)) # writing detection data for the frame into file "YOLO_OUTPUT.txt"
            Bridge.socket_server_detected_objects(str(detected_objects)) # displaying detected objects (memory) over a socket connection
            Bridge.socket_server_status(str(detections), point_cloud_data) # displaying detection data of the frame over a socket connection
            # output = time.time() - probs # check point to see total loop time
            # log.info("Detection time: {}".format(bench_time - start_time)) #checking how long the detection takes
            # log.info("Camera FPS: {}".format(1.0 / (time.time() - bench_time)))
            # log.info("Output FPS: {}".format((1.0 / (time.time() - probs))))
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    log.info("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])