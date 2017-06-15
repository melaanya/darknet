from ctypes import *
import sys, json
import numpy as np
from os import path
from optparse import OptionParser
from PIL import Image


def check_network(dll):
    return c_int.in_dll(dll, "network_created")


class Box(Structure):
    _fields_ = [
        ("left", c_float),
        ("top", c_float),
        ("right", c_float),
        ("bottom", c_float),
        ("class_num", c_int)]    


class BoxArray(Structure):
    _fields_ = [
    ("arr", POINTER(Box)),
    ("size", c_int)]


class GridParam(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int)]


class ImageYolo(Structure):
    _fields_ = [
        ("h", c_int),
        ("w", c_int),
        ("c", c_int),
        ("data", POINTER(c_float))]


 # weights_path - path to weights (for example: "backup/yolo-my_final.weights")
 # hypes_path - path to data description (for example: "ds/my.data")
 # options['model_def_path'] - model description ("cfg/yolo-my-test.cfg") REQUIRED!!!
 # more parameters in options: thresh
def initialize(weights_path, hypes_path, config=None):

    if "yolo" not in config:
        print ("model description required (for example cfg/yolo-my-test.cfg ")
        sys.exit()

    mydll = cdll.LoadLibrary('/darknet/libresult.so')
    if 'pred_options' in config and 'cfg_width' in config['pred_options']:
        grid_parameters = GridParam(config['pred_options']['cfg_width'], config['pred_options']['cfg_width'])
        mydll.initialize_network_test_param(str(config["yolo"]['model_def_path']), str(weights_path), grid_parameters)
    elif 'cfg_width' in config["yolo"]:
        print("changed cfg")
        grid_parameters = GridParam(config["yolo"]['cfg_width'], config["yolo"]['cfg_width'])  # make square as we should save aspect ratio
        mydll.initialize_network_test_param(str(config["yolo"]['model_def_path']), str(weights_path), grid_parameters)
    else:
        mydll.initialize_network_test(str(config["yolo"]['model_def_path']), str(weights_path))

    result = {'dll' : mydll, 'hypes_path': hypes_path, 'thresh': config["yolo"]['thresh'], 'hier_thresh': config["yolo"]['hier_thresh'],
              "sliding_predict": config["sliding_predict"]}

    if "classID" in config:
        result.update({"classID": config["classID"]})

    return result


def hot_predict(image_path, init_params):
    if check_network(init_params['dll']):
        # print(image_path)
        if not path.exists(image_path):
            print("danger! path doesn't exist! \n")
            exit(1)

        if 'pred_options' in init_params:
            init_params['thresh'] = init_params['pred_options']['thresh']

        if "sliding_predict" in init_params and "sliding_window" in init_params["sliding_predict"] and init_params["sliding_predict"]["sliding_window"]:
            print("sliding")
            return sliding_predict(image_path, init_params)
        else:
            print("regular")
            return regular_predict(image_path, init_params)

    else:
        print("no initialisation of network happened before")
        sys.exit()


def regular_predict(image_path, init_params):
    init_params['dll'].hot_predict.restype = BoxArray
    pred_boxes = init_params['dll'].hot_predict(init_params["hypes_path"], c_char_p(image_path), c_float(init_params['thresh']), c_float(init_params['hier_thresh']))
    result = []
    for ind in range(0, pred_boxes.size):
        box = {}
        if np.isnan(pred_boxes.arr[ind].left) or np.isnan(pred_boxes.arr[ind].top) \
                or np.isnan(pred_boxes.arr[ind].right) or np.isnan(pred_boxes.arr[ind].bottom):
            continue
        box['x1'] = int(pred_boxes.arr[ind].left)
        box['y1'] = int(pred_boxes.arr[ind].top)
        box['x2'] = int(pred_boxes.arr[ind].right)
        box['y2'] = int(pred_boxes.arr[ind].bottom)

        # yolo v2 was trained for classes from 0 to 18, but we need label 0 for background - so renumerating
        # note: yolo v2 requires classes from 0 so we can't retrain from 1 to 19
        if "classID" in init_params:
            box['classID'] = init_params["classID"]
        else:
            box['classID'] = int(pred_boxes.arr[ind].class_num) + 1
        result.append(box)
    return result


def sliding_predict(image_path, init_params):
    init_params['dll'].hot_predict_image.restype = BoxArray
    img = Image.open(image_path)
    width, height = img.size

    # cutting here
    result = []
    for count, i in enumerate(range(0, height, init_params["sliding_predict"]["step"])):
        i = i - count * init_params["sliding_predict"]["overlap"]
        print((0, i, width, min(height, i + init_params["sliding_predict"]["step"])))
        img_part = img.crop((0, i, width, min(height, i + init_params["sliding_predict"]["step"])))
        pred_boxes = predict_from_img(img_part, init_params)
        for ind in range(0, pred_boxes.size):
            box = {}
            if np.isnan(pred_boxes.arr[ind].left) or np.isnan(pred_boxes.arr[ind].top) \
                    or np.isnan(pred_boxes.arr[ind].right) or np.isnan(pred_boxes.arr[ind].bottom):
                continue
            box['x1'] = int(pred_boxes.arr[ind].left)
            box['y1'] = int(pred_boxes.arr[ind].top) - i
            box['x2'] = int(pred_boxes.arr[ind].right)
            box['y2'] = int(pred_boxes.arr[ind].bottom) - i

            if "classID" in init_params:
                box['classID'] = init_params["classID"]
            else:
                box['classID'] = int(pred_boxes.arr[ind].class_num) + 1
            result.append(box)
        print(result)
    # here goes combining the boxes


    # temporary return
    return result

def predict_from_img(img, init_params):
    arr = np.array(img)
    h, w, c = arr.shape
    print(arr.shape)

    data = [c_float(0)] * arr.size

    for k in range(0, c):
        for j in range(0, h):
            for i in range(0, w):
                dst_index = i + j * w + w * h * k
                data[dst_index] = arr[j, i, k] / 255.0

    arr = (c_float * arr.size)(*data)
    ptr = cast(arr, POINTER(c_float))

    image = ImageYolo(c_int(h), c_int(w), c_int(c), ptr)
    pred_boxes = init_params['dll'].hot_predict_image(init_params["hypes_path"], image, c_float(init_params['thresh']),
                                                      c_float(init_params['hier_thresh']))

    del arr
    return pred_boxes


def main():
    parser = OptionParser(usage='usage: %prog [options] <config>')
    _, args = parser.parse_args()

    if len(args) < 1:
        print('Provide path configuration json file')
        return

    if not path.exists(args[0]) or not path.isfile(args[0]):
        print ('Bad configuration path provided!')
        return

    config = json.load(open(args[0], "r"))
    # image for test
    image_filename = "19.jpg"

    init_params = initialize(config["weights"], config["hypes"], config)
    result = hot_predict(image_filename, init_params)
    print(result)

if __name__ == '__main__':
    main()
