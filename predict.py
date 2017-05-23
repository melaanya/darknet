from ctypes import *
import sys
import numpy as np
from os import path
from optparse import OptionParser
from operator import itemgetter

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


 # weights_path - path to weights (for example: "backup/yolo-my_final.weights")
 # hypes_path - path to data description (for example: "ds/my.data")
 # options['model_def_path'] - model description ("cfg/yolo-my-test.cfg") REQUIRED!!!
 # more parameters in options: thresh
def initialize(weights_path, hypes_path, options=None):

    # default settings
    if options is None:
        print ("model description required!!! (for example cfg/yolo-my-test.cfg ")
        sys.exit()

    mydll = cdll.LoadLibrary('/darknet/libresult.so')
    if 'pred_options' in options and 'cfg_width' in options['pred_options'] and 'cfg_height' in options['pred_options']:
        grid_parameters = GridParam(options['pred_options']['cfg_width'], options['pred_options']['cfg_height'])
        mydll.initialize_network_test_param(str(options['model_def_path']), str(weights_path), grid_parameters)
        print("initialized")
    else:
        mydll.initialize_network_test(str(options['model_def_path']), str(weights_path))

    result = {'dll' : mydll, 'hypes_path': hypes_path, 'thresh': options['thresh'], 'hier_thresh': options['hier_thresh']}

    if "classID" in options:
        result.update({"classID": options["classID"]})

    return result


def hot_predict(image_path, init_params, to_json=True):
    if check_network(init_params['dll']):

        if not path.exists(image_path):
            print("danger! path doesn't exist! \n")

        if 'pred_options' in init_params:
            init_params['thresh'] = init_params['pred_options']['thresh']

        init_params['dll'].hot_predict.restype = BoxArray
        thresh = c_float(init_params['thresh'])
        hier_thresh = c_float(init_params['hier_thresh'])
        pred_boxes = init_params['dll'].hot_predict(init_params["hypes_path"], c_char_p(image_path), thresh, hier_thresh)

        if to_json:
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
                if box['x1'] == box['x2'] or box['y1'] == box['y2']:
                    continue;

                # print(box)

                # yolo v2 was trained for classes from 0 to 18, but we need label 0 for background - so renumerating 
                #  todo: retrain yolo v2 for classes from 1 to 19 and compare the results
                # note: yolo v2 requires classes from 0 ! maybe crucial, you need to think about it
                if "classID" in init_params:
                    box['classID'] = init_params["classID"]
                else:
                    box['classID'] = int(pred_boxes.arr[ind].class_num) + 1
                result.append(box)
            return result

        return pred_boxes
    else:
        print("no initialisation of network happened before")
        sys.exit()


def main():
    parser = OptionParser(usage='usage: %prog [options] <dataset path> <weights> <hypes>')
    options = {}
    options['thresh'] = 0.2
    options['hier_thresh'] = 0.5
    options['model_def_path'] = "cfg/yolo-my-test.cfg"

    hypes_path = "ds/my.data"
    weights = "backup/yolo-my_final.weights"
    image_filename = "ds/images/1.jpg"


    init_params = initialize(weights, hypes_path, options)

    result = hot_predict(image_filename, init_params, True)

    # print(result)


if __name__ == '__main__':
    main()
