from __future__ import division
from ctypes import *
import sys, json, subprocess
import numpy as np
from os import path
from optparse import OptionParser
from PIL import Image, ImageDraw

if __package__ is None:
    import sys
    sys.path.append(path.abspath(path.join(path.dirname(__file__), path.pardir)))
    sys.path.append("/darknet/detect-widgets")

from geometry import iou

def check_network(dll):
    return c_int.in_dll(dll, "network_created")

def save_results(src_path, dst_path, rects, classes):
    """Saves results of the prediction.

    Args:
        src_path (string): The path to source image to predict bounding boxes.
        dst_path (string): The path to source image to predict bounding boxes.
        rects (list): The collection of boxes to draw on screenshot.
        classes (list): The collection of classes corresponding their ids 

    Returns: 
        Nothing.
    """

    # draw
    new_img = Image.open(src_path)
    draw = ImageDraw.Draw(new_img)
    for r in rects:
        draw.text(((r['x1'] + r['x2']) / 2, (r['y1'] + r['y2']) / 2),
                  text=classes[r['classID']], fill='purple')
        draw.rectangle([r['x1'], r['y1'], r['x2'], r['y2']], outline=(255, 0, 0))
    # save
    new_img.save(dst_path)
    subprocess.call(['chmod', '644', dst_path])


class Box(Structure):
    _fields_ = [
        ("left", c_float),
        ("top", c_float),
        ("right", c_float),
        ("bottom", c_float),
        ("class_num", c_int),
        ("conf", c_float)]


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
 # options['model_def_path'] - model description ("cfg/yolo-my-test.cfg") REQUIRED
def initialize(weights_path, hypes_path, config=None, verbose=False):
    if "yolo" not in config:
        print ("model description required (for example cfg/yolo-my-test.cfg ")
        sys.exit()

    mydll = cdll.LoadLibrary(config["yolo"]["so_library_path"])
    if 'pred_options' in config and 'cfg_width' in config['pred_options']:
        grid_parameters = GridParam(config['pred_options']['cfg_width'], config['pred_options']['cfg_width'])
        mydll.initialize_network_test_param(str(config["yolo"]['model_def_path']), str(weights_path), grid_parameters)
    elif 'cfg_width' in config["yolo"]:
        if verbose:
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


def hot_predict(image_path, init_params, verbose=False):
    if check_network(init_params['dll']):
        # to fix change names of paths in file
        # image_path = ".." + image_path
        if verbose:
            print(image_path)
        if not path.exists(image_path):
            print("danger! path doesn't exist! \n")
            exit(1)

        if 'pred_options' in init_params:
            init_params['thresh'] = init_params['pred_options']['thresh']
        if "sliding_predict" in init_params and "sliding_window" in init_params["sliding_predict"] and init_params["sliding_predict"]["sliding_window"]:
            if verbose:
                print("sliding")
            return sliding_predict(image_path, init_params)
        else:
            if verbose:
                print("regular")
            return regular_predict(image_path, init_params)

    else:
        print("no initialisation of network happened before")
        sys.exit()


def regular_predict(image_path, init_params):
    init_params['dll'].hot_predict.restype = BoxArray
    pred_boxes = init_params['dll'].hot_predict(init_params["hypes_path"], c_char_p(image_path), c_float(init_params['thresh']),
                                                c_float(init_params['hier_thresh']))
    return process_result_boxes(pred_boxes, init_params)


def sliding_predict(image_path, init_params):
    init_params['dll'].hot_predict_image.restype = BoxArray
    img = Image.open(image_path)
    width, height = img.size

    assert(init_params["sliding_predict"]["step"] > init_params["sliding_predict"]["overlap"])

    # cutting
    result = []
    for idx, i in enumerate(range(0, height, init_params["sliding_predict"]["step"] - init_params["sliding_predict"]["overlap"])):
        # print((0, i, width, min(height, i + init_params["sliding_predict"]["step"])))
        img_part = img.crop((0, i, width, min(height, i + init_params["sliding_predict"]["step"])))
        pred_boxes = predict_from_img(img_part, init_params, idx)
        processed_boxes = process_result_boxes(pred_boxes, init_params, i)
        result.extend(processed_boxes)

    # combining the boxes
    result = combine_boxes(result, init_params["sliding_predict"]["iou_min"])
    return result


def process_result_boxes(pred_boxes, init_params, margin=0):
    result = []
    for ind in range(0, pred_boxes.size):
        box = {}
        if np.isnan(pred_boxes.arr[ind].left) or np.isnan(pred_boxes.arr[ind].top) \
                or np.isnan(pred_boxes.arr[ind].right) or np.isnan(pred_boxes.arr[ind].bottom):
            continue
        box['x1'] = int(pred_boxes.arr[ind].left)
        box['y1'] = int(pred_boxes.arr[ind].top) + margin
        box['x2'] = int(pred_boxes.arr[ind].right)
        box['y2'] = int(pred_boxes.arr[ind].bottom) + margin
        box['conf'] = pred_boxes.arr[ind].conf

        if "classID" in init_params:
            box['classID'] = init_params["classID"]
        else:
            box['classID'] = int(pred_boxes.arr[ind].class_num) + 1
        result.append(box)
    return result


# works correctly ONLY in case of one class detection
def calculate_medium_box(boxes):
    x1, y1, x2, y2, conf_sum = 0, 0, 0, 0, 0
    new_box = {}
    for box in boxes:
        x1 = x1 + box["x1"] * box["conf"]
        x2 = x2 + box["x2"] * box["conf"]
        y1 = y1 + box["y1"] * box["conf"]
        y2 = y2 + box["y2"] * box["conf"]
        conf_sum = conf_sum + box["conf"]
    new_box["x1"] = x1 / conf_sum
    new_box["x2"] = x2 / conf_sum
    new_box["y1"] = y1 / conf_sum
    new_box["y2"] = y2 / conf_sum
    new_box["classID"] = boxes[0]["classID"]
    return new_box


def combine_boxes(boxes, iou_min, verbose=False):
    neighbours, result = [], []
    for i, box in enumerate(boxes):
        cur_set = set()
        cur_set.add(i)
        for j, neigh_box in enumerate(boxes):
            if verbose:
                print(i, j, iou(box, neigh_box))
            if i != j and iou(box, neigh_box) > iou_min:
                cur_set.add(j)
        if not len(cur_set):
            result.append(box)

        if len(cur_set):
            for group in neighbours:
                if len(cur_set.intersection(group)):
                    neighbours.remove(group)
                    cur_set = cur_set.union(group)
            neighbours.append(cur_set)

    for group in neighbours:
        cur_boxes = [boxes[i] for i in group]
        medium_box = calculate_medium_box(cur_boxes)
        result.append(medium_box)

    return result


def predict_from_img(img, init_params, count):
    arr = np.array(img)
    h, w, c = arr.shape

    arr = np.transpose(arr, (1, 0, 2))
    arr = np.reshape(arr, w * h * c, order="F")
    data = arr.tolist()
    data = map(lambda x: x / 255.0, data)

    arr = (c_float * arr.size)(*data)
    ptr = cast(arr, POINTER(c_float))

    image = ImageYolo(c_int(h), c_int(w), c_int(c), ptr)
    pred_boxes = init_params['dll'].hot_predict_image(c_char_p(init_params["hypes_path"]), image, c_float(init_params['thresh']),
                                                      c_float(init_params['hier_thresh']), count)

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

    classes = ["background", "banner", "float_banner", "logo", "sitename", "menu", "navigation", "button", "file", "social", "socialGroups", "goods", "form", "search", "header", "text", "image", "video", "map", "table", "slider", "gallery"]
    save_results("19.jpg", "predictions_sliced.png", result, classes)

if __name__ == '__main__':
    main()
