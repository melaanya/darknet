from __future__ import division
from ctypes import *
import sys, json, subprocess
import numpy as np
import random
from os import path
from optparse import OptionParser
from PIL import Image, ImageDraw

if __package__ is None:
    import sys
    sys.path.append(path.abspath(path.join(path.dirname(__file__), path.pardir)))

def check_network(dll):
    return c_int.in_dll(dll, 'network_created')

def save_results(src_path, dst_path, rects, classes):
    '''Saves results of the prediction.

    Args:
        src_path (string): The path to source image to predict bounding boxes.
        dst_path (string): The path to source image to predict bounding boxes.
        rects (list): The collection of boxes to draw on screenshot.
        classes (list): The collection of classes corresponding their ids 

    Returns: 
        Nothing.
    '''

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
        ('left', c_int),
        ('top', c_int),
        ('right', c_int),
        ('bottom', c_int),
        ('class_num', c_int),
        ('conf', c_float)]


class BoxArray(Structure):
    _fields_ = [
    ('arr', POINTER(Box)),
    ('size', c_int)]


class GridParam(Structure):
    _fields_ = [
        ('width', c_int),
        ('height', c_int)]


class ImageYolo(Structure):
    _fields_ = [
        ('h', c_int),
        ('w', c_int),
        ('c', c_int),
        ('data', POINTER(c_float))]


def segment_intersection(rect1, rect2, coord_name):
    one = coord_name + '1'
    two = coord_name + '2'
    d = min(rect1[two], rect2[two]) - max(rect1[one], rect2[one])
    return d if d > 0 else 0


def area(rect):
    return (rect['x2'] - rect['x1']) * (rect['y2'] - rect['y1'])


def intersection_area(rect1, rect2):
    return segment_intersection(rect1, rect2, 'x') * segment_intersection(rect1, rect2, 'y')


def detailed_iou(rect1, rect2):
    intersection = intersection_area(rect1, rect2)
    union = area(rect1) + area(rect2) - intersection
    if union != 0:
        return intersection, union, intersection / float(union)
    else:
        print("Something went wrong!!!!")
        exit(1)


def iou(rect1, rect2):
    _, _, val = detailed_iou(rect1, rect2)
    return val


def initialize(weights_path, hypes_path, options=None):
    """ initialization cover function for initialization function written in C

        Args:
            weights_path (string): the path to weights (for example: 'backup/yolo-my_final.weights')
            hypes (string): the path to parameters file (hypes.json)
            options (dict): additional parameters

        Returns (dict): 
            initialization parameters for hot_predict
    """
    config = json.load(open(hypes_path, 'r'))

    # fields from options are added and they overwrite fields in hypes if the same
    config.update(options)

    param_folder = path.dirname(path.abspath(hypes_path))

    if 'yolo' not in config:
        print('model description required (for example cfg/yolo-my-test.cfg ')
        sys.exit()

    mydll = cdll.LoadLibrary(path.join(param_folder, config['yolo']['so_library_path']))
    if 'pred_options' in config and 'cfg_width' in config['pred_options']:
        grid_parameters = GridParam(config['pred_options']['cfg_width'], config['pred_options']['cfg_width'])
        mydll.initialize_network_test_param(str(config['yolo']['model_def_path']), str(weights_path), grid_parameters)
    elif 'cfg_width' in config['yolo']:
        if 'verbose' in config and config['verbose']:
            print('changed cfg')
        grid_parameters = GridParam(config['yolo']['cfg_width'], config['yolo']['cfg_width'])  # make square as we should save aspect ratio
        mydll.initialize_network_test_param(str(path.join(param_folder, config['yolo']['model_def_path'])), str(path.join(param_folder, weights_path)), grid_parameters)
    else:
        mydll.initialize_network_test(path.join(param_folder, str(config['yolo']['model_def_path'])), path.join(param_folder, str(weights_path)))

    result = {'dll' : mydll, 'thresh': config['yolo']['thresh'], 'hier_thresh': config['yolo']['hier_thresh'],
              'sliding_predict': config['sliding_predict']}

    if 'classID' in config:
        result.update({'classID': config['classID']})
    return result


def hot_predict(image_path, init_params, verbose=False):
    """ gets prediction in json format for one image
        uses two helper-functions^ regular_predict and sliding_predict
        each of them calls the function written in c and stored in the .so library
        and makes specific pre- and postprocessing

        Args:
            image_path (string): the path to image for prediction
            init_params (dict): prediction parameters
            verbose (bool): whether to allow writing additional information to console

        Returns (dict): 
            result of prediction in json format
    """

    if check_network(init_params['dll']):

        # to fix change names of paths in file
        # image_path = '..' + image_path
        init_params['dll'].hot_predict.restype = BoxArray
        if verbose:
            print(image_path)
        if not path.exists(image_path):
            print('danger! path does not exist! \n')
            exit(1)

        if 'pred_options' in init_params:
            init_params['thresh'] = init_params['pred_options']['thresh']
        if 'sliding_predict' in init_params and 'sliding_window' in init_params['sliding_predict'] and init_params['sliding_predict']['sliding_window']:
            if verbose:
                print('sliding')
            return sliding_predict(image_path, init_params)
        else:
            if verbose:
                print('regular')
            return regular_predict(image_path, init_params)

    else:
        print('no initialisation of network happened before')
        sys.exit()


def regular_predict(image_path, init_params):
    """ helper function - wrapper for calling hot_predict in c for the image without sliding_window

        Args:
            image_path (string): the path to image for prediction
            init_params (dict): prediction parameters

        Returns (dict): 
            result of prediction in json format
    """
    empty_image = ImageYolo(c_int(0), c_int(0), c_int(0), pointer(c_float(0)))
    from_image = 0
    pred_boxes = init_params['dll'].hot_predict(c_char_p(image_path), empty_image, c_float(init_params['thresh']),
                                                c_float(init_params['hier_thresh']), c_int(from_image))

    return process_result_boxes(pred_boxes, init_params)


def sliding_predict(image_path, init_params):
    """ helper function - wrapper for calling hot_predict in c for the image with sliding_windows

        Args:
            image_path (string): the path to image for prediction
            init_params (dict): prediction parameters

        Returns (dict): 
            result of prediction in json format
    """
    img = Image.open(image_path)
    width, height = img.size

    assert(init_params['sliding_predict']['step'] > init_params['sliding_predict']['overlap'])

    # cutting
    result = []
    reached_end = False
    for idx, i in enumerate(range(0, height, init_params['sliding_predict']['step'] - init_params['sliding_predict']['overlap'])):
        top, bottom = i, min(height, i + init_params['sliding_predict']['step'])
        if (height <= i + init_params['sliding_predict']['step']):
            top = bottom - init_params['sliding_predict']['step']
            reached_end = True
        img_part = img.crop((0, top, width, bottom))
        pred_boxes = predict_from_img(img_part, init_params)
        processed_boxes = process_result_boxes(pred_boxes, init_params, top)
        result.extend(processed_boxes)
        if reached_end:
            break

    # combining the boxes
    nms = init_params['sliding_predict']['nms'] if 'nms' in init_params['sliding_predict'] else False
    result = combine_boxes(result, init_params['sliding_predict']['iou_min'], nms)
    return result


def process_result_boxes(pred_boxes, init_params, margin=0):
    """ transforming received bounding boxes from ctypes format to json format

        Args:
            pred_boxes (list): predicted boxes for the image
            init_params (dict): prediction parameters
            margin(int): number of pixels to add to the box['y1'] and box['y2'] 
                        in case of sliding_predict

        Returns (dict): 
            result of prediction in json format
    """

    result = []
    for ind in range(0, pred_boxes.size):
        box = {}
        if np.isnan(pred_boxes.arr[ind].left) or np.isnan(pred_boxes.arr[ind].top) \
                or np.isnan(pred_boxes.arr[ind].right) or np.isnan(pred_boxes.arr[ind].bottom):
            continue
        box['x1'] = pred_boxes.arr[ind].left
        box['y1'] = pred_boxes.arr[ind].top + margin
        box['x2'] = pred_boxes.arr[ind].right
        box['y2'] = pred_boxes.arr[ind].bottom + margin
        box['conf'] = pred_boxes.arr[ind].conf

        if 'classID' in init_params:
            box['classID'] = init_params['classID']
        else:
            box['classID'] = int(pred_boxes.arr[ind].class_num) + 1
        result.append(box)
    return result


# works correctly ONLY in case of one class detection
def calculate_medium_box(boxes):
    """ medium box calculation (possible replacement for nms)
    from group boxes it calculates only one medium box 

        Args:
            boxes(list): group of boxes to combine

        Returns (dict): 
            one medium box
    """
    x1, y1, x2, y2, conf_sum = 0, 0, 0, 0, 0
    new_box = {}
    for box in boxes:
        x1 = x1 + box['x1'] * box['conf']
        x2 = x2 + box['x2'] * box['conf']
        y1 = y1 + box['y1'] * box['conf']
        y2 = y2 + box['y2'] * box['conf']
        conf_sum = conf_sum + box['conf']
    new_box['x1'] = x1 / conf_sum
    new_box['x2'] = x2 / conf_sum
    new_box['y1'] = y1 / conf_sum
    new_box['y2'] = y2 / conf_sum
    new_box['classID'] = boxes[0]['classID']
    return new_box


def non_maximum_suppression(boxes):
    conf = [box['conf'] for box in boxes]
    ind = np.argmax(conf)
    if isinstance(ind, int):
        return boxes[ind]
    else:
        random.seed()
        num = random.randint(0, len(ind))
        return boxes[num]


def combine_boxes(boxes, iou_min, nms, verbose=False):
    """ creates groups of boxes (according to their intersection) and later leaves only one box from each group

        Args:
            boxes(list): group of boxes to combine
            iou_min(double): min iou considered to count boxes from one group
            nms (bool): if True - nms used for choosing one box
                        else calculate_medium_box is used
            verbose(bool): verbose (bool): whether to allow writing additional information to console

        Returns (list): 
            list of boxes where boxes are all from different groups
    """
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
        if nms:
            medium_box = non_maximum_suppression(cur_boxes)
        else:
            medium_box = calculate_medium_box(cur_boxes)
        result.append(medium_box)

    return result


def predict_from_img(img, init_params):
    """ helper function which wraps prediction from img (not from image_path) in C
    
        Args:
            img(PIL Image): piece of image in PIL formatting
            init_params (dict): prediction parameters

        Returns (list): 
            list of boxes in ctypes format
    """
    arr = np.array(img)
    h, w, c = arr.shape

    arr = np.transpose(arr, (1, 0, 2))
    arr = np.reshape(arr, w * h * c, order='F')
    data = arr.tolist()
    data = map(lambda x: x / 255.0, data)

    arr = (c_float * arr.size)(*data)
    ptr = cast(arr, POINTER(c_float))

    image = ImageYolo(c_int(h), c_int(w), c_int(c), ptr)
    from_image = 1
    pred_boxes = init_params['dll'].hot_predict(c_char_p(""), image, c_float(init_params['thresh']),
                                                      c_float(init_params['hier_thresh']), c_int(from_image))

    del arr
    return pred_boxes


def main():
    parser = OptionParser(usage='usage: %prog [options] <image_path> <weights> <hypes>')
    _, args = parser.parse_args()

    if len(args) < 3:
        print('Provide image path, weights and hypes')
        return

    if not path.exists(args[2]) or not path.isfile(args[2]):
        print ('Bad hypes path provided!')
        return

    config = json.load(open(args[2], "r"))

    init_params = initialize(args[1], args[2], {})
    result = hot_predict(args[0], init_params)
    print(result)

    classes = ['background', 'banner', 'float_banner', 'logo', 'sitename', 'menu', 'navigation', 'button', 'file', 'social', 'socialGroups', 'goods', 'form', 'search', 'header', 'text', 'image', 'video', 'map', 'table', 'slider', 'gallery']
    save_results(args[0], 'predictions.png', result, classes)

if __name__ == '__main__':
    main()
