#include "network.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#include "classifier.h"
#include "option_list.h"
#include "test_calling_from_python.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

network current_network;
int network_created = 0;

result_box transform_box(box b, box_transform_param config, int class, int prob) {
  result_box res_b;

  int left  = (b.x-b.w/2.)*config.img_width;
  int right = (b.x+b.w/2.)*config.img_width;
  int top   = (b.y-b.h/2.)*config.img_height;
  int bottom   = (b.y+b.h/2.)*config.img_height;

  // printf("\n %f %d %d %d %d \n\n ", prob, left, top, right, bottom);

  if (left > config.end_x || right < config.start_x || top > config.end_y
    || bottom < config.start_x) {
      return (result_box){-1, -1, -1, -1, -1, -1};
  }

  if (left < config.start_x) left = config.start_x;
  if(right > config.end_x) right = config.end_x;
  if(top < config.start_y) top = config.start_y;
  if(bottom > config.end_y) bottom = config.end_y;

  res_b.left = (int)((left - config.start_x) * config.w_coeff);
  res_b.right = (int)((right - config.start_x) * config.w_coeff);
  res_b.top = (int)((top - config.start_y) * config.h_coeff);
  res_b.bottom = (int)((bottom - config.start_y) *config.h_coeff);

  if (res_b.left == res_b.right || res_b.top == res_b.bottom){
    return  (result_box){-1, -1, -1, -1, -1, -1};
  }

  res_b.class_num = class;
  res_b.conf = prob;

  return res_b;
}


box_transform_param box_transform_param_calculation(image im, int width_old, int height_old, int width_resized, int height_resized){
  int start_x, start_y;
  if (((float)im.w/width_old) < ((float)im.h/height_old)) {  // width rescaling
      start_x = 0;
      start_y = (im.h - height_resized) / 2;
  } else { // height rescaling (it is a more common situation)
      start_x = (im.w - width_resized) / 2;
      start_y = 0;
  }

  int end_x = start_x + width_resized - 1;
  int end_y = start_y + height_resized - 1;
  double w_coeff = (double) width_old / width_resized;
  double h_coeff = (double) height_old / height_resized;

  box_transform_param config = {start_x, start_y, end_x, end_y, im.w, im.h,
                                w_coeff, h_coeff};
  return config;
}


result_box_arr result_detection(image im, int num, float thresh, box *boxes, float **probs, int classes, int width_old, int height_old, int width_resized, int height_resized)
{
    // helper part for python - just counting the size of the array
    int i;
   	int arr_size = 0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
        	arr_size++;
        }
    }
    result_box_arr res;
    res.pred_boxes = (result_box *)malloc(arr_size * sizeof(result_box));
    res.size = arr_size;

    box_transform_param config = box_transform_param_calculation(im, width_old, height_old, width_resized, height_resized);
    int count = 0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];
            res.pred_boxes[count] = transform_box(b, config, class, prob);

            assert(res.pred_boxes[count].left < width_old);
            assert(res.pred_boxes[count].right < width_old);
            assert(res.pred_boxes[count].top < height_old);
            assert(res.pred_boxes[count].bottom < height_old);
            ++count;
        }
    }
    return res;
}


void initialize_network_test(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    network_created = 1;
    current_network = net;
}


void initialize_network_test_param(char *cfgfile, char *weightfile, cfg_param grid_parameters)
{
    if (network_created) {
        free_network(current_network);
    }
    current_network = parse_network_cfg_param(cfgfile, grid_parameters);
    if(weightfile){
        load_weights(&current_network, weightfile);
    }
    set_batch_network(&current_network, 1);
    network_created = 1;
}


float * calculate_map_of_probabilities(layer l, float **probs, image im, box *boxes, int num_anchors, int classes, int width_old, int height_old, int width_resized, int height_resized) {
    int i, j;
    box_transform_param config = box_transform_param_calculation(im, width_old, height_old, width_resized, height_resized);
    float * map = calloc(width_old * height_old, sizeof(float));
    // initialization
    for(i = 0; i < width_old; ++i){
      for(j = 0; i < height_old; ++j){
        map[j * width_old + i] = 0;
      }
    }

    for(i = 0; i < num_anchors; ++i){
        for (j = 0; j < classes; ++j) {
          int class = max_index(probs[i], classes);  // need checking !!!
          float prob = probs[i][j];
          box b = boxes[i];
          result_box cur_box = transform_box(b, config, class, prob);

          int idx_w, idx_h;
          for (idx_w = cur_box.left; idx_w < cur_box.right; ++idx_w){
            for (idx_h = cur_box.top; idx_h < cur_box.bottom; ++idx_h){
              int cur_pos = idx_h * width_old + idx_w;
              // intersection of events
              map[cur_pos] += (prob - map[cur_pos] * prob);
            }
          }
        }
      }

    return map;
}


// float ** map_of_probabilities(char *datacfg, char *filename, float thresh, float hier_thresh) {
//     network net;
//     if (network_created)
//         net = current_network;
//     else {
//         printf("network isn't initialized!\n");
//         exit(1);
//     }
//     // printf("%s %s %f %f\n", datacfg, filename, thresh, hier_thresh);
//     srand(2222222);
//     char buff[256];
//     char *input = buff;
//     int j;
//     float nms=.4;
//
//     strncpy(input, filename, 256);
//
//     image im = load_image_color(input,0,0);
//     int old_width = im.w;
//     int old_height = im.h;
//     int width_resized, height_resized;
//     // printf("old_width = %d, old_height = %d\n", old_width, old_height);
//     image sized = letterbox_image_with_info(im, net.w, net.h, &width_resized, &height_resized);
//     // printf("new_width = %d, new_height = %d\n", width_resized, height_resized);
//     layer l = net.layers[net.n-1];
//
//     box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
//     float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
//     for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
//
//     float *X = sized.data;
//     network_predict(net, X);
//     // printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
//     get_region_boxes(l, 1, 1, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
//
//     float ** map_prob = calculate_map_of_probabilities();
//     free_image(im);
//     free_image(sized);
//     free(boxes);
//     free_ptrs((void **)probs, l.w*l.h*l.n);
//
//     return map_prob;
// }



result_box_arr hot_predict(char *datacfg, char *filename, image part_im, float thresh, float hier_thresh, int from_image) {
    network net;
    if (network_created)
        net = current_network;
    else {
        printf("network isn't initialized!\n");
        exit(1);
    }

    srand(2222222);
    int j;
    float nms=.4;
    image im;

    if (from_image == 1) {
      im = part_im;
    }
    else {
      char buff[256];
      char *input = buff;
      strncpy(input, filename, 256);
      im = load_image_color(input, 0, 0);
    }

    int old_width = im.w;
    int old_height = im.h;
    int width_resized, height_resized;
    image sized = letterbox_image_with_info(im, net.w, net.h, &width_resized, &height_resized);
    layer l = net.layers[net.n-1];

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;
    network_predict(net, X);
    get_region_boxes(l, 1, 1, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    result_box_arr res = result_detection(sized, l.w*l.h*l.n, thresh, boxes, probs, l.classes, old_width, old_height, width_resized, height_resized);

    if (from_image != 1) {
      free_image(im);
    }
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    return res;
}
