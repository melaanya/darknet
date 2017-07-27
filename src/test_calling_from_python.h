#ifndef TEST_CALLING_FROM_PYTHON_H
#define TEST_CALLING_FROM_PYTHON_H

#include "network.h"
#include "parser.h"

typedef struct{
    int left, top, right, bottom;
    int class_num;
    float conf;
} result_box;

typedef struct{
    result_box * pred_boxes;
    int size;
} result_box_arr;

typedef struct{
	int width;
	int height;
} cfg_param;

typedef struct{
  int start_x, start_y, end_x, end_y;
  int img_width, img_height;
  float w_coeff, h_coeff;
} box_transform_param;


// void print_detections_to_file(image im, int num, float thresh, box *boxes, float **probs, int classes, int width_old, int height_old);
void initialize_network_test(char *cfgfile, char *weightfile);
void initialize_network_test_param(char *cfgfile, char *weightfile, cfg_param grid_parameters);
result_box_arr hot_predict(char *filename, image part_im, float thresh, float hier_thresh, int from_image);
float * calculate_map_of_probabilities(image im, box *boxes, float **probs, int num_anchors,
              int classes, int width_old, int height_old, int width_resized, int height_resized);


#endif
