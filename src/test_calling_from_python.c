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


void print_detections_to_file(image im, int num, float thresh, box *boxes, float **probs, int classes, int width_old, int height_old)
{
    int i;

    FILE *fp;
    fp = fopen("ds/result.json", "w");
    fprintf(fp, "[\n");

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            fprintf(fp, "  {\n    \"classID\": %d, \n", class);
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            double w_coeff = (double) width_old / im.w;
            double h_coeff = (double) height_old / im.h;

            // count positions on the old image
            double left_old = left * w_coeff;
            double right_old = right * w_coeff;
            double top_old = top * h_coeff;
            double bot_old = bot * h_coeff;

            // check valid positions
            assert(left_old < width_old);
            assert(right_old < width_old);
            assert(top_old < height_old);
            assert(bot_old < height_old);

            fprintf(fp, "    \"x1\": %f, \n", left_old);
            fprintf(fp, "    \"y1\": %f, \n", top_old);
            fprintf(fp, "    \"x2\": %f, \n", right_old);
            fprintf(fp, "    \"y2\": %f\n", bot_old);

            fprintf(fp, "  },\n");
        }
    }
    // deletion last comma after curly bracket
    if (num > 2) {
        fseek(fp, -2, SEEK_END);
    }
    fprintf(fp, "\n]");
    fclose(fp);
}

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

  res_b.left = (left - config.start_x) * config.w_coeff;
  res_b.right = (right - config.start_x) * config.w_coeff;
  res_b.top = (top - config.start_y) * config.h_coeff;
  res_b.bottom = (bottom - config.start_y) *config.h_coeff;

  if ((int)res_b.left == (int)res_b.right || (int)res_b.top == (int)res_b.bottom){
    return  (result_box){-1, -1, -1, -1, -1, -1};
  }

  res_b.class_num = class;
  res_b.conf = prob;

  return res_b;
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

    // rescaling
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

    int count = 0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];
            res.pred_boxes[count] = transform_box(b, config, class, prob);

            // printf("start_x = %d, pred = %f, right = %f, not_mod_left = %d, not_mod_right= %d, max = %d\n", start_x, res.pred_boxes[count].left, res.pred_boxes[count].right, left, right, width_old);
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

    /*   for debugging   */
    // size_t free_byte;
    // size_t total_byte;
    // cudaMemGetInfo( &free_byte, &total_byte );
    // double free_db = (double)free_byte;
    // double total_db = (double)total_byte;
    // double used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    current_network = parse_network_cfg_param(cfgfile, grid_parameters);
    if(weightfile){
        load_weights(&current_network, weightfile);
    }
    set_batch_network(&current_network, 1);

    /*   for debugging   */
    // cudaMemGetInfo( &free_byte, &total_byte );
    // free_db = (double)free_byte;
    // total_db = (double)total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);


    network_created = 1;
}

// float ** calculate_map_of_probabilities(layer l, float **probs, box *boxes, int num, int classes, int width_old, int height_old, int width_resized, int height_resized) {
//     // num = n*w*h
//
//     int i, j;
//
//     double w_coeff = (double) width_old / width_resized;
//     double h_coeff = (double) height_old / height_resized;
//
//     int start_x, start_y;
//     if (((float)im.w/width_old) < ((float)im.h/height_old)) {  // если рескейл по ширине
//         start_x = 0;
//         start_y = (im.h - height_resized) / 2;
//     } else { // если рескейл по высоте (наш обычный случай)
//         start_x = (im.w - width_resized) / 2;
//         start_y = 0;
//     }
//
//     for(i = 0; i < num; ++i){
//         for (j = 0; j < classes; ++j)
//             float prob = probs[i][j];
//
//
//         // вынести в отдельную функцию подсчет боксов!!!
//             box b = boxes[i];
//
//             int left  = (b.x-b.w/2.)*im.w;
//             int right = (b.x+b.w/2.)*im.w;
//             int top   = (b.y-b.h/2.)*im.h;
//             int bot   = (b.y+b.h/2.)*im.h;
//
//
//             // printf("\n %f %f %d %d %d %d \n\n ", thresh, prob, left, top, right, bot);
//             int end_x = start_x + width_resized - 1;
//             int end_y = start_y + height_resized - 1;
//
//             if (left > end_x || right < start_x || top > end_y || bot < start_x)
//                 continue;
//
//             if (left < start_x) left = start_x;
//             if(right > end_x) right = end_x;
//             if(top < start_y) top = start_y;
//             if(bot > end_y) bot = end_y;
//
//             // printf("\n %d %d %f %f \n\n ", width_resized, height_resized, w_coeff, h_coeff);
//
//             // printf("\n %d %d %d %d, last_right = %d, \n\n ", left, top, right, bot, (start_x + width_resized) - 1);
//
//             // count positions on the old image
//             int real_left = (int)((left - start_x) * w_coeff);
//             int real_right = (int)((right - start_x) * w_coeff);
//             int real_top = (int)((top - start_y) * h_coeff);
//             int real_bottom = (int)((bot - start_y) * h_coeff);
//
//             // printf(" l= %f, r = %f, t = %f, b = %f\n", res.pred_boxes[count].left, res.pred_boxes[count].right, res.pred_boxes[count].top, res.pred_boxes[count].bottom);
//
//     }
// }


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

//     strncpy(input, filename, 256);

//     image im = load_image_color(input,0,0);
//     int old_width = im.w;
//     int old_height = im.h;
//     int width_resized, height_resized;
//     // printf("old_width = %d, old_height = %d\n", old_width, old_height);
//     image sized = letterbox_image_with_info(im, net.w, net.h, &width_resized, &height_resized);
//     // printf("new_width = %d, new_height = %d\n", width_resized, height_resized);
//     layer l = net.layers[net.n-1];

//     box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
//     float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
//     for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

//     float *X = sized.data;
//     network_predict(net, X);
//     // printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
//     get_region_boxes(l, 1, 1, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);

//     float ** map_prob = calculate_map_of_probabilities();
//     free_image(im);
//     free_image(sized);
//     free(boxes);
//     free_ptrs((void **)probs, l.w*l.h*l.n);

//     return map_prob;
// }



result_box_arr hot_predict(char *datacfg, char *filename, float thresh, float hier_thresh) {
    network net;
    if (network_created)
        net = current_network;
    else {
        printf("network isn't initialized!\n");
        exit(1);
    }
    // printf("%s %s %f %f\n", datacfg, filename, thresh, hier_thresh);
    srand(2222222);
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;

    strncpy(input, filename, 256);

    image im = load_image_color(input,0,0);
    int old_width = im.w;
    int old_height = im.h;
    int width_resized, height_resized;
    // printf("old_width = %d, old_height = %d\n", old_width, old_height);
    image sized = letterbox_image_with_info(im, net.w, net.h, &width_resized, &height_resized);
    // printf("new_width = %d, new_height = %d\n", width_resized, height_resized);
    layer l = net.layers[net.n-1];

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;
    network_predict(net, X);
    // printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    get_region_boxes(l, 1, 1, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    result_box_arr res = result_detection(sized, l.w*l.h*l.n, thresh, boxes, probs, l.classes, old_width, old_height, width_resized, height_resized);
    // print_detections_to_file(sized, l.w*l.h*l.n, thresh, boxes, probs, l.classes, old_width, old_height);

                 // for drawing bboxes on the pictures

    // list *options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", "data/names.list");
    // char **names = get_labels(name_list);

    // image **alphabet = load_alphabet();
    // draw_detections(sized, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
    // save_image(sized, "predictions");
    // show_image(sized, "predictions");

    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    return res;
}


result_box_arr hot_predict_image(char *datacfg, image im, float thresh, float hier_thresh, int count) {
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

    int old_width = im.w;
    int old_height = im.h;
    int width_resized, height_resized;
    // printf("old_width = %d, old_height = %d\n", old_width, old_height);
    image sized = letterbox_image_with_info(im, net.w, net.h, &width_resized, &height_resized);
    // printf("new_width = %d, new_height = %d\n", width_resized, height_resized);
    layer l = net.layers[net.n-1];

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

    float *X = sized.data;
    network_predict(net, X);
    // printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    get_region_boxes(l, 1, 1, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    result_box_arr res = result_detection(sized, l.w*l.h*l.n, thresh, boxes, probs, l.classes, old_width, old_height, width_resized, height_resized);
    // print_detections_to_file(sized, l.w*l.h*l.n, thresh, boxes, probs, l.classes, old_width, old_height);

                 // for drawing bboxes on the pictures

    // list *options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", "data/names.list");
    // char **names = get_labels(name_list);

    // image **alphabet = load_alphabet();
    // draw_detections(sized, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

    // char count_str[15];
    // sprintf(count_str, "%d", count);
    // char filename[256] = "predictions_";
    // strcat(filename, count_str);

    // save_image(sized, filename);
    // show_image(sized, "predictions");

    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);

    return res;
}

void freeme(float * ptr)
{
    printf("freeing address: %p\n", ptr);
    free(ptr);
}
