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


result_box_arr result_detection(image im, int num, float thresh, box *boxes, float **probs, int classes, int width_old, int height_old, int width_resized, int height_resized)
{
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
    if (((float)im.w/width_old) < ((float)im.h/height_old)) {  // если рескейл по ширине
        start_x = 0;
        start_y = (im.h - height_resized) / 2;
    } else { // если рескейл по высоте (наш обычный случай)
        start_x = (im.w - width_resized) / 2;
        start_y = 0;
    }

    double w_coeff = (double) width_old / width_resized;
    double h_coeff = (double) height_old / height_resized;


    int count = 0;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;


            // printf("\n %f %f %d %d %d %d \n\n ", thresh, prob, left, top, right, bot);
            int end_x = start_x + width_resized - 1;
            int end_y = start_y + height_resized - 1;

            if (left > end_x || right < start_x || top > end_y || bot < start_x) 
                continue;

            if (left < start_x) left = start_x;
            if(right > end_x) right = end_x;
            if(top < start_y) top = start_y;
            if(bot > end_y) bot = end_y;

            if (left == right || top == bot)
                continue;

            // printf("\n %d %d %f %f \n\n ", width_resized, height_resized, w_coeff, h_coeff);

            // printf("\n %d %d %d %d, last_right = %d, \n\n ", left, top, right, bot, (start_x + width_resized) - 1);

            // count positions on the old image
            res.pred_boxes[count].left = (left - start_x) * w_coeff; 
            res.pred_boxes[count].right = (right - start_x) * w_coeff;
            res.pred_boxes[count].top = (top - start_y) * h_coeff;
            res.pred_boxes[count].bottom = (bot - start_y) * h_coeff;

            // printf(" l= %f, r = %f, t = %f, b = %f\n", res.pred_boxes[count].left, res.pred_boxes[count].right, res.pred_boxes[count].top, res.pred_boxes[count].bottom);

            res.pred_boxes[count].class_num = class;

            // check valid positions

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
    // network * net = (network *)malloc(sizeof(network));
    // *net = parse_network_cfg_param(cfgfile, grid_parameters);
    // if(weightfile){
    //     load_weights(net, weightfile);
    // }
    // set_batch_network(net, 1);
    // if (network_created)
    //     free(current_network);

    // network_created = 1;
    // current_network = net;

    network net = parse_network_cfg_param(cfgfile, grid_parameters);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    
    network_created = 1;
    current_network = net;
}


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
    clock_t time;
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
    time=clock();
    network_predict(net, X);
    // printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
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