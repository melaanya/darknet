nvidia-docker run -it \
 -v /home/anna/dev/darknet:/darknet \
-v /home/anna/data:/darknet/data \
-v /home/anna/dev/detect-widgets:/darknet/detect-widgets \
yolo /bin/bash