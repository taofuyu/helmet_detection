#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 detect_helmet.py \
    --src_type=video \
    --model_layers=./model/yolov3-tiny-per.cfg \
    --model_weights=./model/yolov3-tiny-per.weights \
    --input_file=./videos/video1.avi \
    --conf_thr=0.5 \
    --nms_thr=0.8 \
    --class_file=./model/obj.names
