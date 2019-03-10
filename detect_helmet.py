import sys
import os.path
import argparse
import cv2 as cv
import numpy as np
##modules by self
import draw_tool
import nms
import time

def get_class_name(obj_file):
    with open(obj_file, 'r') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def init_model(model_layers, model_weights):
    net = cv.dnn.readNetFromDarknet(model_layers, model_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

def get_out_layer_name(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def forward_net(img, net):
    ##Create a 4D blob from a img.
    blob = cv.dnn.blobFromImage(img, 1/255, (416, 416), [0,0,0], 1, crop=False)
    net.setInput(blob)
    ##Run the forward pass to get output of the output layers
    outs = net.forward(get_out_layer_name(net))

    return outs

if __name__ == '__main__':
    ##get args from shell
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_layers')##model .pt file
    parser.add_argument('--model_weights')##trained model weights
    parser.add_argument('--class_file')##file saves class can be detected
    parser.add_argument('--src_type', choices=['video', 'image'])
    parser.add_argument('--input_file')##video to proc
    parser.add_argument('--conf_thr')##class confidence threshold'
    parser.add_argument('--nms_thr')##NMS threshold

    args = parser.parse_args()
    ##load class name
    class_list = get_class_name(args.class_file)
    ##load model configuration and weight
    detect_net = init_model(args.model_layers, args.model_weights)
    ##use queue to save heads without helmet. max dim is 3
    head_queue = []
    all_head = []
    ##three boxes to disp head without helmet
    DISP_BOXES = [(10,5,110,105), (10,125,110,225), (10,245,110,345)]
    ##proc the video
    if args.src_type == 'video':
        capture = cv.VideoCapture(args.input_file)
        while(1):
            ##obtain a frame
            result, frame = capture.read()
            t = time.clock()
            if result == True:
                ##detect this frame
                detect_result = forward_net(frame, detect_net)
                ##perform nms.box:(left, top, width, height)
                person_boxes = nms.nms(frame, detect_result, args.conf_thr, args.nms_thr, class_list)
                ##resize the box to suit real people.box:(left, top, width, height)
                resized_boxes = draw_tool.resize_box(person_boxes)
                ##update the head queue
                head_queue, all_head = draw_tool.push_head(frame, resized_boxes, head_queue, all_head)
                ##draw head in head_queue on img
                frame = draw_tool.draw_head(frame, head_queue, DISP_BOXES)
                ##draw person box
                frame = draw_tool.draw_box(frame, resized_boxes)
                #t, _ = detect_net.getPerfProfile()
                #print(t)
                ##draw the time(ms) at the mid
                frame = draw_tool.draw_time(frame)

                cv.imshow("video",frame)
                print(time.clock()-t)
                fps = capture.get(cv.CAP_PROP_FPS)
                if cv.waitKey(int(fps))&0xFF==ord('q'):
                    break
            else:
                break
        capture.release()
        cv.destroyAllWindows()