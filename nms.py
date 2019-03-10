##implement the nms

import cv2 as cv
import draw_tool
import numpy as np

def nms(frame, outs, confThreshold, nmsThreshold, class_list):
    confThreshold = float(confThreshold)
    nmsThreshold = float(nmsThreshold)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    confidences = []
    boxes = []          
    confidences = []
    classIds = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            #print int(detection[0] * frameWidth)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                classIds.append(classId)

    ##perform nms in opencv
    indexes = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    person_boxes = []
    for i in indexes:
        i = i[0]
        box=boxes[i]
        ##check if the detected class is helmet or person     
        detected_class = class_list[classId]
        if detected_class == 'Person':
            person_boxes.append(box)

    ##return all the boxes and indexes represent person box
    return person_boxes


        