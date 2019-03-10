##this toolbox is used to draw bounding box / system time / head without helmet

import cv2 as cv
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import compare_ssim

##def colour
WHITE = (255, 255, 255)
RED = (0, 0, 255)

##draw bounding box by the resized coordinates
def draw_box(img, boxes):
        img_width = img.shape[1]
        img_height = img.shape[0]
        for box in boxes:
                left   = box[0]
                top    = box[1]
                right  = box[0] + box[2]
                bottom = box[1] + box[3]
                ##check the detected point whether inside the img
                if left<0 or right<0 or top<0 or bottom<0:
                        print('invalid box coordinate')

                ##set the box style
                pen_width = 1
                cv.rectangle(img, (left, top), (right, bottom), WHITE, pen_width)

        return img

##display the current time at the middle of img
def draw_time(img):
        ##get current time (ms)
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        
        ##set the text style. use PIL to set the str at the mid of img
        blank = Image.new("RGB",[640,320],"white")##random tmp img
        draw = ImageDraw.Draw(blank)
        (str_w, str_h) = draw.textsize(time_stamp)##the pixel w,h of time str
        img_x_mid = img.shape[1]/2
        draw_point = (int(img_x_mid - str_w/2), 25)
                
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, time_stamp, draw_point, font, 0.5, WHITE, 1)

        del draw
        del blank
        return img

##resize the detected bounding box cooedinates
def resize_box(boxes):
        ##left should be small, top should be large
        left_point_scale = 0.97
        top_point_scale = 0.85
        resized_box = []
        for box in boxes:
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                new_left   = int(left*left_point_scale)
                new_top    = int(top*top_point_scale)
                new_width  = int(width + 2*(left-new_left))
                new_height = int(12*height)
                resized_box.append([new_left, new_top, new_width, new_height])

        return resized_box

##obtain the 20% height of the box
def get_upper_box(box):
        box[3] = round(box[3] * 0.2)

        return box

##obtain the head region 
def get_head_img(img, box):
        box = get_upper_box(box)
        #head_img = img[box[0]:(box[0]+box[2]), box[1]:(box[1]+box[3])] ##[xstart:xend ystart:yend]
        img_PIL = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        head_img_PIL = img_PIL.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
        ##cut the side around the head region, leave a square
        if is_large_enough(head_img_PIL):
                left = round(head_img_PIL.width * 0.3)
                top  = round(head_img_PIL.height * 0.1)
                right = left + 20
                bottom = top + 20

                head_img_PIL = head_img_PIL.crop((left, top, right, bottom))
                head_img_PIL = head_img_PIL.resize((100,100))
                return head_img_PIL
        else:
                return []

##for head region in each box, decide whether put in the head queue
def push_head(img, resized_box, head_queue, all_head):
        for box in resized_box:
                ##head_img_PIL:the top region of img and resize to 40*40
                head_img_PIL = get_head_img(img, box)
                if head_img_PIL and is_not_helmet(head_img_PIL) and is_first_appear(head_img_PIL, all_head):
                        if len(head_queue) < 3:
                                head_queue.append(head_img_PIL)
                        elif len(head_queue) == 3:
                                head_queue[0] = head_queue[1]
                                head_queue[1] = head_queue[2]
                                head_queue[2] = head_img_PIL
                        all_head.append(head_img_PIL)
        return head_queue, all_head

##decide if the crop img is large enough to avoid procing the too far img
def is_large_enough(head_img_PIL):
        if head_img_PIL.height >= 30:
                return True
        else:
                return False

##for each head img, compare it with a list(save all appeared head)
def is_first_appear(head_img_PIL, all_head):
        if len(all_head) == 0:
                return True
        for i in range(len(all_head)):
                gray_head_img_cv   = cv.cvtColor(np.asarray(head_img_PIL),cv.COLOR_RGB2GRAY)
                gray_all_head_i_cv = cv.cvtColor(np.asarray(all_head[i]),cv.COLOR_RGB2GRAY)
                ##use ssim to verify the similarity 
                ssim_val = compare_ssim(gray_head_img_cv, gray_all_head_i_cv)
                if ssim_val >= 0.3:
                        return False
        
        return True 

##decide whether the head region including a helmet
def is_not_helmet(head_img_PIL):
        yellow_count = 0
        ##determine if a pixel is yellow
        for i in range(head_img_PIL.width):
                for j in range(head_img_PIL.height):
                        pixel_RGB = head_img_PIL.getpixel((i,j))
                        if pixel_RGB[0]>150 and pixel_RGB[1]>150 and pixel_RGB[2]<120:
                                yellow_count = yellow_count + 1
        print(yellow_count)
        if yellow_count > 15:
                return False
        else:
                return True

##draw three newest head on img by using the head queue
def draw_head(img, head_queue, disp_box):
        ##opencv to PIL
        img_PIL = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        for i in range(len(head_queue)):
                head_img_PIL = head_queue[i]
                img_PIL.paste(head_img_PIL,disp_box[i]) ##disp_box:(x,y,w,h)
        ##PIL to opencv
        img_with_head = cv.cvtColor(np.asarray(img_PIL),cv.COLOR_RGB2BGR)

        return img_with_head
