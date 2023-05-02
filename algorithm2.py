import os
import numpy as np
import pandas as pd
from collections import Counter
import cv2
import imutils
import json
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
from torchvision import ops
from operator import itemgetter


def finalModel(image):
    
    boxes = []

    #convert the image to greyscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #increase the contrast
    cv2.convertScaleAbs(image, image)

    #apply gaussian blur to smooth the image
    image = cv2.GaussianBlur(image, (3, 3), 0)

    #apply adaptive threshholding
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)

    #apply canny edge detection
    image = cv2.Canny(image, 150, 200, 255)

    #find contours in image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #get the center of the image
    h, w = image.shape
    center = (int(w/2), int(h/2))

    #calculate the bounding rectangle of each contour and add it to a dictionary, only if it's near the center
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if abs((x + w/2) - center[0]) < 1.5*w and abs((y + h/2) - center[1]) < 1.5*h:
            boxes.append({'left': x, 'top': y, 'width': w, 'height': h})


    return boxes

def box_iou(boxA, boxB):
    # Calculate the intersection coordinates of the two bounding boxes
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the union area
    union_area = boxA_area + boxB_area - intersection_area
    
    # Compute the IOU
    iou = intersection_area / union_area
    
    return iou


def iouPicTest(truth, predicted, threshold1=0.5):
    ious = []
    for i in range(len(truth)):
        for j in range(len(predicted)):
            truth_box = [truth[i]['left'], truth[i]['top'], truth[i]['left'] + truth[i]['width'], truth[i]['top']+truth[i]['height']]
            predicted_box = [predicted[j]['left'], predicted[j]['top'], predicted[j]['left']+predicted[j]['width'], predicted[j]['top']+predicted[j]['height']]
            iou = box_iou(truth_box, predicted_box)
            if iou >= threshold1:
                ious.append(iou)
    acc = np.average(ious) if ious else 0
    return acc


images = []
labels = []
true_boxes = []

picsFolder_path = "train/"
with open('digitStruct.json') as f:
    data = json.load(f)

# import colored pictures
for i in range(len(data)):
    image = cv2.imread(picsFolder_path + data[i]['filename'])
    images.append(image)
    temp=[]
    for j in range(len(data[i]['boxes'])):
        temp.append(data[i]['boxes'][j]['label'])
    temp = np.array(temp)
    labels.append(temp)
    true_boxes.append(data[i]['boxes'])

# for image in os.listdir(picsFolder_path):
#     images.append(image)
print("we have",len(images),"images")


acc = []

for i in range(0,len(images)):
    boxes = finalModel(images[i])
    acc.append(iouPicTest(true_boxes[i],boxes))
print(np.average(acc)*100)

for i in range(0,10):
    image = images[i].copy()
    predicted_boxes = finalModel(image)
    for i in predicted_boxes:
        cv2.rectangle(image, (i['left'], i['top']), (i['left'] +
                        i['width'], i['top']+i['height']), (0, 255, 0), 2)
    cv2.imshow('image',image)
    cv2.waitKey(0)

image = images[16].copy()
predicted_boxes = finalModel(image)
print((iouPicTest(true_boxes[16],predicted_boxes))*100)
for i in predicted_boxes:
    cv2.rectangle(image, (i['left'], i['top']), (i['left'] +
                    i['width'], i['top']+i['height']), (0, 255, 0), 2)
# plt.imshow(image)