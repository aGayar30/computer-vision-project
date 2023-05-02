#imports:
import os
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
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def overlapTest(truth, predicted, threshold=0.7):
    tp = 0
    fp = 0
    for i in range(len(predicted)):
        matched = False
        for j in range(len(truth)):
            truth_box = [truth[j]['left'], truth[j]['top'], truth[j]['left'] + truth[j]['width'], truth[j]['top'] + truth[j]['height']]
            predicted_box = [predicted[i]['left'], predicted[i]['top'], predicted[i]['left'] + predicted[i]['width'], predicted[i]['top'] + predicted[i]['height']]
            iou = box_iou(truth_box, predicted_box)
            if iou >= threshold:
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1
    if len(truth) > 0:
        fn = len(truth) - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        return tp,precision, recall, f1_score
    else:
        return 0, 0, 0