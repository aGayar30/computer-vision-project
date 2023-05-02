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
from backgroundremover_functions import backgroundremover1,backgroundremover2
from accuracy import iouPicTest,overlapTest,box_iou

# index = 0
# Get each bounding box
# Find the big contours on the filtered image:
# for edgeimage in edgeImages:
def draw_boxes(edgeimage, copyimage):
    contours, hierarchy = cv2.findContours(edgeimage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []
    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))
    for i in range(len(boundRect)):

        color = (0, 255, 0)
        # filter contours according to the area of the rectangle (parameter re5em)
        if ( int(boundRect[i][2])*int(boundRect[i][3])>100 and int(boundRect[i][2])*int(boundRect[i][3])<1000):
            cv2.rectangle(copyimage, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    # index= index + 1
    boxes= []
            # Iterate through each bounding box
    for i in range(len(boundRect)):
        # Extract the coordinates and dimensions of the bounding box
        x, y, w, h = boundRect[i]
            
        # Append the bounding box to the list in the format of a dictionary
        boxes.append({'left': x, 'top': y, 'width': w, 'height': h})
        
    # Return the list of bounding boxes
    return boxes  







images = []
labels = []
true_boxes = []
#Loading Training images and json file
picsFolder_path = "train/train/"
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
#resizing images

resizedImages =[]
for image in images:
    resizedImages.append(cv2.resize(image,(100,75)))
print(len(resizedImages))

#making a copy of resized images

imagesCopy = []
for resizedimage in resizedImages:
    imagesCopy.append(resizedimage)
print(len(imagesCopy))


backgroundremovedImages=[]

for resizedimage in resizedImages:
    background1=backgroundremover1(resizedimage)
    backgroundremovedImages.append(background1) # or 
    #background2=backgroundremover2(resizedimage)
    #backgroundremovedImages.append(background2)


greyImages = [] 
# Convert BGR to grayscale:
for backgroundremovedimages in backgroundremovedImages:
    greyImages.append(cv2.cvtColor(backgroundremovedimages, cv2.COLOR_BGR2GRAY)) 
print(len(greyImages))

sharpenedImages = []

for greyimage in greyImages:
    
    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(greyimage, (3, 3), 0)

    # apply Laplacian filter to extract edges
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)


    # apply edge sharpening using the grayscale original image and the Laplacian edges
    sharpened = cv2.convertScaleAbs(greyimage - laplacian)
    sharpenedImages.append(sharpened)

print(len(sharpenedImages))

# Set the adaptive thresholding:
windowSize = 31
windowConstant = -1

binaryImages = []


# Apply the threshold:
for sharpenedimage in sharpenedImages:
    binaryImages.append(cv2.adaptiveThreshold(sharpenedimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        windowSize, windowConstant))
    

filteredImages =[]

# Perform Connected Components:
for binaryimage in binaryImages:
    componentsNumber, labeledImage, componentStats, componentCentroids = cv2.connectedComponentsWithStats(binaryimage,
                                                                                                          connectivity=4)
    # Set the minimum pixels for the area filter to filter connected components:
    minArea = 20

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]
    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    # y3ni b3d de el mafood yfdal the largest connected components that hopefully contains the digits
    # one problem is that background connected components may still exist
    filteredImages.append(np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8'))



# Set kernel (structuring element) size:
kernelSize = 3

# Set operation iterations:
opIterations = 1

# Get the structuring element:
maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

closingImages = []

# Perform closing:
for filteredimage in filteredImages:
    closingImages.append(cv2.morphologyEx(filteredimage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations,
                                cv2.BORDER_REFLECT101))
    

edgeImages =[]

# perform smoothing
for closingimage in closingImages:
# perform canny edge detection:
    edgeImages.append(cv2.Canny(smoothedimage, 100, 200))
print(len(edgeImages))



acc = []
for i in range (len(images)):
    plt.imshow(images[i])
    plt.show()
    boxes = draw_boxes(edgeImages[i],imagesCopy[i])
    # tp,precision, recall, f1_score = overlapTest(edgeImages[i],imagesCopy[i])
    acc.append(iouPicTest(true_boxes[i],boxes))
    # print (acc)
    plt.imshow(imagesCopy[i])
    plt.show()
    # cv2.waitKey(0)
print(np.average(acc)*100)
# print(precision)
# print(recall)
# print(f1_score)
# print(tp/len(true_boxes))
