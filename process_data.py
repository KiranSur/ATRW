import os
import os.path as opt
import json
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import math
import pandas as pd
import torch

skeleton = [[0, 2], [1, 2], [2, 14], [14, 5], [5, 6], [14, 3], [3, 4], [14, 13],
            [13, 7], [13, 10], [7, 8], [8, 9], [10, 11], [11, 12]]

def get_bboxes(annotation, size):
    # size = (height x width)
    pos = np.array(annotation.reshape((15, 3)))

    # Check if this pair is valid in annotation
    # for pair in skeleton:
    #     if np.all(pos[pair, 2] > 0):             
    #         # Then valid

    #1.body:
    x1 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     
    x2 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]    
    y1 = [pos[1, 1], pos[0, 1], pos[13, 1]]                                       
    y2 = [pos[10, 1], pos[7, 1], pos[3, 1], pos[5, 1]]                           

    x1 = remove0(x1)
    x2 = remove0(x2)
    y1 = remove0(y1)
    y2 = remove0(y2)

    xmin = min(x1)
    xmax = max(x2)
    ymin = min(y1)
    ymax = max(y2)

    pose_bboxes = [[xmin, ymin, xmax, ymax]] # X=width, Y=height

    #2. paw
    pairs = [[5, 6], [3, 4], [7, 8], [8, 9], [10, 11], [11, 12]]
    for ind, pair in enumerate(pairs):
        ind += 1                                                
        i, j = pair[0], pair[1]
        if 0 not in [pos[i, 2], pos[j, 2]]:                     
            x1 = pos[i, 0]
            y1 = pos[i, 1]
            x2 = pos[j, 0]
            y2 = pos[j, 1]
            xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4 = aabb_box(x1, y1, x2, y2, ind)

            xmin = min(xx1, xx2, xx3, xx4)
            xmax = max(xx1, xx2, xx3, xx4)
            ymin = min(yy1, yy2, yy3, yy4)
            ymax = max(yy1, yy2, yy3, yy4)

            if xmin <= 0:
                xmin = 0
            if ymin <= 0:
                ymin = 0
            if xmax > size[1]:
                xmax = size[1]
            if ymax > size[0]:
                ymax = size[0]       

            pose_bboxes.append([xmin, ymin, xmax, ymax])

    return pose_bboxes                    

def aabb_box(x1, y1, x2, y2, ind):
    L = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

    tha = math.atan((x2 -x1)/((y2 -y1)+0.000000000001))
    if ind == 1 or ind == 2:
        detalx = math.cos(tha) * 0.3 * L
        detaly = math.sin(tha) * 0.3 * L
    if ind == 3 or ind == 5:
        detalx = math.cos(tha) * 0.45 * L
        detaly = math.sin(tha) * 0.45 * L
    else:
        detalx = math.cos(tha) * 0.3 * L
        detaly = math.sin(tha) * 0.3 * L

    xx1 = int(x1 - detalx)
    yy1 = int(y1 + detaly)
    xx2 = int(x1 + detalx)
    yy2 = int(y1 - detaly)
    xx4 = int(x2 - detalx)
    yy4 = int(y2 + detaly)
    xx3 = int(x2 + detalx)
    yy3 = int(y2 - detaly)

    return xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4

def remove0(x):
    while len(x) != 0:
        if 0 in x:
            x.remove(0)
        else:
            break
    return x

def get_valid_pose_bboxes(root_dir):
    valid = {}
    with open(os.path.join(root_dir,"reid_keypoints_train.json")) as f:
        pos_json = json.load(f)
        for image in pos_json.keys():
            pos = np.array(pos_json[image]).reshape((15, 3))
            if all([np.all(pos[pair, 2] > 0) for pair in skeleton]):
                x = Image.open(os.path.join(root_dir, "train", image))
                width, height = x.size
                pos = get_bboxes(pos, (height, width))
                valid[image] = torch.tensor(pos, dtype=torch.float)
    return valid

def get_unique_inidivuals(valid_pose_images):
    data = pd.read_csv("train_atrw_reid/reid_list_train.csv")

    data = data.reset_index()  # make sure indexes pair with number of rows

    unique = []
    all_images = {}

    # Get all id's and a dict with image : original_id

    for _, row in data.iterrows():
        id, img = row[1], row[2]
        if id not in unique and img in valid_pose_images:
            unique.append(id)
        all_images[img] = id

    # Sor the id's 
    unique = sorted(unique)

    # Get a dict with dict[original_id] = label
    unique_reverse_dict = {}
    for i in range(len(unique)):
        unique_reverse_dict[unique[i]] = i
    
    image_to_label = {}
    for img in all_images.keys():
        if img in valid_pose_images:
            image_to_label[img] = unique_reverse_dict[all_images[img]]

    # dict[image_name] = label
    # list[label] = original_id

    return image_to_label, unique

