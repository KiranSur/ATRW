import os
import os.path as opt
import json
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import math

# Freeze ResNet50 Backbone weights
# 000568.jpg has half filled bboxes example

skeleton = [[0, 2], [1, 2], [2, 14], [14, 5], [5, 6], [14, 3], [3, 4], [14, 13],
            [13, 7], [13, 10], [7, 8], [8, 9], [10, 11], [11, 12]]
img_path = "./train_atrw_reid/train/"

def get_pose_bboxes(name):
    return

def valid_pose_bboxes(pos):
    # with open("train_atrw_reid/reid_keypoints_train.json") as f:
    #     pos_json = json.load(f)
    #     pos = np.array(pos_json[name]).reshape((15, 3))
    pose_checks = [np.all(pos[pair, 2] > 0) for pair in skeleton]
    return sum(pose_checks) > 0

def remove0(x):
    while len(x) != 0:
        if 0 in x:
            x.remove(0)
        else:
            break
    return x

def get_ann_bbox_example(name, pos):
    img = Image.open(opt.join(img_path, name))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pair in skeleton:
        if np.all(pos[pair, 2] > 0):             
            color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            plt.plot((pos[pair[0], 0], pos[pair[1], 0]), (pos[pair[0], 1], pos[pair[1], 1]), linewidth=3, color=color)

    for i in range(15):
        if pos[i, 2] > 0:
            plt.plot(pos[i, 0], pos[i, 1], 'o', markersize=8, markeredgecolor='k', markerfacecolor='r')

    #1.body:
    x1 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     
    x2 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]    
    y1 = [pos[1, 1], pos[0, 1], pos[13, 1]]                                       
    y2 = [pos[10, 1], pos[7, 1], pos[3, 1], pos[5, 1]]                           

    x1 = remove0(x1)
    x2 = remove0(x2)
    y1 = remove0(y1)
    y2 = remove0(y2)

    if x1 and x2 and y1 and y2:      
    
        if len(x1) != 1:
            xmin = min(x1)
            xmax = max(x2)
            ymin = min(y1)
            ymax = max(y2)
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='yellow', fill=False, linewidth=2)
            ax.add_patch(rect)

            img_body = img.crop((xmin, ymin, xmax, ymax))
            img_body.save('./bbox_testing/'+ name.split('.')[0] +'_body.jpg')
        else:
            pass
    else:
        pass

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
            plt.plot((xx1, xx2), (yy1, yy2), linewidth=3, color='yellow')
            plt.plot((xx2, xx3), (yy2, yy3), linewidth=3, color='yellow')
            plt.plot((xx3, xx4), (yy3, yy4), linewidth=3, color='yellow')
            plt.plot((xx4, xx1), (yy4, yy1), linewidth=3, color='yellow')

            img1 = masked(opt.join(img_path, name), xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4)
            if type(img1) == int:
                break
            else:
                img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))              
                img1.save('./bbox_testing/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg')
                # cv.imwrite('data/Pseudo_Mask/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg' ,img1)


    plt.imshow(img)
    plt.axis('off')
    # plt.show()
    # plt.savefig(opt.join('data/split_by_id/pose/', name))
    plt.close('all')                               

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

def masked(img_path, x1, x2, x3, x4, y1, y2, y3, y4):
    img = cv.imread(img_path)
    (h, w, c) = img.shape
    mask = np.zeros((h, w, 3), np.uint8)                                
    triangle = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])       
    cv.fillConvexPoly(mask, triangle, (255, 255, 255))                  
    img_mask = cv.bitwise_and(img, mask)                              
    # plt.imshow(img_mask)
    # plt.show()
    xmin = min(x1, x2, x3, x4)
    xmax = max(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    ymax = max(y1, y2, y3, y4)

    if xmin <= 0:
        xmin = 0
    if ymin <= 0:
        ymin = 0
    if xmax > w:
        xmax = w
    if ymax > h:
        ymax = h
    if xmax <= 0 or ymax <=0 or xmin >= xmax or ymin >= ymax:
        return 1
    else:
        # print('name:{}, xmin:{},xmax:{}, ymin:{}, ymax:{}'.format(img_path, xmin, xmax, ymin, ymax))
        img_crop = img_mask[ymin: ymax, xmin: xmax]
        return img_crop

with open("train_atrw_reid/reid_keypoints_train.json") as f:
    pos_json = json.load(f)
    count = 0
    for img in pos_json.keys():
        pos = np.array(pos_json[img]).reshape((15, 3))
        count += valid_pose_bboxes(pos)
    print(count, len(pos_json.keys()))
    # pos = np.array(pos_json[pic_name]).reshape((15, 3))
    # get_ann_bbox_example(pic_name, pos)
