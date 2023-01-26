import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torchvision.ops import roi_pool
from torchvision.io import read_image
from model import PPBM_Network
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from PIL import Image
import os.path as opt
import math
import json
from process_data import *

# def bbox_tester(name):
    
#     with open("train_atrw_reid/reid_keypoints_train.json", 'r') as f:
#         pos = np.array(json.load(f)[name])
#         pos = np.array(pos.reshape((15, 3)))

#         im = Image.open("train_atrw_reid/train/" + name)
#         width, height = im.size
#         bboxes = get_bboxes(pos, (height, width))

#         # Create figure and axes
#         fig, ax = plt.subplots()

#         # Display the image
#         ax.imshow(im)

#         # Create a Rectangle patch
#         for box in bboxes:
#             rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='b', facecolor='none')
#             # Add the patch to the Axes
#             ax.add_patch(rect)

#         plt.savefig("body_annotated.png")

# bbox_tester("001260.jpg")

# with open("train_atrw_reid/reid_keypoints_train.json", 'r') as f:
#     pos = np.array(json.load(f)["001260.jpg"])
#     pos = np.array(pos.reshape((15, 3)))
#     print(get_bboxes(pos, (538, 891)))

# x = read_image("train_atrw_reid/train/001260.jpg")
# print(x.shape)

# pose_ann_path = './train_atrw_reid/reid_keypoints_train.json'
# img_path = './train_atrw_reid/train/'

# skeleton = [[0, 2], [1, 2], [2, 14], [14, 5], [5, 6], [14, 3], [3, 4], [14, 13],
#             [13, 7], [13, 10], [7, 8], [8, 9], [10, 11], [11, 12]]

# def show_ann_pic_bbox(name, pos):
#     img = Image.open(opt.join(img_path, name))
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     #画出关节点的连线
#     for pair in skeleton:
#         if np.all(pos[pair, 2] > 0):             #关节点可见
#             color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
#             plt.plot((pos[pair[0], 0], pos[pair[1], 0]), (pos[pair[0], 1], pos[pair[1], 1]), linewidth=3, color=color)

#     #标记出关节点
#     for i in range(15):
#         if pos[i, 2] > 0:
#             plt.plot(pos[i, 0], pos[i, 1], 'o', markersize=8, markeredgecolor='k', markerfacecolor='r')

#     #===========================画出bbox
#     #1.body:
#     x1 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     #x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
#     x2 = [pos[2, 0], pos[13, 0], pos[0, 0], pos[1, 0], pos[10, 0], pos[7, 0]]     #x，点3（鼻子），点14（尾巴），点1、2（耳朵），点8、11（屁股）
#     y1 = [pos[1, 1], pos[0, 1], pos[13, 1]]                                       #y1，点1、2（耳朵），点14（尾巴）
#     y2 = [pos[10, 1], pos[7, 1], pos[3, 1], pos[5, 1]]                            #y2，点8、11（屁股），点4、6（肩膀）

#     x1 = remove0(x1)
#     x2 = remove0(x2)
#     y1 = remove0(y1)
#     y2 = remove0(y2)

#     if x1 and x2 and y1 and y2:       #都不为空
#         #计算x
#         if len(x1) != 1:
#             xmin = min(x1)
#             xmax = max(x2)
#             ymin = min(y1)
#             ymax = max(y2)
#             rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='yellow', fill=False, linewidth=2)
#             ax.add_patch(rect)

#             #切图 & 保存图片
#             img_body = img.crop((xmin, ymin, xmax, ymax))
#             # img_body.save('./pr_data/Pseudo_Mask/'+ name.split('.')[0] +'_body.jpg')
#         else:
#             pass
#     else:
#         pass

#     #2. paw
#     pairs = [[5, 6], [3, 4], [7, 8], [8, 9], [10, 11], [11, 12]] #四个爪子的线段配对：左前，右前，右后上，右后下，左后上，左后下
#     for ind, pair in enumerate(pairs):
#         ind += 1                                                 #身体部位的编号
#         i, j = pair[0], pair[1]
#         if 0 not in [pos[i, 2], pos[j, 2]]:                      #判断是否有0在每一对线段中
#             x1 = pos[i, 0]
#             y1 = pos[i, 1]
#             x2 = pos[j, 0]
#             y2 = pos[j, 1]
#             xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4 = aabb_box(x1, y1, x2, y2, ind)
#             plt.plot((xx1, xx2), (yy1, yy2), linewidth=3, color='yellow')
#             plt.plot((xx2, xx3), (yy2, yy3), linewidth=3, color='yellow')
#             plt.plot((xx3, xx4), (yy3, yy4), linewidth=3, color='yellow')
#             plt.plot((xx4, xx1), (yy4, yy1), linewidth=3, color='yellow')

#             img1 = masked(opt.join(img_path, name), xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4)
#             if type(img1) == int:
#                 break
#             else:
#                 img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))              #转成PIL格式，避免opencv保存图片会产生错误
#                 # img1.save('./pr_data/Pseudo_Mask/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg')
#                 # cv.imwrite('data/Pseudo_Mask/'+ name.split('.')[0] +'_'+ str(ind) +'part.jpg' ,img1)

#     #显示图片
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#     #plt.savefig('name')
#     plt.close('all')                                #清理画布，防止一张图上多个图

# def aabb_box(x1, y1, x2, y2, ind):
#     L = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

#     tha = math.atan((x2 -x1)/((y2 -y1)+0.000000000001))
#     if ind == 1 or ind == 2:
#         detalx = math.cos(tha) * 0.3 * L
#         detaly = math.sin(tha) * 0.3 * L
#     if ind == 3 or ind == 5:
#         detalx = math.cos(tha) * 0.45 * L
#         detaly = math.sin(tha) * 0.45 * L
#     else:
#         detalx = math.cos(tha) * 0.3 * L
#         detaly = math.sin(tha) * 0.3 * L
#     #按照从左上角的位置开始顺时针1,2,3,4
#     xx1 = int(x1 - detalx)
#     yy1 = int(y1 + detaly)
#     xx2 = int(x1 + detalx)
#     yy2 = int(y1 - detaly)
#     xx4 = int(x2 - detalx)
#     yy4 = int(y2 + detaly)
#     xx3 = int(x2 + detalx)
#     yy3 = int(y2 - detaly)

#     return xx1, xx2, xx3, xx4, yy1, yy2, yy3, yy4

# #移除x列表中所有的0
# def remove0(x):
#     while len(x) != 0:
#         if 0 in x:
#             x.remove(0)
#         else:
#             break
#     return x

# def masked(img_path, x1, x2, x3, x4, y1, y2, y3, y4):
#     img = cv.imread(img_path)
#     (h, w, c) = img.shape
#     mask = np.zeros((h, w, 3), np.uint8)                                #定义mask的图片尺寸
#     triangle = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])       #定义mask部分位置
#     cv.fillConvexPoly(mask, triangle, (255, 255, 255))                  #取mask
#     img_mask = cv.bitwise_and(img, mask)                                #mask与原图融合
#     # plt.imshow(img_mask)
#     # plt.show()
#     #防止越界
#     xmin = min(x1, x2, x3, x4)
#     xmax = max(x1, x2, x3, x4)
#     ymin = min(y1, y2, y3, y4)
#     ymax = max(y1, y2, y3, y4)

#     if xmin <= 0:
#         xmin = 0
#     if ymin <= 0:
#         ymin = 0
#     if xmax > w:
#         xmax = w
#     if ymax > h:
#         ymax = h
#     if xmax <= 0 or ymax <=0 or xmin >= xmax or ymin >= ymax:
#         return 1
#     else:
#         # print('name:{}, xmin:{},xmax:{}, ymin:{}, ymax:{}'.format(img_path, xmin, xmax, ymin, ymax))
#         img_crop = img_mask[ymin: ymax, xmin: xmax]
#         return img_crop

# with open(pose_ann_path, 'r') as f:
#     pos = np.array(json.load(f)["001260.jpg"])
# pos = np.array(pos.reshape((15, 3)))

# show_ann_pic_bbox("001260.jpg", pos)

# # Model Testing

# # model = PPBM_Network()
# # for param in model.parameters():
# #     param.requires_grad = False

# # x = torch.from_numpy(np.array([np.zeros((3, 128, 256)), np.zeros((3, 128, 256))])).float()

# # parts = [torch.tensor([[0, 0, 2, 2], [3, 3, 4, 4], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=torch.float),
# #         torch.tensor([[0, 0, 2, 2], [3, 3, 4, 4], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=torch.float)]

# # p = torch.from_numpy(np.random.uniform(low=0, high=255, size=(1, 3, 128, 256))).float()

# # parts_2 = [torch.tensor([[0, 0, 2, 2], [3, 3, 4, 4], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=torch.float)]

# # y, z = model.forward(x, parts)
# # a, b = model.forward(p, parts_2)

# # print(y, a)

# # print(z[:1], "\n", b)
# # print(torch.allclose(y[:1], a, atol=0.0000001))

# # print("Global:", y[:1], y[1:], a)
# # print("\nLocal:", z, b)

# # Regional Average Pooling (RAP) Testing

# # x = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=float)
# # x = torch.softmax(x, dim=1)
# # print(x, x.shape)
# # x = np.arange(1, 11).reshape(2, 5)

# # x = [np.array([1, 2, 3]), np.array([4, 5, 6])]

# # print(np.array(x))

# # print(x)
# # print("\n\n")
# # print(x[:, :, 0:2, 0:2])

# # x = np.arange(0, 100).reshape(1, 4, 5, 5)
# # y = np.zeros((1, 4, 5, 5))
# # x = torch.tensor(x, dtype=torch.float)
# # y = torch.tensor(y, dtype=torch.float)
# # x = roi_pool(x, [torch.tensor([[0, 0, 2, 2], [3, 3, 4, 4]], dtype=torch.float)], (2, 2), 1)
# # y = roi_pool(y, [torch.tensor([[0, 0, 2, 2], [3, 3, 4, 4]], dtype=torch.float)], (2, 2), 1)
# # a = torch.stack([x, y])
# # print(a, a.shape)

# # x = torch.zeros((3, 5, 10, 10))
# # avgp = nn.AvgPool2d((10, 10))
# # a = avgp(x)
# # print(a, a.shape)
# # a = torch.flatten(a, start_dim=1)
# # print(a, a.shape)


# # BACKBONE TESTING

# # res50 = models.resnet50(weights='DEFAULT')
# # print(*list(res50.children())[7:10])

# # split = 5
# # x = torch.ones((1, 3, 256, 128))

# # backbone_p1 = nn.Sequential(*list(res50.children())[:split])
# # backbone_p2 = nn.Sequential(*list(res50.children())[split:-1])

# # global_fc1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
# # global_fc2 = nn.Linear(in_features=1024, out_features=107, bias=True)

# # fla = nn.Flatten()

# # x = backbone_p1(x)
# # print(x.shape)
# # x = backbone_p2(x)

# # x = fla(x)
# # x = global_fc1(x)
# # x = global_fc2(x)

# # print(x.shape)