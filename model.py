import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_pool
import numpy as np

class PPBM_Network(nn.Module):
    def __init__(self, num_individuals):
        super().__init__()

        # ResNet50 Backbone
        res50 = models.resnet50(weights='DEFAULT')
        split = 5

        # Split the ResNet50 backbone into two parts and remove the last linear layer
        self.backbone_p1 = nn.Sequential(*list(res50.children())[:split])
        self.backbone_p2 = nn.Sequential(*list(res50.children())[split:-1])

        # Global Layers
        self.global_fc1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.global_fc2 = nn.Linear(in_features=1024, out_features=num_individuals, bias=True)

        # Local Layers
        self.local_fc1 = nn.Linear(in_features=256, out_features=512)
        self.local_fc2 = nn.Linear(in_features=3584, out_features=1024, bias=True)
        self.local_fc3 = nn.Linear(in_features=1024, out_features=num_individuals, bias=True)

        # Misc
        self.ap2d = nn.AvgPool2d((16, 8))

    def forward(self, x, parts):
        x = self.backbone_p1(x) # output is (batch size, 256, 32, 64)

        # Get Global Vector
        global_vec = self.backbone_p2(x)
        global_vec = torch.flatten(global_vec, start_dim=1)
        global_vec = self.global_fc1(global_vec)
        global_vec = self.global_fc2(global_vec)
        global_vec = torch.softmax(global_vec, dim=-1)

        # Regional Average Pooling (RAP)

        # First, do Region of Interest (ROI) Pooling
        y = roi_pool(x, parts, (16, 8), 0.25)
        
        # Then, use average pool 2d on each feature
        y = self.ap2d(y)
        y = torch.flatten(y, start_dim=1)

        # Then, put each vector through a fc layer
        y = self.local_fc1(y)

        # Then, concatenate the vectors from each batch
        local_vec = []
        part_num = 7
        for i in range(0, y.size(dim=0), part_num):
            concatenated_vec = torch.squeeze(torch.reshape(y[i:i+part_num], (1, 3584)))
            local_vec.append(concatenated_vec)
        local_vec = torch.stack(local_vec)
        
        # Get Local Vector
        local_vec = self.local_fc2(local_vec)
        local_vec = self.local_fc3(local_vec)
        local_vec = torch.softmax(local_vec, dim=-1)

        return global_vec, local_vec


