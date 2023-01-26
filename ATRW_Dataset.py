import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from process_data import *

# NOT 107 INDIVIDUALS??
# NEED TO GO THROUGH THE CSV AND GET # OF UNIQUE INDIVIDUALS AND MAP THOSE TO AN ID

valid = get_valid_pose_bboxes("train_atrw_reid")
print(max(get_unique_inidivuals(valid)[0].values()))

class AtrwAllPoseDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = root_dir + "/train/"
        self.valid_poses = get_valid_pose_bboxes(self.root_dir)
        self.valid_images = self.valid_poses.keys()
        self.image_to_label, self.label_to_original_id = get_unique_inidivuals(self.valid_images)
        
    def __len__(self):
        return len(self.valid_images.keys())

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = self.train_dir + img_name
        image = read_image(img_path)
        
        