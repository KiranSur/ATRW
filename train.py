import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from ATRW_Dataset import AtrwAllPoseDataset
from tqdm import tqdm

# Freeze ResNet50 Backbone weights
# 000568.jpg has half filled bboxes example

# Dataloader
# Save Model after

def train(num_epochs, save_every):

    ATRWDataset = AtrwAllPoseDataset(root_dir="/test_atrw_reid/")

    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        pass
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ATRW ReID with PPBM")
    parser.add_argument("--num_epochs", default=100)
    parser.add_argument("--save_every", default=20)
    args = parser.parse_args()

    train(num_epochs=args.num_epochs, save_every=args.save_every)