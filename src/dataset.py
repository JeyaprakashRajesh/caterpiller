import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        assert len(self.image_files) == len(self.depth_files), "Mismatch between images and depth maps!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        # Load and normalize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))  # Resize
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # Convert to PyTorch format

        # Load and normalize depth map
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (128, 128))  # Resize
        depth = depth.astype(np.float32) / 255.0
        depth = torch.tensor(depth).unsqueeze(0)  # Convert to single-channel tensor

        return img, depth
