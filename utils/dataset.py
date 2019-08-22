import glob
import random
import os
import sys
import csv
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataloaders import *

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.files = [file for file in self.files if '.csv' not in file]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class CSVDataset(Dataset):
    def __init__(self, csv_path, kargs={}):
        self.csv_path = csv_path
        with open(csv_path, 'r') as f:
            self.lines = [line for line in csv.reader(f, delimiter=',')]
        # header = lines[0]
        # lines = lines[1:]

        self.img_files = []
        self.labels = []
        for line in lines:
            self.img_files.append(line[0])
            self.labels = line[1:]

        self.kargs = kargs

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.maxdeg = 45

    def __getitem__(self, index):
        line = self.lines[index % len(self.lines)]
        img_path = line[0].rstrip()
        label = line[1:]
        if 'type' in self.kargs:
            if 'landmark' == self.kargs['type'] or 'classes' == self.kargs['type'] or 'twoobj' == self.kargs['type']:
                img, targets = hip_loader(img_path, label)
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size,
                                                        self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
