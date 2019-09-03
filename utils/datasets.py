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
from torchvision.datasets import MNIST

from utils.dataloaders import *


def expand3chans(a, dims=(3,-1,-1)):
    return a.expand(dims)


class MNISTClasses(Dataset):
    """docstring for MNISTClasses."""

    def __init__(self, root, train=True, augment=False, download=False):
        super(MNISTClasses, self).__init__()
        dotrans = [
            transforms.ToTensor(),
            transforms.Lambda(expand3chans)
        ]
        if augment:
            dotrans += [
                transforms.RandApply(transforms.ColorJitter(
                    0.5,0.5,0.5,0.25), 0.4),
                transforms.RandApply(transforms.RandomAffine(
                    45, translate=(0.1,0.1), scale=(0.9,1.1)), 0.4)
            ]
        dotrans = transforms.Compose(dotrans)
        self.mnist = MNIST(root,
            train=train, download=download, transform=dotrans)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return self.mnist[idx]


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

        self.kargs = kargs
        self.loader = basic_loader
        if 'type' in kargs:
            if kargs['type'] in ['landmark', 'classes', 'twoobj']:
                self.loader = hip_loader
            elif kargs['type'] == 'part2':
                self.loader = part2_loader

        self.img_size = kargs['img_size'] if 'img_size' in kargs else 416
        self.max_objects = 100
        self.augment = kargs['augment'] if 'augment' in kargs else False
        self.multiscale = kargs['multiscale'] if 'multiscale' in kargs else True
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index % len(self.lines)]
        img_path = line[0].rstrip()
        label = line[1:]
        img, targets = self.loader(img_path, label, self.augment)
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
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32) )
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
