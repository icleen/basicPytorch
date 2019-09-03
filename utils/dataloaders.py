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
from utils.augmentations import *


def basic_loader(img, label, augment=False):
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    label = torch.ToTensor(int(label))
    return img, label


def mnist_loader(img, label, augment=False):
    if augment:
        img = img.numpy()[0]
        rnd = np.random.random()
        if rnd < 0.1:
            img = cv2.medianBlur(img, 5)
        elif rnd < 0.2:
            img = cv2.blur(img,(7,7))
        elif rnd < 0.5:
            img = cv2.GaussianBlur(img,(7,7),0)
        if np.random.random() < 0.5:
            img = rotate(img, np.random.normal(0.0, 7.0))
        img = np.stack((img,)*3, axis=-1)
        img = img.reshape((3, self.height, self.width))
        img = transforms.ToTensor()(img)
    return img, label


def hip_loader(img, label, augment=False):
    #  Image
    img = cv2.imread(img, 1)
    #  Label
    boxes = None
    if len(label)%5==0:
        boxes = np.array(label, dtype=np.float64).reshape((-1, 5))
    elif len(label)%3==0:
        boxes = np.array(label, dtype=np.float64).reshape((-1, 3))
        boxes = np.pad(boxes, ((0,0),(0,2)), 'constant', constant_values=0.1)
    elif len(label)%7==0:
        boxes = np.array(label, dtype=np.float64).reshape((-1, 7))
        boxes = np.concatenate((boxes[:,:1], boxes[:,3:], boxes[:,1:3]), axis=1)

    # Apply augmentations
    if augment:
        rnd = np.random.random()
        if rnd < 0.1:
            img = cv2.medianBlur(img, 5)
        elif rnd < 0.2:
            img = cv2.blur(img,(7,7))
        elif rnd < 0.5:
            img = cv2.GaussianBlur(img,(7,7),0)
        # if np.random.random() < 0.5:
        #     img, boxes = rotate(img, boxes, np.random.normal(0.0, 7.0))
        boxes[:,1:] += np.random.normal(0.0, 0.002, boxes[:,1:].shape)

    img = transforms.ToTensor()(img)
    boxes = torch.from_numpy(boxes)
    if augment and np.random.random() < 0.5:
        img, targets = horisontal_flip(img, boxes)

    targets = None
    if len(label)%7==0:
        targets = torch.zeros((len(boxes), 8))
    else:
        targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes
    return img, targets


def part2_loader(img, label, augment=False):
    #  Image
    img = cv2.imread(img, 1)
    #  Label
    boxes = np.array(label, dtype=np.float64).reshape((-1, 7))
    minxy = np.maximum(boxes[:,3:5] - (boxes[:,5:7]/2), 0)
    maxxy = np.minimum(boxes[:,3:5] + (boxes[:,5:7]/2), 1)
    minxy *= img.shape[0]
    maxxy *= img.shape[0]
    img1 = img[int(minxy[0,1]):int(maxxy[0,1]), int(minxy[0,0]):int(maxxy[0,0])]
    img2 = img[int(minxy[1,1]):int(maxxy[1,1]), int(minxy[1,0]):int(maxxy[1,0])]
    if img1.shape[0] < 1 or img1.shape[1] < 1 or img2.shape[0] < 1 or img2.shape[1] < 1:
        import pdb; pdb.set_trace()
    boxes[:,1:3] *= img.shape[0]
    if (boxes[:,1:3] < minxy).any():
        boxes[:,1:3] = np.maximum(boxes[:,1:3], minxy)
    if (boxes[:,1:3] > maxxy).any():
        boxes[:,1:3] = np.minimum(boxes[:,1:3], maxxy)
    boxes[:,1:3] = boxes[:,1:3] - minxy
    boxes[:,1:3] = boxes[:,1:3] / np.array([[img1.shape[1], img1.shape[0]],
                                            [img2.shape[1], img2.shape[0]]])
    boxes[:,1:3] = np.minimum(boxes[:,1:3], 1)

    boxes = boxes[:, :3]
    boxes = np.pad(boxes, ((0,0),(0,2)), 'constant', constant_values=0.1)

    # Apply augmentations
    if augment:
        rnd = np.random.random()
        if rnd < 0.1:
            img1 = cv2.medianBlur(img1, 5)
            img2 = cv2.medianBlur(img2, 5)
        elif rnd < 0.2:
            img1 = cv2.blur(img1,(7,7))
            img2 = cv2.blur(img2,(7,7))
        elif rnd < 0.5:
            img1 = cv2.GaussianBlur(img1,(7,7),0)
            img2 = cv2.GaussianBlur(img2,(7,7),0)
        # if np.random.random() < 0.5:
        #     img, boxes = rotate(img, boxes, np.random.normal(0.0, 7.0))
        boxes[:,1:] += np.random.normal(0.0, 0.002, boxes[:,1:].shape)

    img1 = transforms.ToTensor()(img1)
    # img2 = transforms.ToTensor()(img2)
    boxes = torch.from_numpy(boxes)
    # if augment and np.random.random() < 0.5:
    #     img, targets = horisontal_flip(img, boxes)

    # targets = torch.zeros((len(boxes), 6))
    targets = torch.zeros((1, 6))
    targets[0, 1:] = boxes[0:1]
    # import pdb; pdb.set_trace()
    return img1, targets
