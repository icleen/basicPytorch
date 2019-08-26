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
    minxy = boxes[:,3:5] - (boxes[:,5:7]/2)
    maxxy = boxes[:,3:5] + (boxes[:,5:7]/2)
    if (boxes[:,1:3] < minxy).any():
        boxes[:,1:3] = np.maximum(boxes[:,1:3], minxy)
    if (boxes[:,1:3] > maxxy).any():
        boxes[:,1:3] = np.minimum(boxes[:,1:3], maxxy)

    # import pdb; pdb.set_trace()
    minxy *= img.shape[0]
    maxxy *= img.shape[0]
    img1 = img[int(minxy[0,1]):int(maxxy[0,1]), int(minxy[0,0]):int(maxxy[0,0])]
    img2 = img[int(minxy[1,1]):int(maxxy[1,1]), int(minxy[1,0]):int(maxxy[1,0])]
    boxes[:,1:3] *= img.shape[0]
    boxes[0,1:3] *= 1/img1.shape[0]
    boxes[1,1:3] *= 1/img2.shape[0]

    boxes = boxes[:, :3]
    boxes = np.pad(boxes, ((0,0),(0,2)), 'constant', constant_values=0.1)

    # img = cv2.rectangle(img, (int(minxy[0,0]), int(minxy[0,1])), (int(maxxy[0,0]), int(maxxy[0,1])), (255,0,0), 1)
    # img = cv2.rectangle(img, (int(minxy[1,0]), int(minxy[1,1])), (int(maxxy[1,0]), int(maxxy[1,1])), (255,0,0), 1)
    # cv2.imwrite('test.png', img)
    # cv2.imwrite('img1.png', img1)
    # cv2.imwrite('img2.png', img2)

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

    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes
    return img, targets
