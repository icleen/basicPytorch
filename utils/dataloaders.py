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


class BasicLoader(object):
    """docstring for BasicLoader."""

    def __init__(self):
        super(BasicLoader, self).__init__()

    def load(self, img, label, augment=False):
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        label = torch.ToTensor(int(label))
        return img, label

    def augment(self, img, boxes):
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
        if np.random.random() < 0.5:
            img, boxes = horisontal_flip(img, boxes)
        return img, boxes

class MNISTLoader(BasicLoader):
    """docstring for MNISTLoader."""

    def __init__(self):
        super(MNISTLoader, self).__init__()

    def load(self, img, label, augment=False):
        if augment:
            img, label = self.augment(img, label)
        return img, label

    def augment(self, img, label):
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


class LandmarkLoader(BasicLoader):
    """docstring for LandmarkLoader."""

    def __init__(self, img_size):
        super(LandmarkLoader, self).__init__()
        self.img_size = (img_size, img_size)

    def load(self, img, label, augment=False):
        #  Image
        img = cv2.imread(img, 1)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        #  Label
        label = np.array(label, dtype=np.float64).reshape((-1, 3))[:,1:].reshape(-1)
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
            label += np.random.normal(0.0, 0.002, label.shape)

        img = transforms.ToTensor()(img)
        # img = img / 255.0
        if augment and np.random.random() < 0.5:
            img = torch.flip(img, [-1])
            label[1::2] = 1 - label[1::2]
        label = torch.FloatTensor(label)
        return img, label


class HipLoader(BasicLoader):
    """docstring for HipLoader."""

    def __init__(self):
        super(HipLoader, self).__init__()

    def load(self, img, label, augment=False):
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
            img, boxes = self.augment(img, boxes)
        else:
            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)

        targets = None
        if len(label)%7==0:
            targets = torch.zeros((len(boxes), 8))
        else:
            targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        return img, targets


class HipFileLoader(BasicLoader):
    """docstring for HipFileLoader."""

    def __init__(self):
        super(HipFileLoader, self).__init__()

    def load(self, filename, augment=False):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]
        img_path = lines[0]

        #  Image
        img = cv2.imread(img_path, 1)

        #  Label
        boxes = [[float(info) for info in line.split(',')] for line in lines[1:]]
        boxes = np.array(boxes, dtype=np.float64)

        # Apply augmentations
        if augment:
            img, boxes = self.augment(img, boxes)
        else:
            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)

        targets = None
        targets = torch.zeros((len(boxes), 8))
        targets[:, 1:] = boxes
        return img_path, img, targets


class Part2Loader(BasicLoader):
    """docstring for Part2Loader."""

    def __init__(self, img_size):
        super(Part2Loader, self).__init__()
        self.img_size = (img_size, img_size)

    def load(self, img, label, augment=False):
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
        boxes = boxes[:, 1:3]
        boxes *= img.shape[0]
        if (boxes < minxy).any():
            boxes = np.maximum(boxes, minxy)
        if (boxes > maxxy).any():
            boxes = np.minimum(boxes, maxxy)
        boxes = boxes - minxy
        boxes = boxes / np.array([[img1.shape[1], img1.shape[0]],
                                                [img2.shape[1], img2.shape[0]]])
        boxes = np.minimum(boxes, 1)
        img1 = cv2.resize(img1, self.img_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, self.img_size, interpolation=cv2.INTER_CUBIC)

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
            boxes += np.random.normal(0.0, 0.002, boxes.shape)

        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        if img1.shape != img2.shape:
            print('not same:', img1.shape, img2.shape)
        img = torch.stack([img1, img2])
        targets = torch.FloatTensor(boxes)
        return img, targets
