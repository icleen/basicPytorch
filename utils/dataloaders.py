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
        boxes[:,1:] = np.maximum(boxes[:,1:], 0)
        boxes[:,1:] = np.minimum(boxes[:,1:], 0.9999)
        img = transforms.ToTensor()(img)
        boxes = torch.FloatTensor(boxes)
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


class PhantomLoad(BasicLoader):
    """docstring for PhantomLoad."""

    def __init__(self, config):
        super(PhantomLoad, self).__init__()
        self.img_size = config['img_size']
        self.widths = config['data_config']['widths']
        from utils.phantom import phantom
        self.imgsource = phantom

    def load(self, idx):
        img, points = self.imgsource(self.img_size)
        img = np.dstack((img, img, img)).astype(np.float32)*255
        img = transforms.ToTensor()(img)
        points = np.array(points, dtype=np.float32) / self.img_size
        points = np.pad(points, ((0,0),(2,0)), 'constant', constant_values=0)
        points = np.pad(points, ((0,0),(0,2)), 'constant', constant_values=self.widths)
        points[:,1] += [i for i in range(len(points))]
        points = torch.from_numpy(points)
        return 'phantom_{}.png'.format(idx), img, points

class PhantomObjLoad(BasicLoader):
    """docstring for PhantomObjLoad."""

    def __init__(self, config):
        super(PhantomObjLoad, self).__init__()
        self.img_size = config['img_size']
        self.widths = config['data_config']['widths']
        from utils.phantom import phantom
        self.imgsource = phantom

    def load(self, idx):
        img, points = self.imgsource(self.img_size)
        img = np.dstack((img, img, img)).astype(np.float32)*255
        # if idx == 0:
        #     print('writing')
        #     cv2.imwrite('mytest.png', img)
        img = transforms.ToTensor()(img)
        # if idx == 0:
        #     print('writing')
        #     img2 = img.permute(1, 2, 0).numpy()
        #     cv2.imwrite('mytest2.png', img2)
        points = np.array(points, dtype=np.float32)
        tenp = 10
        minxy = np.array([np.min(points[:,0])-tenp, np.min(points[:,1])-tenp])
        maxxy = np.array([np.max(points[:,0])+tenp, np.max(points[:,1])+tenp])
        wh = maxxy - minxy
        cenxy = (maxxy + minxy) / 2
        targets = np.zeros((1, 6+(points.shape[0]*2)), dtype=np.float32)
        targets[0,1] += [i for i in range(len(targets))]
        targets[0,2:4] = cenxy
        targets[0,4:6] = wh
        targets[0,6:] = points.flatten()
        targets /= self.img_size

        targets = torch.from_numpy(targets)
        return 'phantomobj_{}.png'.format(idx), img, targets

class BoxEdit(object):
    """docstring for BoxEdit."""
    def __init__(self, lands=2):
        super(BoxEdit, self).__init__()
        self.lands = lands
    def edit(self, boxes):
        return boxes

class LandBoxEdit(BoxEdit):
    """docstring for LandBoxEdit."""
    def __init__(self, lands=2, widths=0.1):
        super(LandBoxEdit, self).__init__(lands)
        self.widths = widths
    def edit(self, boxes):
        boxes = np.concatenate((boxes[:,:1], boxes[:,-2:]), axis=1)
        boxes = np.pad(boxes, ((0,0),(0,2)), 'constant', constant_values=self.widths)
        return boxes

class ObjBoxEdit(BoxEdit):
    """docstring for ObjBoxEdit."""
    def __init__(self, lands=2):
        super(ObjBoxEdit, self).__init__(lands)
    def edit(self, boxes):
        return boxes[:,:-self.lands]

class HipFileLoader(BasicLoader):
    """docstring for HipFileLoader."""

    def __init__(self, config):
        super(HipFileLoader, self).__init__()
        self.num_lands = config['data_config']['landmarks']*2
        self.boxedit = BoxEdit(self.num_lands)
        self.outsize = 6 + self.num_lands
        if config['type'] == 'landmark':
            self.boxedit = LandBoxEdit(
                self.num_lands, widths=config['data_config']['widths'] )
            self.outsize = 6
        elif config['type'] == 'hipobj':
            self.boxedit = ObjBoxEdit(self.num_lands)
            self.outsize = 6


    def load(self, filename, augment=False):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]
        img_path = lines[0]

        #  Image
        img = cv2.imread(img_path, 1)

        #  Label
        boxes = [[float(info) for info in line.split(',')] for line in lines[1:]]
        boxes = np.array(boxes, dtype=np.float64)
        boxes = self.boxedit.edit(boxes)

        # Apply augmentations
        if augment:
            img, boxes = self.augment(img, boxes)
        else:
            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)

        targets = torch.zeros((len(boxes), self.outsize))
        targets[:, 1:] = boxes
        return img_path, img, targets


class Part2Loader(BasicLoader):
    """docstring for Part2Loader."""

    def __init__(self, config):
        super(Part2Loader, self).__init__()
        self.num_lands = config['data_config']['landmarks']
        self.outsize = 6


    def load(self, filename, augment=False):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]
        img_path = lines[0]

        #  Image
        img = cv2.imread(img_path, 1)

        #  Label
        boxes = [[float(info) for info in line.split(',')] for line in lines[1:]]
        boxes = np.array(boxes)

        # Apply augmentations
        if augment:
            img, boxes = self.augment(img, boxes)
        else:
            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)

        landmarks = torch.zeros((len(boxes), self.num_lands+1))
        landmarks[:,1:] = boxes[:, -self.num_lands:]
        targets = torch.zeros((len(boxes), self.outsize))
        targets[:,1:] = boxes[:, :-self.num_lands]

        return img_path, img, (targets, landmarks)


class RegressFileLoader(BasicLoader):
    """docstring for RegressFileLoader."""

    def __init__(self, config):
        super(RegressFileLoader, self).__init__()
        self.img_size = (config['img_size'], config['img_size'])

    def load(self, filename, augment=False):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]
        img_path = lines[0]

        #  Image
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        #  Label
        boxes = [[float(info) for info in line.split(',')] for line in lines[1:]]
        boxes = np.array(boxes)[:,1:3].reshape(1, -1)

        # Apply augmentations
        if augment:
            img, boxes = self.augment(img, boxes)
        else:
            img = transforms.ToTensor()(img)
            boxes = torch.from_numpy(boxes)

        return img_path, img, boxes[0]
