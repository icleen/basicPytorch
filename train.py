from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from test_yolo import evaluate

from terminaltables import AsciiTable

import os
import os.path as osp
import sys
import time
import json
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "-c", "--config", type=str, default="configs/twoobj/config.json",
      help="path to config file"
    )
    parser.add_argument(
      "-v", "--verbose", default=False, help="if print all info"
    )
    parser.add_argument(
      "--continu", type=str, default=None,
      help="if continuing training from checkpoint model"
    )
    opt = parser.parse_args()
    # print(opt)

    config = json.load(open(opt.config))

    os.makedirs(config['log_path'], exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('/'.join(config['checkpoint_path'].split('/')[:-1]), exist_ok=True)

    logger = Logger(config['log_path'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config['task'] == 'yolov3':
        from train_yolo import train

    train(opt, config, logger, device)
