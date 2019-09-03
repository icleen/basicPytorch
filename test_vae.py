from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import json
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, config, verbose=False):
    model.eval()

    # Get dataloader
    dataset = MNISTClasses( config['data_config'], train=False, augment=False )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dlen = len(dataset)/100
    labels = []
    acc = 0.0
    vloss = 0.0
    loop = tqdm.tqdm(total=len(dataloader), position=0)
    for batch_i, (imgs, targets) in enumerate(dataloader):

        imgs = Variable(imgs.to(device), requires_grad=False)
        targets = Variable(targets.to(device), requires_grad=False)

        with torch.no_grad():
            outputs, loss = model(imgs, targets)
            vloss += loss.cpu().item()
            preds = torch.argmax(outputs, -1)
            labels += preds.tolist()
            acc += (preds == targets.cpu()).sum().numpy()
            loop.set_description( 'acc:{}/{}={:.2f}%'.format(acc,dlen,acc/dlen) )
            loop.update(1)
    loop.close()
    return acc/dlen, vloss/len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
        default="configs/config_twoobj.json", help="path to config file")
    parser.add_argument("-w", "--weights_path", type=str,
        default="checkpoints/yolov3_ckpt_0.pth", help="path to weights file")
    opt = parser.parse_args()
    print(opt)

    config = json.load(open(opt.config))
    # print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = ConfigModel( config ).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))


    print("Compute mAP...")

    results = evaluate( model, config )
    print('vloss:', results[1])
