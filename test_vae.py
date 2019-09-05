from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import cv2
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


def evaluate(model, config, verbose=False, save_imgs=0):
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
    saved = 0
    acc = 0.0
    vloss = 0.0
    loop = tqdm.tqdm(total=len(dataloader), position=0)
    for batch_i, (imgs, targets) in enumerate(dataloader):

        imgs = Variable(imgs.to(device), requires_grad=False)

        with torch.no_grad():
            outputs, loss, losses = model(imgs, imgs)
            vloss += loss.cpu().item()

            if saved < save_imgs:
                # take the first img in each batch and save the predicted img
                outimg = outputs[0].cpu()
                outimg = (outimg.permute(1, 2, 0).numpy()*255).astype(np.uint8)
                # print(outimg.shape)
                cv2.imwrite('output/recon_{}.png'.format(batch_i), outimg)
                saved += 1

            loop.set_description( 'loss:{:.4f}'.format(
                vloss/(batch_i+1) ) )
            loop.update(1)
    loop.close()
    return vloss/len(dataloader)


def generate(model, config, k=1):
    model.eval()
    genimgs = model.generate(k=k).cpu()
    for genimg in genimgs:
        genimg = (genimg.permute(1, 2, 0).numpy()*255).astype(np.uint8)
        cv2.imwrite('output/genimg_test.png', genimg)


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
    model = VAEModel( config ).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    results = evaluate( model, config, save_imgs=10 )
    print('vloss:', results)
    generate(model, config)
