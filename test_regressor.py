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


def evaluate(model, config, verbose=False, save_imgs=0):
    model.eval()

    # Get dataloader
    dataset = CSVDataset( config, train=False, augment=False )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dlen = len(dataset)/100
    stats = []
    saved = 0
    acc = 0.0
    vloss = 0.0
    loop = tqdm.tqdm(total=len(dataloader), position=0)
    for batch_i, (imgp, imgs, targets) in enumerate(dataloader):

        imgsv = Variable(imgs.to(device), requires_grad=False)
        targetsv = Variable(targets.to(device), requires_grad=False)

        with torch.no_grad():
            outputs, loss = model(imgsv, targetsv)
            loss = loss.cpu().item()
            vloss += loss
            outputs = regress_postp(outputs)
            stats += list(get_regress_statistics(outputs*imgs.size(-1), targets*imgs.size(-1)))


            if saved < save_imgs:
                # take the first img in each batch and save the predicted img
                img = cv2.imread(imgp[0], 1)
                pred = outputs[0].cpu().numpy()
                label = targets[0].cpu().numpy()
                pred *= img.shape[0]
                label *= img.shape[0]
                img = cv2.circle(img,
                    (pred[0], pred[1]), 5, (0,255,0), 1)
                img = cv2.circle(img,
                    (pred[2], pred[3]), 5, (0,255,0), 1)
                img = cv2.circle(img,
                    (label[0], label[1]), 5, (0,0,255), 1)
                img = cv2.circle(img,
                    (label[2], label[3]), 5, (0,0,255), 1)
                # print(outimg.shape)
                cv2.imwrite('output/regimg_{}.png'.format(batch_i), img)
                saved += 1

            loop.set_description( 'vloss:{:3f},avg_dist:{:3f}'.format(loss, np.mean(stats)) )
            loop.update(1)
    loop.close()
    stats = np.array(stats)
    dist5 = np.sum(stats<5.0)/len(stats)
    dist10 = np.sum(stats<10.0)/len(stats)
    print('regress dists:')
    print('\tunder5:', dist5)
    print('\tunder10:', dist10)
    # print('\tavg_dist:', np.mean(stats))
    print('\tmax_dist:', np.max(stats))
    return vloss/len(dataloader), np.mean(stats)


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

    results = evaluate( model, config, save_imgs=2 )
    print('vloss:', results)
