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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
        default="configs/config_part2.json", help="path to config file")
    parser.add_argument("-w", "--weights_path", type=str,
        default="checkpoints/yolov3_ckpt_0.pth", help="path to weights file")
    opt = parser.parse_args()
    print(opt)

    config = json.load(open(opt.config))
    # print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(config['data_config'].format(config['type']))
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    dataset = CSVDataset( valid_path,
        {'augment':False, 'multiscale':False,
        'type':config['type']} )

    # for _ in range(100):
    #     inst = dataset[random.randrange(len(dataset))]
    #     if inst is None:
    #         print('is none')
    #         import pdb; pdb.set_trace()


    model = make_model(config).to(device)
    print(model)

    # model = Darknet(
    #     config['model_def'].format(config['type']),
    #     type=config['type'] ).to(device)
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))
    #
    # print('model type', model.type)
    #
    # print("Compute mAP...")
    #
    #
    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    #
    # print(f"mAP: {AP.mean()}")
    #
    # if model.type == 'twoobj' or model.type == 'landmark':
    #     dist5 = np.sum(landm<5.0)/len(landm)
    #     dist10 = np.sum(landm<10.0)/len(landm)
    #     print('landmark dists:')
    #     print('\tunder5:', dist5)
    #     print('\tunder10:', dist10)
    #     print('\tavg_dist:', np.mean(landm))
    #     print('\tmax_dist:', np.max(landm))
