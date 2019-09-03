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


def evaluate(model, config, iou_thres, conf_thres, nms_thres, img_size, batch_size, type, verbose=False):
    model.eval()

    # Get dataloader
    dataset = get_dataset(config, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    land_metrics = None
    landm_set = ['twoobj', 'landmark', 'part2']
    if model.type in landm_set:
        land_metrics = []  # List of np arrs (landmark dists)
    for batch_i, (imgps, imgs, targets) in enumerate(
            tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            if model.type == 'twoobj':
                outputs = non_max_suppression_twoobj(outputs,
                    conf_thres=conf_thres, nms_thres=nms_thres)
            else:
                outputs = non_max_suppression(outputs,
                    conf_thres=conf_thres, nms_thres=nms_thres)
            outputs = post_process(outputs)

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
        targets[:, 2:6] *= img_size
        metrics = get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        sample_metrics += metrics
        if model.type in landm_set:
            if model.type == 'twoobj':
                targets[:, -2:] *= img_size
            pdists = get_land_statistics(outputs, targets)
            if False:
                indcs = np.arange(pdists.shape[0])[pdists[:,0] > 10.0]
                targets = targets.numpy().reshape(-1,2,8)
                outputs = np.array([out.numpy() for out in outputs])
                for d in indcs:
                    print('Over 10:')
                    out = outputs[d]
                    pland = out[:,5:7]
                    print('img:', imgps[d])
                    print('predicted confidence:', out[:,4])
                    print('predicted lands:',pland)
                    print('predicted classes',out[:,-1])
                    tar = targets[d]
                    tland = tar[:,-2:]
                    print('target lands:',tland)
                    print('target classes',tar[:,1])
                    print('distances',pdists[d])
            land_metrics += list(pdists)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics)) ]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    if land_metrics:
        land_metrics = np.array(land_metrics).reshape(-1)
    return precision, recall, AP, f1, ap_class, land_metrics



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

    data_config = config['data_config2']
    class_names = load_classes( data_config['names'].format(config['type']) )

    # Initiate model
    model = make_model(config).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print('model type', model.type)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, landm = evaluate(
        model,
        config=config,
        iou_thres=config['iou_thres'],
        conf_thres=config['conf_thres'],
        nms_thres=config['nms_thres'],
        img_size=config['img_size'],
        batch_size=config['vbatch_size'],
        type=config['type'],
        verbose=True,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    if model.type in ['twoobj', 'landmark', 'part2']:
        dist5 = np.sum(landm<5.0)/len(landm)
        dist10 = np.sum(landm<10.0)/len(landm)
        print('landmark dists:')
        print('\tunder5:', dist5)
        print('\tunder10:', dist10)
        print('\tavg_dist:', np.mean(landm))
        print('\tmax_dist:', np.max(landm))
