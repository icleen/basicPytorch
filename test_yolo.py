from __future__ import division

from models import *
from utils.utils import *
from utils.post_process import *
from utils.datasets import *
from utils.parse_config import *
from utils.drawers import *

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
    if 'phantom' in config['type']:
        dataset = PhantomSet( config, train=False, augment=False )
    else:
        dataset = FolderDataset( config, train=False, augment=False )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['vbatch_size'], shuffle=False,
        num_workers=1, collate_fn=dataset.collate_fn
    )

    expected = 2 if 'expected_objects' not in config['data_config'] else config['data_config']['expected_objects']
    lands = 0 if 'landmarks' not in config['data_config'] else config['data_config']['landmarks']

    conf_thres = config['conf_thres']
    iou_thres = config['iou_thres']
    nms_thres = config['nms_thres']
    img_size = config['img_size']

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    land_metrics = None
    landm_set = ['landmark', 'part2', 'phantom']
    landsm_set = ['multilands', 'phantomobj', 'twoobj']
    if model.type in landm_set or model.type in landsm_set:
        land_metrics = []  # List of np arrs (landmark dists)
    loop = tqdm.tqdm(total=len(dataloader), position=0)
    for batch_i, (imgps, imgs, targets) in enumerate(dataloader):

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            if model.type in landsm_set:
                outputs = non_max_suppression_multilands(outputs,
                    conf_thres=conf_thres, nms_thres=nms_thres,
                    landmarks=config['data_config']['landmarks'])
                outputs = post_process_expected(outputs, expected=expected)
            elif model.type == 'twoobj':
                outputs = non_max_suppression_multilands(outputs,
                    conf_thres=conf_thres, nms_thres=nms_thres)
                outputs = post_process_expected(outputs, expected=expected)
            else:
                outputs = non_max_suppression(outputs,
                    conf_thres=conf_thres, nms_thres=nms_thres)
                outputs = post_process_expected(outputs, expected=expected)

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
            pdists = get_land_statistics(outputs, targets, expected=expected)
            land_metrics += pdists.tolist()
            if batch_i < 1:
                print(pdists)

            loop.set_description( 'avg_dist:{:.3f}'.format(np.mean(land_metrics)) )
        elif model.type in landsm_set:
            targets[:, -(lands*2):] *= img_size
            pdists = get_multiland_statistics(outputs, targets, lands=lands, expected=expected)
            land_metrics += pdists.tolist()
            if batch_i < 1:
                print(pdists)

            loop.set_description( 'avg_dist:{:.3f}'.format(np.mean(land_metrics)) )
        else:
            loop.set_description( 'detecting' )
        if batch_i < 1:
            draw_predictions(imgps, imgs, outputs, targets, lands=lands)
        loop.update(1)
    loop.close()

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
        default="configs/twoobj/config.json", help="path to config file")
    parser.add_argument("-w", "--weights_path", type=str,
        default="checkpoints/yolov3_twoobj_best.pth", help="path to weights file")
    opt = parser.parse_args()
    print(opt)

    config = json.load(open(opt.config))
    # print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(
        config['data_config']['names'].format(config['type']) )

    # Initiate model
    model = Darknet( config ).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print('model type', model.type)

    precision, recall, AP, f1, ap_class, landm = evaluate(
        model,
        config=config,
        verbose=True,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    if landm is not None:
        dist5 = np.sum(landm<5.0)/len(landm)
        dist10 = np.sum(landm<10.0)/len(landm)
        print('landmark dists:')
        print('\tunder5:', dist5)
        print('\tunder10:', dist10)
        print('\tavg_dist:', np.mean(landm))
        print('\tmax_dist:', np.max(landm))
