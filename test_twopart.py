from __future__ import division

from models import *
from utils.utils import *
from utils.post_process import *
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


def evaluate(model_yolo, model_regress, config, verbose=False, save_imgs=0):
    model_yolo.eval()
    model_regress.eval()

    # Get dataloader
    dataset = TwoPartDataset( config, train=False, augment=False )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['vbatch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    conf_thres = config['conf_thres']
    iou_thres = config['iou_thres']
    nms_thres = config['nms_thres']
    img_size = config['img_size']['yolo']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    land_metrics = []  # List of np arrs (landmark dists)
    vloss = 0
    saved = 0
    loop = tqdm.tqdm(total=len(dataloader), position=0)
    for batch_i, (imgps, imgs, targets) in enumerate(dataloader):

        imgsy = Variable(imgs[0].to(device))
        targetsy = Variable(targets[0].to(device), requires_grad=False)
        imgsr = Variable(imgs[1].to(device))
        targetsr = Variable(targets[1].to(device), requires_grad=False)

        with torch.no_grad():
            outputsy = model_yolo(imgsy)
            outputsy = non_max_suppression(outputsy,
                conf_thres=conf_thres, nms_thres=nms_thres)
            outputsy = post_process_twoobj(outputsy)

            outputsr, loss = model_regress(imgsr, targetsr)
            vloss += loss.cpu().item()
            outputsr = outputsr.view(-1, 4)
            land_metrics += list(get_regress_statistics(outputsr*imgsr.size(-1), targetsr.cpu()*imgsr.size(-1)))

        if saved < save_imgs:
            # take the first img in each batch and save the predicted img
            img = cv2.imread(imgp[0], 1)
            pred = outputsr[0].cpu().numpy()
            label = targetsr[0].cpu().numpy()
            pred *= img.shape[0]
            label *= img.shape[0]
            img = cv2.circle(img, (pred[0], pred[1]), 5, (0,255,0), 1)
            img = cv2.circle(img, (pred[2], pred[3]), 5, (0,255,0), 1)
            img = cv2.circle(img, (label[0], label[1]), 5, (0,0,255), 1)
            img = cv2.circle(img, (label[2], label[3]), 5, (0,0,255), 1)
            # print(outimg.shape)
            cv2.imwrite('output/regimg_{}.png'.format(batch_i), img)
            saved += 1

        # Extract labels
        labels += targetsy[:, 1].tolist()
        # Rescale target
        targetsy[:, 2:6] = xywh2xyxy(targetsy[:, 2:6])
        targetsy[:, 2:6] *= img_size
        metrics = get_batch_statistics(outputsy, targetsy.cpu(), iou_threshold=iou_thres)
        sample_metrics += metrics

        loop.set_description( 'vloss:{:3f},avg_dist:{:3f}'.format(
            loss, np.mean(land_metrics)) )
        loop.update(1)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics)) ]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    land_metrics = np.array(land_metrics).reshape(-1)
    return precision, recall, AP, f1, ap_class, land_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
        default="configs/two_part/config.json", help="path to config file")
    parser.add_argument("-y", "--weights_pathy", type=str,
        default="checkpoints/two_part-yolo-0.pth", help="path to weights file")
    parser.add_argument("-r", "--weights_pathr", type=str,
        default="checkpoints/two_part-regress-0.pth", help="path to weights file")
    opt = parser.parse_args()
    print(opt)

    config = json.load(open(opt.config))
    # print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(
        config['data_config']['names'].format(config['type']) )

    # Initiate model
    model_def = config['model_def']
    img_size = config['img_size']
    config['model_def'] = model_def['yolo']
    config['img_size'] = img_size['yolo']
    model_yolo = Darknet( config ).to(device)
    model_yolo.load_state_dict(torch.load(opt.weights_pathy))

    config['model_def'] = model_def['regress']
    config['img_size'] = img_size['regress']
    model_regress = ConfigModel( config ).to(device)
    model_regress.load_state_dict(torch.load(opt.weights_pathr))
    config['img_size'] = img_size

    precision, recall, AP, f1, ap_class, landm = evaluate(
        model_yolo, model_regress,
        config=config,
        verbose=True,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    dist5 = np.sum(landm<5.0)/len(landm)
    dist10 = np.sum(landm<10.0)/len(landm)
    print('landmark dists:')
    print('\tunder5:', dist5)
    print('\tunder10:', dist10)
    print('\tavg_dist:', np.mean(landm))
    print('\tmax_dist:', np.max(landm))
