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
    parser.add_argument("-c", "--config", type=str,
        default="configs/twoobj/config.json", help="path to config file")
    parser.add_argument("-v", "--verbose", default=False,
        help="if print all info")
    parser.add_argument("--continu", type=str, default=None,
        help="if continuing training from checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    config = json.load(open(opt.config))
    # print(config)

    landm_set = ['twoobj', 'landmark', 'part2', 'phantom']

    config['log_path'] = config['log_path']
    os.makedirs(config['log_path'], exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('/'.join(config['checkpoint_path'].split('/')[:-1]), exist_ok=True)

    logger = Logger(config['log_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(
        config['data_config']['names'] )

    # Initiate model
    model = Darknet( config ).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.continu:
        config['pretrained_weights'] = opt.continu
    if 'pretrained_weights' in config:
        if config['pretrained_weights'].endswith(".pth"):
            model.load_state_dict(torch.load(config['pretrained_weights']))
        else:
            model.load_darknet_weights(config['pretrained_weights'])

    # Get dataloader
    if config['type'] == 'phantom':
        dataset = PhantomSet( config, train=True, augment=True )
    else:
        dataset = FolderDataset( config, train=True, augment=True )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['n_cpu'],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    metrics = [
        "grid_size",
        "loss",
        "xl",
        "yl",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    val_acc = []
    modi = len(dataloader) // 5
    with open('{}_log.txt', 'w') as f:
        f.write('')
    bsf = 0.0
    if model.type in landm_set:
        bsf = 100000
    for epoch in range(config['epochs']):
        model.train()
        start_time = time.time()
        loop = tqdm.tqdm(total=len(dataloader), position=0)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            outputs, loss = model(imgs, targets)
            loss.backward()

            if batches_done % config['gradient_accumulations']:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            loop.set_description(
                'ep:{},loss:{:.3f}'.format( epoch, loss )
            )
            loop.update(1)

        loop.close()

        if epoch % config['checkpoint_interval'] == 0:
            torch.save(model.state_dict(),
                config['checkpoint_path'].format(epoch)
            )
            # print('saved:', config['checkpoint_path'].format(epoch))

        if epoch % config['evaluation_interval'] == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, landm = evaluate(
                model,
                config=config,
            )
            APmean = AP.mean()
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", APmean),
                ("val_f1", f1.mean()),
            ]
            if landm is not None:
                evaluation_metrics.append( ("landm", landm.mean()) )
            logger.list_of_scalars_summary(evaluation_metrics, epoch)
            val_acc.append(evaluation_metrics)
            with open(config['val_metrics'], 'w') as f:
                f.write(str(val_acc))
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {APmean}")
            if landm is not None:
                dist5 = np.sum(landm<5.0)/len(landm)
                dist10 = np.sum(landm<10.0)/len(landm)
                avg_dist = np.mean(landm)
                print('landmark dists:')
                print('\tunder5:', dist5)
                print('\tunder10:', dist10)
                print('\tavg_dist:', avg_dist)
                print('\tmax_dist:', np.max(landm))

                if bsf > avg_dist:
                    torch.save(model.state_dict(),
                        config['checkpoint_path'].format('best')
                    )
                    bsf = avg_dist
            else:
                if bsf < APmean:
                    torch.save(model.state_dict(),
                        config['checkpoint_path'].format('best')
                    )
                    bsf = APmean




"""
Notes on how to run
"""
