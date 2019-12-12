from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from test_regressor import evaluate

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
        default="configs/landmark/config_regress.json", help="path to config file")
    parser.add_argument("-v", "--verbose", default=False,
        help="if print all info")
    parser.add_argument("--continu", type=str, default=None,
        help="if continuing training from checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    config = json.load(open(opt.config))
    # print(config)

    config['log_path'] = config['log_path'].format(config['type'])
    os.makedirs(config['log_path'], exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('/'.join(config['checkpoint_path'].split('/')[:-1]), exist_ok=True)

    logger = Logger(config['log_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = ConfigModel( config ).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.continu:
        config['pretrained_weights'] = opt.continu
    if 'pretrained_weights' in config:
        if config['pretrained_weights'].endswith(".pth"):
            model.load_state_dict(torch.load(config['pretrained_weights']))
        else:
            model.load_weights(config['pretrained_weights'])

    # Get dataloader
    # Get dataloader
    if 'phantom' in config['type']:
        dataset = PhantomSet( config, train=True, augment=True )
    elif 'dots' in config['type']:
        dataset = IndexDataset( config, set='train', augment=True )
    else:
        dataset = FolderDataset( config, train=True, augment=True )
    temp = dataset[0]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['n_cpu'],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    metrics = {
    'avg_loss': [],
    'vloss': [],
    'avg_dist': [],
    }
    bsf = 100000000
    modi = len(dataloader) // 5
    for epoch in range(config['epochs']):
        model.train()
        avg_loss = 0.0
        start_time = time.time()
        loop = tqdm.tqdm(total=len(dataloader), position=0)
        for batch_i, (imgp, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            outputs, loss = model(imgs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.cpu().item()

            loop.set_description(
                'epoch:{},loss:{:.4f}'.format( epoch, avg_loss/(batch_i+1) ) )
            loop.update(1)
        loop.close()

        if epoch % config['checkpoint_interval'] == 0:
            torch.save(
              model.state_dict(), config['checkpoint_path'].format(str(epoch))
            )

        if epoch % config['evaluation_interval'] == 0:
            vloss, avg_dist = evaluate( model, config, save_imgs=2 )
            metrics['avg_loss'].append(avg_loss/len(dataloader))
            metrics['vloss'].append(vloss)
            metrics['avg_dist'].append(avg_dist)
            with open(osp.join(config['log_path'], 'log.txt'), 'w') as f:
                f.write(str(metrics))

            if bsf > avg_dist:
                torch.save(
                  model.state_dict(), config['checkpoint_path'].format('best')
                )
                bsf = avg_dist

"""
Notes on how to run
"""
