from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from test_other import evaluate

from terminaltables import AsciiTable

import os
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
        default="configs/mnist/config.json", help="path to config file")
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
    os.makedirs(config['checkpoint_path'], exist_ok=True)

    logger = Logger(config['log_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(
        config['data_config2']['names'].format(config['type']) )

    # Initiate model
    model = make_model(config).to(device)
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
    dataset = get_dataset( config )
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
        "recon_loss",
        "kl_loss",
    ]
    val_acc = []
    modi = len(dataloader) // 5
    with open('{}_log.txt'.format(config['type']), 'w') as f:
        f.write('')
    for epoch in range(config['epochs']):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % config['gradient_accumulations']:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

        if epoch % config['checkpoint_interval'] == 0:
            torch.save(model.state_dict(),
                '{}/{}-{}-{}.pth'.format( config['checkpoint_path'],
                config['task'], config['type'], epoch) )

        if epoch % config['evaluation_interval'] == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            results = evaluate( model, config )
            present_results(results, config)


"""
Notes on how to run
"""
