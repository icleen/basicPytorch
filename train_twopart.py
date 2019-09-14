from __future__ import division

from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from test_twopart import evaluate

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
        default="configs/twoobj/config_twoobj.json", help="path to config file")
    parser.add_argument("-v", "--verbose", default=False,
        help="if print all info")
    parser.add_argument("--continu", type=str, default=None,
        help="if continuing training from checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    config = json.load(open(opt.config))
    # print(config)

    landm_set = ['twoobj', 'landmark', 'part2']

    config['log_path'] = config['log_path'].format(config['type'])
    os.makedirs(config['log_path'], exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs(config['checkpoint_path'], exist_ok=True)

    # logger = Logger(config['log_path'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(
        config['data_config']['names'].format(config['type']) )

    # Initiate model
    model_def = config['model_def']
    img_size = config['img_size']
    config['model_def'] = model_def['yolo']
    config['img_size'] = img_size['yolo']
    model_yolo = Darknet( config ).to(device)
    model_yolo.apply(weights_init_normal)
    # If specified we start from checkpoint
    if opt.continu:
        config['pretrained_weights'] = opt.continu
    if 'pretrained_weights' in config:
        # config['pretrained_weights'] = config['pretrained_weights']['yolo']
        if config['pretrained_weights'].endswith(".pth"):
            model_yolo.load_state_dict(torch.load(config['pretrained_weights']))
        else:
            model_yolo.load_darknet_weights(config['pretrained_weights'])

    config['model_def'] = model_def['regress']
    config['img_size'] = img_size['regress']
    model_regress = ConfigModel( config ).to(device)
    model_regress.apply(weights_init_normal)
    # If specified we start from checkpoint
    # if opt.continu:
    #     config['pretrained_weights'] = opt.continu
    # if 'pretrained_weights' in config:
    #     config['pretrained_weights'] = config['pretrained_weights']['regress']
    #     if config['pretrained_weights'].endswith(".pth"):
    #         model_regress.load_state_dict(torch.load(config['pretrained_weights']))
    #     else:
    #         model_regress.load_darknet_weights(config['pretrained_weights'])

    config['img_size'] = img_size

    # Get dataloader
    dataset = TwoPartDataset( config, train=True, augment=True )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['n_cpu'],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer_yolo = torch.optim.Adam(model_yolo.parameters(), lr=config['learning_rate'])
    optimizer_regress = torch.optim.Adam(model_regress.parameters(), lr=config['learning_rate'])

    metrics_yolo = [
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
    metrics_regress = {
    'avg_loss': [],
    'vloss': [],
    }
    val_acc = []
    modi = len(dataloader) // 5
    with open('{}_log.txt'.format(config['type']), 'w') as f:
        f.write('')
    for epoch in range(config['epochs']):
        model_yolo.train()
        model_regress.train()
        start_time = time.time()
        loop = tqdm.tqdm(total=len(dataloader), position=0)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgsy = Variable(imgs[0].to(device))
            targetsy = Variable(targets[0].to(device), requires_grad=False)
            outputs, loss = model_yolo(imgsy, targetsy)
            loss.backward()
            yolo_loss = loss.cpu().item()
            if batches_done % config['gradient_accumulations']:
                # Accumulates gradient before each step
                optimizer_yolo.step()
                optimizer_yolo.zero_grad()

            imgsr = Variable(imgs[1].to(device))
            targetsr = Variable(targets[1].to(device), requires_grad=False)
            outputs, loss = model_regress(imgsr, targetsr)
            loss.backward()
            regress_loss = loss.cpu().item()
            if batches_done % config['gradient_accumulations']:
                # Accumulates gradient before each step
                optimizer_regress.step()
                optimizer_regress.zero_grad()

            # # ----------------
            # #   Log progress
            # # ----------------
            #
            # log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch,
            #     config['epochs'], batch_i, len(dataloader))
            # log_str2 = "[Epoch %d/%d, Batch %d/%d] -" % (epoch,
            #     config['epochs'], batch_i, len(dataloader))
            #
            # metric_table = [["Metrics", *[f"YOLO Layer {i}"
            #     for i in range(len(model_yolo.yolo_layers))]]]
            #
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics_yolo):
            #     formats = {m: "%.6f" for m in metrics_yolo}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0)
            #         for yolo in model_yolo.yolo_layers]
            #     metric_table += [[metric, *row_metrics]]
            #
            #     # Tensorboard logging
            #     # tensorboard_log = []
            #     # for j, yolo in enumerate(model.yolo_layers):
            #     #     for name, metric in yolo.metrics.items():
            #     #         if name != "grid_size":
            #     #             tensorboard_log += [(f"{name}_{j+1}", metric)]
            #     # tensorboard_log += [("loss", loss.item())]
            #     # logger.list_of_scalars_summary(tensorboard_log, batches_done)
            #
            # log_str += AsciiTable(metric_table).table
            # log_str += f"\nTotal loss {loss.item()}"
            # log_str2 += f" Totloss {loss.item()}"
            #
            # # Determine approximate time left for epoch
            # epoch_batches_left = len(dataloader) - (batch_i + 1)
            # secs=epoch_batches_left*(time.time()-start_time)/(batch_i + 1)
            # time_left = datetime.timedelta(seconds=secs)
            # log_str += f"\n---- ETA {time_left}"
            # log_str2 += f" - ETA {time_left}"
            #
            # if batch_i % modi == 0:
            #     if opt.verbose:
            #         print(log_str)
            #     else:
            #         with open(join(config['log_path'], 'log.txt'), 'w') as f:
            #             f.write(log_str)
            #             f.write('\n')
            #         print(log_str2)
            # # print(log_str)
            #
            # model_yolo.seen += imgs.size(0)

            loop.set_description(
                'ep:{},yolo_l:{:.3f},reg_l:{:.3f}'.format(
                    epoch, yolo_loss, regress_loss ) )
            loop.update(1)

        loop.close()

        if epoch % config['checkpoint_interval'] == 0:
            torch.save(model_yolo.state_dict(),
                join(config['checkpoint_path'],
                '{}-{}-{}.pth'.format(config['task'], 'yolo', epoch))
            )
            torch.save(model_regress.state_dict(),
                join(config['checkpoint_path'],
                '{}-{}-{}.pth'.format(config['task'], 'regress', epoch))
            )

        if epoch % config['evaluation_interval'] == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, landm = evaluate(
                model_yolo, model_regress,
                config=config,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
                ("landm", landm.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)
            val_acc.append(evaluation_metrics)
            with open(config['val_metrics'], 'w') as f:
                f.write(str(val_acc))
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            if model.type in landm_set:
                dist5 = np.sum(landm<5.0)/len(landm)
                dist10 = np.sum(landm<10.0)/len(landm)
                print('landmark dists:')
                print('\tunder5:', dist5)
                print('\tunder10:', dist10)
                print('\tavg_dist:', np.mean(landm))
                print('\tmax_dist:', np.max(landm))


"""
Notes on how to run
"""
