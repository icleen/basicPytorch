from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.model_utils import *
from utils.parse_config import *
from utils.layers import *
from utils.utils import to_cpu


class BasicModel(nn.Module):
    """docstring for BasicModel."""

    def __init__(self, arg):
        super(BasicModel, self).__init__()
        self.arg = arg


class ConfigModel(nn.Module):
    """model defined by configuration file"""

    def __init__(self, config):
        super(ConfigModel, self).__init__()
        self.module_defs = parse_model_config(
            config['model_def'].format(config['task']) )
        self.module_list = create_modules(self.module_defs, config)
        self.type = config['type']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None, layer_outs=False):
        return run_modules(x, zip(self.module_defs, self.module_list),
            targets=targets, layer_outs=layer_outs)

    def load_weights(self, weights_path, cutoff=None):
        """Parses and loads the weights stored in 'weights_path'"""
        self.header_info, self.seen = load_pertrained(
            weights_path, zip(self.module_defs, self.module_list), cutoff )

    def save_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff
                              (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        save_weights(path, zip(self.module_defs[:cutoff], self.module_list[:cutoff]))


class VAEModel(nn.Module):
    """model defined by configuration file"""

    def __init__(self, config):
        super(VAEModel, self).__init__()
        self.module_defs = parse_model_config(
            config['model_def'].format(config['task']) )
        self.module_list = create_modules(self.module_defs, config)
        latent = [m for m, moddef in enumerate(self.module_defs) if moddef['type'] == 'latent']
        latent = latent[0]
        self.decoder = self.module_list[latent+1:]
        self.type = config['type']
        self.gensize = 7
        self.repsize = int(self.module_defs[latent]['repsize'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, k=1):
        x = torch.randn(k, self.repsize, self.gensize, self.gensize)
        x = Variable(x.to(self.device), requires_grad=False)
        for i, module in enumerate(self.decoder):
            # print('x:',x.shape,module)
            x = module(x)
        return x.detach()

    def forward(self, x, targets=None, layer_outs=False):
        img_dim = x.shape[2]
        loss = 0
        losses = []
        layer_outputs, metric_outputs = [], []
        zs = []
        normals = ['convolutional', 'upsample', 'maxpool',
            'linear', 'flatten', 'convtranspose2d']
        losslays = ['reconstruction', 'classifier']
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # print('x:',x.shape,module_def['type'])
            if module_def['type'] in normals:
                x = module(x)
            elif module_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)]
                    for layer_i in module_def['layers'].split(",")], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'latent':
                x, layer_loss = module[0](x, targets)
                losses.append( ('latent', x) )
                loss += layer_loss
                zs.append(x)
                # x = module(x)
                # metric_outputs.append(x)
            elif module_def['type'] in losslays:
                x, layer_loss = module[0](x, targets)
                losses.append( ('reconstruction', x) )
                loss += layer_loss
                metric_outputs.append(x)
            layer_outputs.append(x)
        metric_outputs = to_cpu(torch.cat(metric_outputs, 1))
        if layer_outs:
            return (metric_outputs, layer_outputs) \
                if targets is None else (metric_outputs, loss, layer_outputs)
        return metric_outputs if targets is None else (metric_outputs, loss, losses)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(
            config['model_def'].format(config['task']))
        self.module_list = create_modules(self.module_defs, config)
        self.yolo_layers = [layer[0]
            for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = config['img_size']
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.type = config['type']

    def forward(self, x, targets=None, layer_outs=False):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)]
                    for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        if layer_outs:
            return (yolo_outputs, layer_outputs) \
                if targets is None else (loss, yolo_outputs, layer_outputs)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        self.header_info, self.seen = load_pertrained(
            weights_path, zip(self.module_defs, self.module_list), cutoff)

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

if __name__ == '__main__':
    model = Darknet("config/yolov3.cfg")
    print(model)
