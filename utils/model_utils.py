from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.utils import to_cpu


def run_modules(x, modules, targets=None, layer_outs=False):
    img_dim = x.shape[2]
    loss = 0
    layer_outputs, metric_outputs = [], []
    normals = ['convolutional', 'upsample', 'maxpool',
        'linear', 'flatten', 'convtranspose2d']
    losslays = ['reconstruction', 'classifier', 'regressor', 'latent']
    for i, (module_def, module) in enumerate(modules):
        # print('x:',x.shape,module_def['type'])
        if module_def['type'] in normals:
            x = module(x)
        elif module_def['type'] == 'route':
            x = torch.cat([layer_outputs[int(layer_i)]
                for layer_i in module_def['layers'].split(",")], 1)
        elif module_def['type'] == 'shortcut':
            layer_i = int(module_def['from'])
            x = layer_outputs[-1] + layer_outputs[layer_i]
            # x = x + layer_outputs[layer_i]
        elif module_def['type'] in losslays:
            x, layer_loss = module[0](x, targets)
            loss += layer_loss
            if module_def['type'] != 'latent':
                metric_outputs.append(x)
        elif module_def['type'] == 'yolo':
            x, layer_loss = module[0](x, targets, img_dim)
            loss += layer_loss
            metric_outputs.append(x)
        layer_outputs.append(x)
    metric_outputs = to_cpu(torch.cat(metric_outputs, 1))
    if layer_outs:
        return (metric_outputs, layer_outputs) \
            if targets is None else (metric_outputs, loss, layer_outputs)
    return metric_outputs if targets is None else (metric_outputs, loss)


def load_pertrained(weights_path, modules, cutoff):
    header_info = None
    seen = None
    # Open the weights file
    with open(weights_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
        header_info = header  # Needed to write header when saving weights
        seen = header[3]  # number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    # Establish cutoff for loading backbone weights
    if cutoff is None and "darknet53.conv.74" in weights_path:
        cutoff = 75

    ptr = 0
    for i, (module_def, module) in enumerate(modules):
        if i == cutoff:
            break
        if module_def["type"] == "convolutional":
            conv_layer = module[0]
            if module_def["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return header_info, seen


def save_weights(self, path, modules):
    # Iterate through layers
    for i, (module_def, module) in enumerate(modules):
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
