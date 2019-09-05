from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets
from utils.utils import to_cpu


def create_modules(module_defs, configs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    activations = {
        'leaky': nn.LeakyReLU(0.1), 'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()
    }
    output_filters = [int(configs['img_dim'])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        module_name = f"{module_def['type']}_{module_i}"

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                module_name,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module( f"batch_norm_{module_i}",
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5) )
            # if module_def["activation"] == "leaky":
            #     modules.add_module( f"leaky_{module_i}", nn.LeakyReLU(0.1) )
            if 'activation' in module_def:
                modules.add_module(
                    f"{module_def['activation']}_{module_i}",
                    activations[module_def['activation']]
                )

        elif module_def['type'] == 'convtranspose2d':
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                module_name,
                nn.ConvTranspose2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    output_padding=int(module_def['out_pad'])
                )
            )
            if 'activation' in module_def:
                modules.add_module(
                    f"{module_def['activation']}_{module_i}",
                    activations[module_def['activation']]
                )

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module( f"_debug_padding_{module_i}",
                    nn.ZeroPad2d((0, 1, 0, 1)) )
            maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(module_name, maxpool)

        elif module_def['type'] == 'linear':
            modules.add_module(
                module_name,
                nn.Linear(
                    int(module_def['input']),
                    int(module_def['output'])
                )
            )
            if 'activation' in module_def:
                modules.add_module(
                    f"{module_def['activation']}_{module_i}",
                    activations[module_def['activation']]
                )

        elif module_def['type'] == 'flatten':
            dims = tuple( np.fromstring(module_def['dims'], dtype=int, sep=',') )
            modules.add_module( module_name, FlattenLayer(dims) )

        elif module_def['type'] == 'latent':
            filters = int(module_def['repsize'])
            alpha = float(module_def['alpha'])
            gamma = float(module_def['gamma'])
            modules.add_module(
                module_name,
                Latent2dLayer(filters, output_filters[-1], alpha=alpha, gamma=gamma)
            )

        elif module_def['type'] == 'upsample':
            upsample = Upsample(
                scale_factor=int(module_def['stride']), mode="nearest" )
            modules.add_module(module_name, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(module_name, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module(module_name, EmptyLayer())

        elif module_def['type'] == 'reconstruction':
            modules.add_module(
                module_name,
                ReconstructionLayer(module_def['loss'])
            )

        elif module_def['type'] == 'classifier':
            modules.add_module(
                module_name,
                ClassifyLayer(module_def['loss'])
            )

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_size = config['img_size']
            # Define detection layer
            if 'ftype' in module_def:
                yolo_layer = YOLOLayer(anchors, num_classes, img_size, type=module_def['ftype'])
            else:
                yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list


class FlattenLayer(nn.Module):
    """docstring for Flatten."""

    def __init__(self, dims):
        super(FlattenLayer, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(self.dims)


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Latent2dLayer(nn.Module):
    """Represents the latent space"""

    def __init__(self, repsize, infilters, alpha=0.0, gamma=1.0):
        super(Latent2dLayer, self).__init__()
        self.repsize = repsize
        self.alpha = alpha
        self.gamma = gamma
        self.mu = nn.Conv2d(
            in_channels=infilters,
            out_channels=repsize,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.std = nn.Conv2d(
            in_channels=infilters,
            out_channels=repsize,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, targets=None):
        """returns the z variable (latent space object)"""
        mu, logvar = self.mu(x), self.std(x)
        if targets is not None:
            # calculate KL divergence
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
            dkl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (1 - self.alpha) * dkl
            loss *= 0
            # info = torch.Tensor([0.0])
            # loss += (self.alpha + self.gamma - 1) * info
            return z, loss
        return mu


class ReconstructionLayer(nn.Module):
    """Classifies"""

    def __init__(self, loss):
        super(ReconstructionLayer, self).__init__()
        if loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif loss == 'mse':
            self.loss = nn.MSELoss()

    def forward(self, x, targets=None):
        if targets is not None:
            return x, self.loss(x, targets)
        return x


class ClassifyLayer(nn.Module):
    """Classifies"""

    def __init__(self, loss):
        super(ClassifyLayer, self).__init__()
        if loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif loss == 'nll':
            self.loss = nn.NLLLoss()

    def forward(self, x, targets=None):
        if targets is not None:
            return x, self.loss(x, targets)
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416, type='normal'):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.type = type

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        features = 5 if self.type == 'normal' else 7
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + features,
                grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        if self.type != 'normal':
            xl = torch.tanh(prediction[..., -2])  # Center x
            yl = torch.tanh(prediction[..., -1])  # Center y
            prediction = prediction[..., :-2]
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[...,  4])  # Conf
        pred_cls  = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        if self.type != 'normal':
            pred_lands = FloatTensor(prediction[..., :2].shape)
            pred_lands[..., 0] = (xl * pred_boxes[..., 2] / 2) + pred_boxes[..., 0]
            pred_lands[..., 1] = (yl * pred_boxes[..., 3] / 2) + pred_boxes[..., 1]
            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride,
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                    pred_lands.view(num_samples, -1, 2) * self.stride,
                ),
                -1,
            )
        else:
            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride,
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                ),
                -1,
            )

        if targets is None:
            return output, 0
        else:
            total_loss = 0
            built = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                isLandmark=(self.type!='normal'),
            )
            iou_scores, class_mask, obj_mask, noobj_mask = built[:4]
            tx, ty, tw, th, tcls, tconf = built[4:10]

            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj
            loss_conf += self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss += loss_x + loss_y + loss_w + loss_h
            total_loss += loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            if self.type != 'normal':
                txl, tyl = built[10:]
                # Loss : Mask outputs to ignore non-existing objects
                # (except with conf. loss)
                loss_xl = self.mse_loss(xl[obj_mask], txl[obj_mask])
                loss_yl = self.mse_loss(yl[obj_mask], tyl[obj_mask])
                total_loss += loss_xl + loss_yl
                self.metrics["xl"] = to_cpu(loss_xl).item()
                self.metrics["yl"] = to_cpu(loss_yl).item()
            self.metrics["loss"] = to_cpu(total_loss).item()

            return output, total_loss
