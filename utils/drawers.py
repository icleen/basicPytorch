
import os
import os.path as osp
import numpy as np
import cv2
import torch
from utils.utils import center2box

def draw_predictions(imgps, imgs, preds, targets, config, lands=1):
    os.makedirs(config['val_examples'], exist_ok=True)
    tlands = lands*2
    imgs = imgs.cpu()
    targets = targets.cpu().numpy()
    red, green = (0,0,255), (0,255,0)
    for pred_i, pred in enumerate(preds):
        if pred is None:
            continue
        pred = pred.cpu().numpy()
        img = imgs[pred_i].permute(1, 2, 0).numpy()
        target = targets[targets[:, 0] == pred_i]
        imgp = imgps[pred_i].split('/')[-1]

        for obj_i, pobj in enumerate(pred):
            tobj = target[obj_i]
            img = cv2.rectangle(img, (int(pobj[0]), int(pobj[1])), (int(pobj[2]), int(pobj[3])), red, 1)
            img = cv2.rectangle(img, (int(tobj[2]), int(tobj[3])), (int(tobj[4]), int(tobj[5])), green, 1)

            pred_lands = pobj[5:5+tlands].reshape(-1, 2)
            targ_lands = tobj[-tlands:].reshape(-1, 2)
            for land_i in range(lands):
                img = cv2.circle(img, (int(pred_lands[land_i,0]), int(pred_lands[land_i,1])), 5, red, 1)
                img = cv2.circle(img, (int(targ_lands[land_i,0]), int(targ_lands[land_i,1])), 5, green, 1)
        # print(outimg.shape)
        cv2.imwrite(osp.join(config['val_examples'], imgp), img)
