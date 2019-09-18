from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

import os
import sys
import json
import argparse


def twoobj_analyze(config):
    dataset = FolderDataset( config, train=True, augment=True )
    colors = [(0,255,0), (0,0,255)]
    for i, (imgp, img, target) in enumerate(dataset):
        print('iter:', i)
        img = img.permute(1, 2, 0).numpy()*255
        # builtx, builty = fake_target_build(target, img.shape[1])
        tarbox = xywh2xyxy(target[:, 2:6]).numpy()
        target = target.numpy()
        labels = target[:,1]

        # Rescale boxes to original image
        tarbox = tarbox * img.shape[1]
        target[:, 2:6] = target[:, 2:6] * img.shape[1]
        target[:, -2:] = target[:, -2:] * img.shape[1]
        for j, box in enumerate(tarbox):
            x1, y1, x2, y2 = box
            x, y, w, h = target[j, 2:6]
            xl, yl = target[j, -2:]
            color = colors[int(labels[j])]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            img = cv2.circle(img, (xl, yl), 5, color, 1)

        cv2.imwrite('shown_imgs/show{}.png'.format(i), img)
    return


def landmark_analyze(config):
    data_config = parse_data_config(
        config['data_config'].format(config['type']))
    train_path = data_config["train"]
    class_names = load_classes(data_config["names"])
    dataset = CSVDataset(train_path, augment=False,
        multiscale=config['multiscale_training'])
    colors = [(0,255,0), (0,0,255)]
    for i, (imgp, img, target) in enumerate(dataset):
        print('iter:', i)
        img = img.permute(1, 2, 0).numpy()*255
        target = target.numpy()
        labels = target[:,1]
        target = target[:,2:]

        # Rescale boxes to original image
        target[:,:] = target[:,:] * img.shape[1]
        for j, box in enumerate(target):
            xl, yl = box[:2]
            color = colors[int(labels[j])]
            img = cv2.circle(img, (xl, yl), 5, color, 1)

        cv2.imwrite('shown_imgs/show{}.png'.format(i), img)
        # if (i+1)%10==0:
        #     return
            # input('wait...')
    return


def hipobj_analyze(config):
    data_config = parse_data_config(
        config['data_config'].format(config['type']))
    train_path = data_config["train"]
    class_names = load_classes(data_config["names"])
    dataset = CSVDataset(train_path, augment=False,
        multiscale=config['multiscale_training'])
    colors = [(0,255,0), (0,0,255)]
    for i, (imgp, img, target) in enumerate(dataset):
        print('iter:', i)
        img = img.permute(1, 2, 0).numpy()*255
        target[:, 2:] = xywh2xyxy(target[:, 2:])
        target = target.numpy()
        labels = target[:,1]
        target = target[:,2:]

        # Rescale boxes to original image
        target[:,:] = target[:,:] * img.shape[1]
        for j, box in enumerate(target):
            x1, y1, x2, y2 = box

            color = colors[int(labels[j])]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        cv2.imwrite('shown_imgs/show{}.png'.format(i), img)
        # if (i+1)%10==0:
        #     return
            # input('wait...')
    return


def multilands_analyze(config, examples=10):
    os.makedirs('visualized', exist_ok=True)
    landmarks = 2 if 'landmarks' not in config['data_config'] else config['data_config']['landmarks']
    dataset = FolderDataset( config, train=True, augment=False )
    colors = [(0,255,0), (0,0,255)]
    for i, (imgp, img, target) in enumerate(dataset):
        print('iter:', i)
        # print(imgp, ',', target)
        img = img.permute(1, 2, 0).numpy()*255

        tarbox = xywh2xyxy(target[:, 2:6]).numpy() * img.shape[1]
        targ_lands = target[:, -landmarks:].numpy() * img.shape[1]
        labels = target[:,1]

        for j, box in enumerate(tarbox):
            x1, y1, x2, y2 = box
            color = colors[int(labels[j])]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            for li in range(landmarks//2):
                li *= 2
                xl, yl = targ_lands[j, li:li+2]
                img = cv2.circle(img, (xl, yl), 5, color, 1)

        cv2.imwrite('visualized/show{}.png'.format(i), img)
        if i > (examples-1):
            break
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
        default="config/twoobj/config.json", help="path to config file")
    parser.add_argument("-e", "--examples", type=int,
        default=10, help="number of visualized examples to save")
    opt = parser.parse_args()
    print(opt)

    config = json.load(open(opt.config))
    # print(config)

    # model = Darknet(
    #     config['model_def'].format(config['type']),
    #     type=config['type'] ).to(device)
    #
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    multilands_analyze(config, examples=opt.e)
    # if 'landmark' in opt.config:
    #     landmark_analyze(config)
    # elif 'hipobj' in opt.config:
    #     hipobj_analyze(config)
    # else:
    #     multilands_analyze(config)




if __name__ == "__main__":
    main()
