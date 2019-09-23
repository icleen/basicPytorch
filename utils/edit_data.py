import os, sys
import os.path as osp
import csv, json, cv2
import numpy as np

from utils import *


def edit_instance(imgp, boxes):
    deg = 0.02
    landmarks = boxes[:,-2:]
    bxys = center2box(boxes[:,1:-2])
    for i in range(2):
        landm = landmarks[i]
        xy = bxys[i]
        if (landm >= xy[2:]).any():
            if landm[0] >= xy[2]: # check x values
                xy[2] = landm[0]+deg
                xy[0] += deg
            else: # landm[1] >= xy[3]: # check y values
                xy[3] = landm[1]+deg
                xy[1] += deg
            boxes[i,1:-2] = box2center(np.expand_dims(xy,axis=0))
        elif (landm < xy[:2]).any():
            if landm[0] < xy[0]: # check x values
                xy[0] = landm[0]-deg
                xy[2] -= deg
            else: # landm[1] < xy[1]: # check y values
                xy[1] = landm[1]-deg
                xy[3] -= deg
            boxes[i,1:-2] = box2center(np.expand_dims(xy,axis=0))

    return boxes


def main():
    if len(sys.argv) < 2:
        print('Usage: python', sys.argv[0], 'csv_file')
        exit()
    csv_path = sys.argv[1]
    with open(csv_path, 'r') as f:
        lines = [line for line in csv.reader(f, delimiter=',')]

    folp = csv_path.split('/')[-1].split('.')[0]
    if not osp.exists(folp):
        os.makedirs(folp)

    alt(lines, folp)


if __name__ == '__main__':
    main()
