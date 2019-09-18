import os, sys
import os.path as osp
import csv, json, cv2
import numpy as np


def orig(lines, folp):
    for line in lines:
        imgp = line[0]
        line = line[1:]
        if len(line)%5==0:
            boxes = np.array(line, dtype=np.float64).reshape((-1, 5))
        elif len(line)%3==0:
            boxes = np.array(line, dtype=np.float64).reshape((-1, 3))
            boxes = np.pad(boxes, ((0,0),(0,2)), 'constant', constant_values=0.1)
        elif len(line)%7==0:
            boxes = np.array(line, dtype=np.float64).reshape((-1, 7))
            boxes = np.concatenate((boxes[:,:1], boxes[:,3:], boxes[:,1:3]), axis=1)

        wfile = osp.join(folp, imgp.split('/')[-1].split('.')[0]+'.txt')
        with open(wfile, 'w') as f:
            f.write(imgp)
            f.write('\n')
            for box in boxes:
                for el in range(box.shape[0]):
                    f.write(str(box[el]))
                    if el < box.shape[0]-1:
                        f.write(',')
                f.write('\n')


def alt(lines, folp):
    for line in lines:
        imgp = line[0]
        line = line[1:]

        boxes = np.array(line, dtype=np.float64).reshape((-1, 7))
        minxy = np.maximum(boxes[0,3:5] - (boxes[0,5:7]/2), 0)
        maxxy = np.minimum(boxes[1,3:5] + (boxes[1,5:7]/2), 1)
        nwh = np.maximum(maxxy-minxy, 0)
        nxy = np.maximum((maxxy+minxy)/2, 0)
        outlabel = nxy.tolist() + nwh.tolist()
        outlabel += boxes[0, 1:3].tolist() + boxes[1, 1:3].tolist()

        wfile = osp.join(folp, imgp.split('/')[-1].split('.')[0]+'.txt')
        with open(wfile, 'w') as f:
            f.write(imgp)
            f.write('\n')
            f.write('0,')
            for el in range(len(outlabel)):
                f.write(str(outlabel[el]))
                if el < len(outlabel)-1:
                    f.write(',')
            f.write('\n')


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
