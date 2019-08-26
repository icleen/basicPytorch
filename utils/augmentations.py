import torch
import torch.nn.functional as F
import numpy as np
import cv2


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0),
        size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 1] = 1 - targets[:, 1]
    targets[:, 0] = 1 + (targets[:, 0]*-1)
    if targets.shape[-1] > 5:
        targets[:, -2] = 1 - targets[:, -2]
    return images, targets


def rotate(image, targets, degree):
    # img2 = draw_centers(image.copy(), targets)
    # img2 = draw_boxes(img2, targets)
    # cv2.imwrite('befor_rotate.png', img2)
    rows, cols = image.shape[:2]
    rotM = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    image = cv2.warpAffine(image, rotM, (cols, rows))
    for tar in targets:

        x1 = tar[1]*cols
        y1 = tar[2]*rows
        w = tar[3]*cols
        h = tar[4]*rows
        x2, y2 = int(x1 + (w/2)), int(y1 + (h/2))
        x1, y1 = int(x1 - (w/2)), int(y1 - (h/2))
        vec1 = np.array([x1, y1, 1])
        vec1 = np.dot(rotM, vec1)
        vec2 = np.array([x2, y2, 1])
        vec2 = np.dot(rotM, vec2)

        wh = np.absolute(vec2 - vec1)
        xy = (vec1 + vec2)/2
        tar[1:3] = xy / np.array([cols, rows])
        tar[3:5] = wh / np.array([cols, rows])

        if len(tar) > 5:
            xl = tar[-2]*cols
            yl = tar[-1]*rows
            vecl = np.array([xl, yl, 1])
            vecl = np.dot(rotM, vecl)
            vecl = vecl / np.array([cols, rows])
            tar[-1] = vecl[0]
            tar[-2] = vecl[1]

    # img2 = draw_centers(image.copy(), targets)
    # img2 = draw_boxes(img2, targets)
    # cv2.imwrite('after_rotate.png', img2)

    return image, targets


def rotate2obj(image, targets, degree):
    rows, cols = image.shape[:2]
    rotM = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    image = cv2.warpAffine(image, rotM, (cols, rows))
    for tar in targets:

        xl = tar[1]*cols
        yl = tar[2]*rows
        x1 = tar[3]*cols
        y1 = tar[4]*rows
        w = tar[5]*cols
        h = tar[6]*rows
        x2, y2 = int(x1 + (w/2)), int(y1 + (h/2))
        x1, y1 = int(x1 - (w/2)), int(y1 - (h/2))
        vec1 = np.array([x1, y1, 1])
        vec1 = np.dot(rotM, vec1)
        vec2 = np.array([x2, y2, 1])
        vec2 = np.dot(rotM, vec2)
        vecl = np.array([xl, yl, 1])
        vecl = np.dot(rotM, vecl)

        wh = np.absolute(vec2 - vec1)
        xy = (vec1 + vec2)/2
        tar[1:3] = vecl / np.array([cols, rows])
        tar[3:5] = xy / np.array([cols, rows])
        tar[5:7] = wh / np.array([cols, rows])

    return image, targets


def draw_centers(image, targets):
    for tar in targets:
        x1 = int(tar[1]*image.shape[1])
        y1 = int(tar[2]*image.shape[0])
        image = cv2.circle(image, (x1, y1), 5, (0,0,255), 1)
    return image

def draw_boxes(image, targets):
    for tar in targets:
        x1 = tar[1]*image.shape[1]
        y1 = tar[2]*image.shape[0]
        w = tar[3]*image.shape[1]
        h = tar[4]*image.shape[0]
        x2, y2 = int(x1 + (w/2)), int(y1 + (h/2))
        x1, y1 = int(x1 - (w/2)), int(y1 - (h/2))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
    return image
