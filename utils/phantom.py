import os, sys
import os.path as osp
import numpy as np
import cv2


def phantom(psize=256, ellipses=None, numpts=4, type='modified'):
    if ellipses is None:
        ellipses = phantom_ellipses(type)

    img = np.zeros((psize, psize))
    wh = psize//2
    points = []
    boxes = []
    rot = ellipses[0,-1]
    boxw = 0
    boxcenter = (0,0)
    buffer = min(5, 0.5*psize)
    for i, ellip in enumerate(ellipses):
        I   = ellip[0]
        xi = ellip[3] * wh + wh
        yi = ellip[4] * wh + wh
        center = (int(xi+0.5), int(yi+0.5))
        ai = ellip[1]*wh
        bi = ellip[2]*wh
        axes = (int(ai+0.5), int(bi+0.5))
        phii = int(ellip[5]+0.5)
        try:
            img += cv2.ellipse(np.zeros((psize, psize)), center, axes, phii, 0, 360, I, -1)
        except Exception as e:
            import pdb; pdb.set_trace()

        if i in [2, 3]:
            M = cv2.getRotationMatrix2D((0,0),-phii,1)
            transxy = [xi, yi]

            point = [0, axes[1], 1]
            point = np.matmul(M, point) + transxy
            points.append(point)

            point = [0, -axes[1], 1]
            point = np.matmul(M, point) + transxy
            points.append(point)

            if numpts > 4:
                point = [axes[0], 0, 1]
                point = np.matmul(M, point) + transxy
                points.append(point)

                point = [-axes[0], 0, 1]
                point = np.matmul(M, point) + transxy
                points.append(point)

            if i == 2:
                boxw = max(ai, bi)
                boxcenter = np.array(transxy)
            else:
                boxh = max(ai, bi)
                boxh = max(boxw, boxh)*2 + buffer
                boxw += max(ai, bi)
                dist = abs(transxy[0] - boxcenter[0])
                boxw = boxw + dist + buffer
                boxcenter = (boxcenter + transxy)/2
                box = np.array([boxcenter[0], boxcenter[1], boxw, boxh])
                boxes.append(box)

        if False and type == 'same_shape' and i in [4, 5]:
            M = cv2.getRotationMatrix2D((0,0),-phii,1)
            transxy = [xi, yi]

            point = [0, axes[1], 1]
            point = np.matmul(M, point) + transxy
            points.append(point)

            point = [0, -axes[1], 1]
            point = np.matmul(M, point) + transxy
            points.append(point)

            if numpts > 4:
                point = [axes[0], 0, 1]
                point = np.matmul(M, point) + transxy
                points.append(point)

                point = [-axes[0], 0, 1]
                point = np.matmul(M, point) + transxy
                points.append(point)

            if i == 4:
                boxw = max(ai, bi)
                boxcenter = np.array(transxy)
            else:
                boxh = max(ai, bi)
                boxh = max(boxw, boxh)*2 + buffer
                boxw += max(ai, bi)
                dist = abs(transxy[0] - boxcenter[0])
                boxw = boxw + dist + buffer
                boxcenter = (boxcenter + transxy)/2
                box = np.array([boxcenter[0], boxcenter[1], boxw, boxh])
                boxes.append(box)

    npoints = []
    if False:
        M = cv2.getRotationMatrix2D((wh,wh),rot,1)
        img = cv2.warpAffine(img,M,(psize,psize))
        for point in points:
            point = point.tolist() + [1]
            point = np.matmul(M, point)
            npoints.append(point)
    else:
        npoints = points
    npoints = np.array(npoints).reshape(len(boxes), -1)
    boxes = np.array(boxes)
    outs = np.concatenate((boxes, npoints), 1)
    return img, outs


def phantom_ellipses(name='modified'):
    if name == 'normal':
        return _shepp_logan()
    elif name == 'same_shape':
        return same_shape()
    elif name == 'same_intensity':
        return same_intensity()
    elif name == 'random':
        return _random_shepp_logan()
    else: # if name == 'modified':
        return _mod_shepp_logan()


def _shepp_logan():
    # Standard head phantom, taken from Shepp & Logan
    # intensity, alpha, beta, centerx, centery, pheta, total rotation
    # 0,         1,     2,    3,       4,       5,     6
    return np.array([
        [   2,   .69,   .92,    0,      0,   0, 0],
        [-.98, .6624, .8740,    0, -.0184,   0, 0],
        [-.02, .1100, .3100,  .22,      0, -18, 0],
        [-.02, .1600, .4100, -.22,      0,  18, 0],
        [ .01, .2100, .2500,    0,    .35,   0, 0],
        [ .01, .0460, .0460,    0,     .1,   0, 0],
        [ .02, .0460, .0460,    0,    -.1,   0, 0],
        [ .01, .0460, .0230, -.08,  -.605,   0, 0],
        [ .01, .0230, .0230,    0,  -.606,   0, 0],
        [ .01, .0230, .0460,  .06,  -.605,   0, 0]
    ])

def _mod_shepp_logan():
    # Modified version of Shepp & Logan's head phantom,
    # adjusted to improve contrast.  Taken from Toft.
    # intensity, alpha, beta, centerx, centery, pheta, total rotation
    # 0,         1,     2,    3,       4,       5,     6
    return np.array([
        [   1,   .69,   .92,    0,      0,   0, 0],
        [-.80, .6624, .8740,    0, -.0184,   0, 0],
        [-.20, .1100, .3100,  .22,      0, -18, 0],
        [-.20, .1600, .4100, -.22,      0,  18, 0],
        [ .10, .2100, .2500,    0,    .35,   0, 0],
        [ .10, .0460, .0460,    0,     .1,   0, 0],
        [ .10, .0460, .0460,    0,    -.1,   0, 0],
        [ .10, .0460, .0230, -.08,  -.605,   0, 0],
        [ .10, .0230, .0230,    0,  -.606,   0, 0],
        [ .10, .0230, .0460,  .06,  -.605,   0, 0]
    ])


def _random_shepp_logan():
    phntm = _mod_shepp_logan()

    shape_cov = np.array([
     [1.0, 0.9, 0.1, 0.0], # alpha1
     [0.9, 1.0, 0.0, 0.1], # alpha2
     [0.1, 0.0, 1.0, 0.9], # beta1
     [0.0, 0.1, 0.9, 1.0]  # beta2
    ])

    rot_cov = np.array([
     [1.0, 0.5], # theta1
     [0.5, 1.0]  # theta2
    ])

    alphabeta = np.random.multivariate_normal(np.zeros((4,)), shape_cov)*0.05
    phntm[2, 1] += alphabeta[0]
    phntm[2, 2] += alphabeta[2]
    phntm[3, 1] += alphabeta[1]
    phntm[3, 2] += alphabeta[3]
    phntm[2, 1:3] = np.maximum(phntm[2, 1:3], 0.01)
    phntm[3, 1:3] = np.maximum(phntm[3, 1:3], 0.01)
    pheta = np.random.multivariate_normal(np.zeros((2,)), rot_cov)*5.0
    phntm[2][5] += pheta[0]
    phntm[3][5] += pheta[1]

    return phntm


def same_shape():
    phntm = _random_shepp_logan()

    phntm = np.concatenate((phntm[:4], phntm[2:4], phntm[4:]), 0)
    movy = np.random.choice([0.3, -0.3, 0.45, -0.45])
    phntm[2:4, 4] -= movy
    phntm[4:6, 4] += movy
    phntm[4:6, 0] = 0.1

    return phntm


def same_intensity():
    phntmog = _mod_shepp_logan()
    phntm = _random_shepp_logan()

    phntm = np.concatenate((phntm[:4], phntmog[2:4], phntm[4:]), 0)
    movy = np.random.choice([0.3, -0.3, 0.45, -0.45])
    phntm[2:4, 4] -= movy
    phntm[4:6, 4] += movy
    x = phntm[4, 3]
    phntm[4, 3] = phntm[5, 3]
    phntm[5, 3] = x
    phntm[4:6, 1] += np.random.normal(0.0, 0.03, 1)
    phntm[4:6, 2] += np.random.normal(0.0, 0.03, 1)
    phntm[4, 1:3] = np.maximum(phntm[4, 1:3], 0.01)
    phntm[5, 1:3] = np.maximum(phntm[5, 1:3], 0.01)
    phntm[4:6, -2] += np.random.randint(0, 45, 2)

    return phntm


def main():
    psize = 416 if len(sys.argv) < 2 else int(sys.argv[1])
    tenp = 10 if psize >= 100 else (0.1 * psize)
    os.makedirs('phantom/images', exist_ok=True)
    os.makedirs('phantom/labels', exist_ok=True)
    for i in range(400):
        img, boxes = phantom(psize, numpts=8, type='random')
        with open('phantom/labels/phantom_{}.txt'.format(i), 'w') as f:
            f.write('phantom/images/phantom_{}.png\n'.format(i))
            boxes /= psize
            for b_i, box in enumerate(boxes):
                f.write(str(b_i) + ',')
                for p_i, point in enumerate(box):
                    f.write(str(point))
                    if p_i < (len(box)-1):
                        f.write(',')
                f.write('\n')
        cv2.imwrite('phantom/images/phantom_{}.png'.format(i), img*255)

    os.makedirs('phantom_sameshape/images', exist_ok=True)
    os.makedirs('phantom_sameshape/labels', exist_ok=True)
    for i in range(400):
        img, boxes = phantom(psize, numpts=8, type='same_shape')
        with open('phantom_sameshape/labels/phantom_{}.txt'.format(i), 'w') as f:
            f.write('phantom_sameshape/images/phantom_{}.png\n'.format(i))
            boxes /= psize
            for b_i, box in enumerate(boxes):
                f.write(str(b_i) + ',')
                for p_i, point in enumerate(box):
                    f.write(str(point))
                    if p_i < (len(box)-1):
                        f.write(',')
                f.write('\n')
        cv2.imwrite('phantom_sameshape/images/phantom_{}.png'.format(i), img*255)


    os.makedirs('phantom_sameintensity/images', exist_ok=True)
    os.makedirs('phantom_sameintensity/labels', exist_ok=True)
    for i in range(400):
        img, boxes = phantom(psize, numpts=8, type='same_intensity')
        with open('phantom_sameintensity/labels/phantom_{}.txt'.format(i), 'w') as f:
            f.write('phantom_sameintensity/images/phantom_{}.png\n'.format(i))
            boxes /= psize
            for b_i, box in enumerate(boxes):
                f.write(str(b_i) + ',')
                for p_i, point in enumerate(box):
                    f.write(str(point))
                    if p_i < (len(box)-1):
                        f.write(',')
                f.write('\n')
        cv2.imwrite('phantom_sameintensity/images/phantom_{}.png'.format(i), img*255)


if __name__ == '__main__':
    main()
