import os, sys
import os.path as osp
import numpy as np
import cv2


def phantom(psize=256,ellipses=None):
    if ellipses is None:
        ellipses = phantom_ellipses('random')
        # ellipses = phantom_ellipses()

    img = np.zeros((psize, psize))
    cimg = np.zeros((psize, psize))
    wh = psize//2
    true_center = (wh, wh)
    for i, ellip in enumerate(ellipses):
        I   = ellip[0]
        xi = ellip[3] * wh + wh
        yi = ellip[4] * wh + wh
        center = (int(xi), int(yi))
        ai = ellip[1]*wh
        bi = ellip[2]*wh
        axes = (int(ai), int(bi))
        phii = ellip[5]
        img += cv2.ellipse(np.zeros((psize, psize)), center, axes, phii, 0, 360, I, -1)

        # phii = 30
        # print(phii)
        # M = cv2.getRotationMatrix2D((wh,wh),-phii,1)
        # img2 = cv2.warpAffine(img.copy(),M,(psize,psize))
        # cv2.imwrite('phantoms/test2.png', img2*255)

        cimg += cv2.ellipse(np.zeros((psize, psize)), center, axes, phii, 0, 360, I, -1)
        # point = (center[0]+axes[0], center[1])
        # temp = np.zeros((psize, psize))
        # temp[point] = 1
        # temp = cv2.warpAffine(temp, M, (psize,psize))
        # ind = np.unravel_index(temp.argmax(), temp.shape)
        # cimg = cv2.circle(cimg, ind, 5, 0.5, -1)
        #
        # point = [center[0]+axes[0], center[1], 1]
        # point = np.matmul(M, point)
        # cimg = cv2.circle(cimg, (int(point[0]), int(point[1])), 5, 0.5, -1)

        # M = cv2.getRotationMatrix2D((wh,wh),-1*(phii+90),1)

        if i < 4:
            # img = cv2.circle(img, (center[0]+axes[0], center[1]), 5, 0.5, -1)
            # img = cv2.circle(img, (center[0]-axes[0], center[1]), 5, 0.5, -1)
            img = cv2.circle(img, (center[0], center[1]+axes[1]), 5, 0.5, -1)
            img = cv2.circle(img, (center[0], center[1]-axes[1]), 5, 0.5, -1)

            M = cv2.getRotationMatrix2D((wh,wh),-phii,1)
            transxy = [center[0] - true_center[0], center[1] - true_center[1]]

            # point = [true_center[0]+axes[0], true_center[1], 1]
            # point = np.matmul(M, point) + transxy
            # point = (int(point[0]), int(point[1]))
            # cimg = cv2.circle(cimg, point, 5, 0.5, -1)
            #
            # point = [true_center[0]-axes[0], true_center[1], 1]
            # point = np.matmul(M, point) + transxy
            # point = (int(point[0]), int(point[1]))
            # cimg = cv2.circle(cimg, point, 5, 0.5, -1)

            point = [true_center[0], true_center[1]+axes[1], 1]
            point = np.matmul(M, point) + transxy
            point = (int(point[0]), int(point[1]))
            cimg = cv2.circle(cimg, point, 5, 0.5, -1)

            point = [true_center[0], true_center[1]-axes[1], 1]
            point = np.matmul(M, point) + transxy
            point = (int(point[0]), int(point[1]))
            cimg = cv2.circle(cimg, point, 5, 0.5, -1)
        #     break
        #     import pdb; pdb.set_trace()
        # input('wait...')
        # import pdb; pdb.set_trace()
    # rot = ellipses[0,-1]
    # M = cv2.getRotationMatrix2D((wh,wh),rot,1)
    # img = cv2.warpAffine(img,M,(psize,psize))
    cv2.imwrite('phantoms/test.png', img*255)
    cv2.imwrite('phantoms/test3.png', cimg*255)
    return img


def test():
    psize = 512
    wh = psize//2
    true_center = (wh, wh)
    center = (200, 256)
    axes = (100, 50)
    phii = 30

    img = np.zeros((psize, psize))
    img = cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
    img = cv2.circle(img, (center[0]+axes[0], center[1]), 5, 190, -1)
    img = cv2.circle(img, (center[0]-axes[0], center[1]), 5, 190, -1)
    img = cv2.circle(img, (center[0], center[1]+axes[1]), 5, 190, -1)
    img = cv2.circle(img, (center[0], center[1]-axes[1]), 5, 190, -1)
    M = cv2.getRotationMatrix2D((wh,wh),-phii,1)
    img = cv2.warpAffine(img,M,(psize,psize))
    cv2.imwrite('phantoms/test.png', img)

    img = np.zeros((psize, psize))
    img = cv2.ellipse(img, center, axes, phii, 0, 360, 255, -1)

    transxy = [center[0] - true_center[0], center[1] - true_center[1]]

    point = [true_center[0]+axes[0], true_center[1], 1]
    point = np.matmul(M, point) + transxy
    point = (int(point[0]), int(point[1]))
    img = cv2.circle(img, point, 5, 190, -1)

    point = [true_center[0]-axes[0], true_center[1], 1]
    point = np.matmul(M, point) + transxy
    point = (int(point[0]), int(point[1]))
    img = cv2.circle(img, point, 5, 190, -1)

    point = [true_center[0], true_center[1]+axes[1], 1]
    point = np.matmul(M, point) + transxy
    point = (int(point[0]), int(point[1]))
    img = cv2.circle(img, point, 5, 190, -1)

    point = [true_center[0], true_center[1]-axes[1], 1]
    point = np.matmul(M, point) + transxy
    point = (int(point[0]), int(point[1]))
    img = cv2.circle(img, point, 5, 190, -1)
    cv2.imwrite('phantoms/test2.png', img)

    # import pdb; pdb.set_trace()


def phantom_ellipses(name='modified'):
    if name == 'normal':
        return _shepp_logan()
    elif name == 'random':
        return _random_shepp_logan()
    else: # if name == 'modified':
        return _mod_shepp_logan()


def _shepp_logan():
    #  Standard head phantom, taken from Shepp & Logan
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
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
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
    # phntm += np.random.normal(0.0, 0.01, phntm.shape)
    phntm[2:,-2] += np.random.normal(0.0, 5, phntm[2:,-1].shape)
    phntm[:, -2] += np.random.normal(0.0, 5, 1)
    phntm[:, -1] += np.random.normal(0.0, 90, 1)
    phntm[:, 3] += np.random.normal(0.0, 0.1, 1)
    phntm[:, 4] += np.random.normal(0.0, 0.1, 1)
    # phntm[:, 1:3] += np.random.normal(0.0, 0.01, 2)
    return phntm


def main():
    psize = 416 if len(sys.argv) < 2 else int(sys.argv[1])

    os.makedirs('phantoms', exist_ok=True)
    for i in range(1):
        img = phantom(psize)
        cv2.imwrite('phantoms/phantom_{}.png'.format(i), img*255)

    # test()

if __name__ == '__main__':
    main()
