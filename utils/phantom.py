import os, sys
import os.path as osp
import numpy as np
import cv2


def phantom(psize=256,ellipses=None):
    if ellipses is None:
        ellipses = phantom_ellipses('random')

    img = np.zeros((psize, psize))
    cimg = np.zeros((psize, psize))
    wh = psize//2
    prevI = 0
    ygrid, xgrid = np.mgrid[-1:1:(1j*psize), -1:1:(1j*psize)]
    maxes = []
    colors = [255, 50, 0, 0, 200, 200, 200, 200, 200, 200]
    for i, ellip in enumerate(ellipses):
        I   = ellip[0]
        a2  = ellip[1]**2
        b2  = ellip[2]**2
        x0  = ellip[3]
        y0  = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians
        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        # Find the pixels within the ellipse
        dists = (((x * cos_p + y * sin_p)**2) / a2
              + ((y * cos_p - x * sin_p)**2) / b2)
        locs = dists <= 1
        # dists[dists > 1] = 0
        # print(dists.max(), dists.argmax())
        # maxes.append(dists == dists.max())
        # Add the ellipse intensity to those pixels
        img[locs] += I

        # cv2.imwrite('phantoms/test.png', img*255)
        print('value:', I)
        xi = x0 * wh + wh
        yi = y0 * wh + wh
        center = (int(xi), int(yi))
        print('center:', center)
        ai = ellip[1]*wh
        bi = ellip[2]*wh
        axes = (int(ai), int(bi))
        print('axes:', axes)
        phii = ellip[5]
        print('angle:', phii)
        # cimg = cv2.circle(cimg, (center[0], center[1]+axes[1]), 5, 255, -1)
        # cimg = cv2.circle(cimg, (center[0]+axes[0], center[1]), 5, 255, -1)
        cimg += cv2.ellipse(np.zeros((psize, psize)), center, axes, phii, 0, 360, I, -1)
        # import pdb; pdb.set_trace()
        # cv2.imwrite('phantoms/test2.png', cimg)

    # rot = ellipses[0,-1]
    # rows,cols = img.shape
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    # img = cv2.warpAffine(img,M,(cols,rows))

    # cimg = img.copy()
    # for mx in maxes:
    #     cimg[mx] = 2
    # cimg = np.ones((n, n, 3))
    # cimg *= np.expand_dims(img, -1)
    # # cimg = np.stack([img, img, img]).reshape(n, n, 3)
    # cimg *= 255
    # for mx in maxes:
    #     cimg = cv2.circle(cimg, np.unravel_index(mx, img.shape), 5, (255, 0, 0), -1)
    # import pdb; pdb.set_trace()
    cv2.imwrite('phantoms/test.png', cimg*255)
    return img


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


if __name__ == '__main__':
    main()
