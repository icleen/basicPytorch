import os, sys
import os.path as osp
import numpy as np
import cv2


def phantom(n=256,ellipses=None):
    if ellipses is None:
        ellipses = phantom_ellipses('random')

    img = np.zeros((n, n))
    wh = n//2
    prevI = 0
    ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]
    for ellip in ellipses:
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
        locs = (((x * cos_p + y * sin_p)**2) / a2
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
        # Add the ellipse intensity to those pixels
        img[locs] += I

    rot = ellipses[0,-1]
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    img = cv2.warpAffine(img,M,(cols,rows))
    return img


# def apply_phantom(img, ellipses):
#     n = img.shape[0]
#     wh = n//2
#     prevI = 0
#     ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]
#     for ellip in ellipses:
#         I   = ellip[0]
#         a2  = ellip[1]**2
#         b2  = ellip[2]**2
#         x0  = ellip[3]
#         y0  = ellip[4]
#         phi = ellip[5] * np.pi / 180  # Rotation angle in radians
#
#         # Create the offset x and y values for the grid
#         x = xgrid - x0
#         y = ygrid - y0
#
#         cos_p = np.cos(phi)
#         sin_p = np.sin(phi)
#
#         # Find the pixels within the ellipse
#         locs = (((x * cos_p + y * sin_p)**2) / a2
#               + ((y * cos_p - x * sin_p)**2) / b2) <= 1
#         # Add the ellipse intensity to those pixels
#         img[locs] += I
#         # print(I)
#
#         # a2  = ellip[1]**2 * wh
#         # b2  = ellip[2]**2 * wh
#         # ab = (int(a2), int(b2))
#         # x0  = ellip[3]*n + wh
#         # y0  = ellip[4]*n + wh
#         # xy = (int(x0+0.5), int(y0+0.5))
#         # phi = ellip[5]
#         # prevI += I
#         # temp = cv2.ellipse(cvimg.copy(), center=xy, axes=ab, angle=phi, startAngle=0, endAngle=360, color=I, thickness=-1)
#         # cvimg[temp!=cvimg] += I
#         # # img = cv2.ellipse(img, (125, 125), (100, 50), angle=90, startAngle=0, endAngle=360, color=255, thickness=-1)
#         # # print(prevI)
#
#     # cv2.imwrite('temp.png', cvimg*255)
#     rot = ellipses[0,-1]
#     rows,cols = img.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#     dst = cv2.warpAffine(img,M,(cols,rows))
#     return img


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
    psize = 416 if len(sys.argv) < 2 else sys.argv[1]

    os.makedirs('phantoms', exist_ok=True)
    for i in range(100):
        img = phantom(psize)
        cv2.imwrite('phantoms/phantom_{}.png'.format(i), img*255)


if __name__ == '__main__':
    main()
