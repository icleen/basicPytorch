import os, sys
import os.path as osp
import numpy as np
import cv2


def phantom(psize=256,ellipses=None, numpts=4):
    if ellipses is None:
        ellipses = phantom_ellipses('random')
        # ellipses = phantom_ellipses()

    img = np.zeros((psize, psize))
    wh = psize//2
    points = []
    rot = ellipses[0,-1]
    for i, ellip in enumerate(ellipses):
        I   = ellip[0]
        xi = ellip[3] * wh + wh
        yi = ellip[4] * wh + wh
        center = (int(xi), int(yi))
        ai = ellip[1]*wh
        bi = ellip[2]*wh
        axes = (int(ai), int(bi))
        phii = int(ellip[5])
        try:
            img += cv2.ellipse(np.zeros((psize, psize)), center, axes, phii, 0, 360, I, -1)
        except Exception as e:
            import pdb; pdb.set_trace()

        if i in [2, 3]:
            M = cv2.getRotationMatrix2D((0,0),-phii,1)
            transxy = [center[0], center[1]]

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

    M = cv2.getRotationMatrix2D((wh,wh),rot,1)
    img = cv2.warpAffine(img,M,(psize,psize))
    npoints = []
    for point in points:
        point = point.tolist() + [1]
        point = np.matmul(M, point)
        npoints.append(point)
    return img, npoints


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
    # Standard head phantom, taken from Shepp & Logan
    # intensity, alpha, beta, centerx, centery, pheta
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
    phntm[2, 1:3] += np.maximum(alphabeta[:2], 0)
    phntm[3, 1:3] += np.maximum(alphabeta[2:], 0)
    pheta = np.random.multivariate_normal(np.zeros((2,)), rot_cov)*10.0
    phntm[2][5] += pheta[0]
    phntm[3][5] += pheta[1]

    # phntm[2:4, 1] += np.random.normal(0.0, 0.03, 1)
    # phntm[2:4, 2] += np.random.normal(0.0, 0.03, 1)
    # phntm[2:4, -2] += np.random.randint(0, 45, 2)
    return phntm


def main():
    psize = 416 if len(sys.argv) < 2 else int(sys.argv[1])
    tenp = 10 if psize >= 100 else (0.1 * psize)
    os.makedirs('phantom/images', exist_ok=True)
    os.makedirs('phantom/labels', exist_ok=True)
    for i in range(400):
        img, points = phantom(psize, numpts=4)
        with open('phantom/labels/phantom_{}.txt'.format(i), 'w') as f:
            for p_i, point in enumerate(points):
                # img = cv2.circle(img, (int(point[0]), int(point[1])), 5, 0.5, -1)
                point /= psize
                f.write(str(point[0]) + ',' + str(point[1]))
                if p_i < (len(points)-1):
                    f.write(',')
            f.write('\n')
        cv2.imwrite('phantom/images/phantom_{}.png'.format(i), img*255)



        # arpts = np.array(points)
        # minxy = (np.min(arpts[:,0])-tenp, np.min(arpts[:,1])-tenp)
        # maxxy = (np.max(arpts[:,0])+tenp, np.max(arpts[:,1])+tenp)
        # minxy = (int(minxy[0]+0.5), int(minxy[1]+0.5))
        # maxxy = (int(maxxy[0]+0.5), int(maxxy[1]+0.5))
        # img = cv2.rectangle(img, minxy, maxxy, 0.5, 1)
        # cv2.imwrite('phantoms/phantom_t{}.png'.format(i), img*255)

        # points = np.array(points, dtype=np.float32)
        # tenp = 10
        # minxy = np.array([np.min(points[:,0])-tenp, np.min(points[:,1])-tenp])
        # maxxy = np.array([np.max(points[:,0])+tenp, np.max(points[:,1])+tenp])
        # wh = maxxy - minxy
        # cenxy = (maxxy + minxy) / 2
        # targets = np.zeros((1, 6+(points.shape[0]*2)), dtype=np.float32)
        # targets[0,1] += [i for i in range(len(targets))]
        # targets[0,2:4] = cenxy
        # targets[0,4:6] = wh
        # targets[0,6:] = points.flatten()
        # # targets /= self.img_size
        #
        # minxy = (int(targets[0,2]-(targets[0,4]/2)), int(targets[0,3]-(targets[0,5]/2)))
        # maxxy = (int(targets[0,2]+(targets[0,4]/2)), int(targets[0,3]+(targets[0,5]/2)))
        # img = cv2.rectangle(img, minxy, maxxy, 0.5, 1)
        # cv2.imwrite('phantoms/phantom_t{}.png'.format(i), img*255)

    # test()

if __name__ == '__main__':
    main()
