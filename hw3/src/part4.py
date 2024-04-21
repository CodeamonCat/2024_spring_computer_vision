import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)


def find_homography_ransac(src_pts, dst_pts, max_iters=1000, threshold=5.0):
    best_H = None
    max_inliers = 0

    for _ in range(max_iters):
        indices = random.sample(range(len(src_pts)), 4)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        H = solve_homography(src_sample, dst_sample)

        onerow = np.ones((1, len(src_pts)))
        dst_pts_transformed = np.concatenate((dst_pts.T, onerow), axis=0)

        src_pts_transformed = np.dot(
            H, np.concatenate((src_pts.T, onerow), axis=0))
        src_pts_transformed = np.divide(src_pts_transformed,
                                        src_pts_transformed[-1, :])
        dist = np.linalg.norm(
            (src_pts_transformed - dst_pts_transformed)[:-1, :], ord=1, axis=0)
        inliers = sum(dist < threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

    return best_H


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]  # trainImage: box_in_scene.png
        im2 = imgs[idx + 1]  # queryImage: box.png

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # matches = bf.match(queryImage, trainImage)
        matches = bf.match(des2, des1)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        # src = queryImage, dst = trainImage
        src_pts = np.array([kp2[m.queryIdx].pt
                            for m in matches]).reshape(-1, 2)
        dst_pts = np.array([kp1[m.trainIdx].pt
                            for m in matches]).reshape(-1, 2)

        # TODO: 2. apply RANSAC to choose best H
        best_H = find_homography_ransac(src_pts, dst_pts)

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    out = dst
    return out


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [
        cv2.imread('../resource/frame{:d}.jpg'.format(x))
        for x in range(1, FRAME_NUM + 1)
    ]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
