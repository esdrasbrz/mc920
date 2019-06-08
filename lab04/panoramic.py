import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os


PLOT_DESCRIPTORS = False
REPROJ_THR = 4.0


def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 panoramic.py in_image_A_path in_image_B_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2], sys.argv[3]


def _draw_keypoints(img, kp):
    img_draw = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    return img_draw


def _surf_descriptor(img, thr=400):
    surf = cv2.xfeatures2d.SURF_create(thr)
    kp, des = surf.detectAndCompute(img, None)
    img_draw = _draw_keypoints(img, kp)

    return kp, des, img_draw


def _brief_descriptor(img):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img, None)
    kp, des = brief.compute(img, kp)
    img_draw = _draw_keypoints(img, kp)

    return kp, des, img_draw


def _orb_descriptor(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    img_draw = _draw_keypoints(img, kp)

    return kp, des, img_draw


def _compute_descriptor(img, descriptor):
    kp = descriptor.detect(img, None)
    kp, des = descriptor.detectAndCompute(img, None)
    img_draw = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)

    return kp, des, img_draw


def _plot_descriptors(img):
    for kp, des, img_des in [_surf_descriptor(img, thr=400),
                             _surf_descriptor(img, thr=4000),
                             _brief_descriptor(img), 
                             _orb_descriptor(img)]:
        _plot(img_des)


def _match_keypoints(img1, img2, kp1, kp2, des1, des2, n_matches_plot=50):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches_plot], None, flags=2)
    return matches, matches_img


def _find_homography(matches, kpA, kpB, reproj_thr):
    if len(matches) < 4:
        return None

    # construct the two sets of points
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # compute the homography between the two sets of points
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thr)

    return H, status


def _warp_perspective(imgA, imgB, H):
    img_out = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], max(imgA.shape[0], imgB.shape[0])))
    img_out[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return img_out


def _save_H(H, output_path):
    with open(os.path.join(output_path, 'homography_matrix.m'), 'w') as fout:
        for row in H:
            fout.write(' '.join(map(str, row)))
            fout.write('\n')


def _plot(img):
    plt.imshow(img)
    plt.show()


def main():
    imgA_path, imgB_path, output_path = _get_args()

    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)
    imgA_color = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
    imgB_color = cv2.imread(imgB_path, cv2.IMREAD_COLOR)

    descriptor_fn = _brief_descriptor

    if PLOT_DESCRIPTORS:
        _plot_descriptors(imgA)

    # calculate descriptors for each image
    kpA, desA, __ = descriptor_fn(imgA)
    kpB, desB, __ = descriptor_fn(imgB)
    matches, matches_img = _match_keypoints(imgA, imgB, kpA, kpB, desA, desB)
    cv2.imwrite(os.path.join(output_path, 'matches.jpeg'), matches_img)

    # find homography matrix
    H, status = _find_homography(matches, kpA, kpB, REPROJ_THR)
    _save_H(H, output_path)
    img_out = _warp_perspective(imgA_color, imgB_color, H)
    cv2.imwrite(os.path.join(output_path, 'panorama.jpeg'), img_out)


if __name__ == '__main__':
    main()
