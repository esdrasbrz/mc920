from skimage.morphology import binary_opening, binary_closing
import config as cfg
import cv2
import numpy as np
import sys
import os


_img2bin = lambda i: ~((i / 255).astype(np.bool))
_bin2img = lambda i: ((~i) * 255).astype(np.uint8)


def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def _imshow(imgs):
    for label, img in imgs:
        cv2.imshow(label, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _preprocess(img):
    row = binary_closing(img, selem=cfg.ROW_CLOSING_SELEM)
    column = binary_closing(img, selem=cfg.ROW_CLOSING_SELEM)

    img = row & column
    return img


def _connected_components(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(255*img.astype(np.uint8), 4, cv2.CV_32S)
    return stats


def _draw_rect(img, stats, ratios):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for label, stat in enumerate(stats):
        min_x, min_y, w, h, area = stat
        area_ratio = ratios[label]

        cv2.rectangle(img, (min_x, min_y), (min_x+w, min_y+h), (0, 255, 0), 3)
        cv2.putText(img, '%.3f' % area_ratio, (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def _calc_ratios(img, stats):
    ratios = []

    for label in stats:
        min_x, min_y, w, h, area = label
        ratios.append(area / (w*h))

    return ratios


def main():
    # read image
    input_path, output_path = _get_args()
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = _img2bin(input_img)

    # processing phase
    img = _preprocess(img)
    img = binary_closing(img, selem=cfg.POS_CLOSING_SELEM)
    stats = _connected_components(img)
    ratios = _calc_ratios(img, stats)

    blobs_img = _draw_rect(_bin2img(img), stats, ratios)
    cv2.imwrite(output_path, blobs_img)
    

if __name__ == '__main__':
    main()
