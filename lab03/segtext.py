from skimage.morphology import binary_opening, binary_closing, binary_dilation
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
    column = binary_closing(img, selem=cfg.COLUMN_CLOSING_SELEM)

    img = row & column
    return img


def _process_words(img):
    img = binary_dilation(img, selem=cfg.WORDS_DILATION_SELEM)
    img = binary_closing(img, selem=cfg.WORDS_CLOSING_SELEM)

    return img


def _connected_components(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(255*img.astype(np.uint8), 4, cv2.CV_32S)
    return stats[1:]


def _count_transitions(img, min_x, min_y, w, h):
    img_h, img_w = img.shape
    ntrans = 0

    for dy in range(h):
        for dx in range(w):
            x = min_x+dx
            y = min_y+dy

            if x < img_w and y < img_h and img[y,x]:
                if x-1 >= min_x and not img[y,x-1]:
                    ntrans += 1
                if y-1 >= min_y and not img[y-1,x]:
                    ntrans += 1

    return ntrans


def _calc_ratios(img, stats):
    ratios = []

    for label in stats:
        min_x, min_y, w, h, area = label

        ntrans = _count_transitions(img, min_x, min_y, w, h)
        area_ratio = area / (w*h)
        trans_ratio = ntrans / area
        ratios.append((area_ratio, trans_ratio))

    return ratios


def _apply_thr(stats, ratios):
    filtered_stats = []

    for label, stat in enumerate(stats):
        area_ratio, trans_ratio = ratios[label]

        if cfg.TEXT_AREA_RATIO_THR[0] < area_ratio < cfg.TEXT_AREA_RATIO_THR[1] \
                and cfg.TEXT_TRANS_RATIO_THR[0] < trans_ratio < cfg.TEXT_TRANS_RATIO_THR[1]:
            filtered_stats.append(stat)

    return filtered_stats


def _stats2mask(stats, shape):
    mask = np.zeros(shape)

    for stat in stats:
        min_x, min_y, w, h, __ = stat
        mask[min_y:min_y+h, min_x:min_x+w] = 1

    return mask.astype(np.bool)


def _draw(img, stats, ratios=None):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for label, stat in enumerate(stats):
        min_x, min_y, w, h, area = stat
        cv2.rectangle(img, (min_x, min_y), (min_x+w, min_y+h), (0, 255, 0), 3)

        if ratios:
            area_ratio, trans_ratio = ratios[label]
            cv2.putText(img, '%.3f | %.3f' % (area_ratio, trans_ratio), (min_x, min_y), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def main():
    # read image
    input_path, output_path = _get_args()
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = _img2bin(input_img)

    # text processing phase
    img = _preprocess(img)
    img = binary_closing(img, selem=cfg.POS_CLOSING_SELEM)
    stats = _connected_components(img)
    ratios = _calc_ratios(img, stats)

    blobs_img = _draw(_bin2img(img), stats, ratios=ratios)
    cv2.imwrite(os.path.join(output_path, 'blobs_img.png'), blobs_img)

    text_stats = _apply_thr(stats, ratios)
    text_blobs_img = _draw(input_img, text_stats)
    cv2.imwrite(os.path.join(output_path, 'text_blobs_img.png'), text_blobs_img)

    # segmentating words
    words = _process_words(_img2bin(input_img))
    cv2.imwrite(os.path.join(output_path, 'words_img.png'), _bin2img(words))

    words_stats = _connected_components(words)
    text_mask = _stats2mask(text_stats, img.shape)
    words_mask = _stats2mask(words_stats, img.shape) & text_mask

    words_stats = _connected_components(words_mask)
    words_blobs_img = _draw(input_img, words_stats)
    cv2.imwrite(os.path.join(output_path, 'words_blobs_img.png'), words_blobs_img)

    print('number of lines: %d' % len(text_stats))
    print('number of words: %d' % len(words_stats))


if __name__ == '__main__':
    main()
