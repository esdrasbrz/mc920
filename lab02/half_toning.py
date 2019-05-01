from math import floor
import cv2
import numpy as np
import sys
import os

HALF_TONING_MATRIX = {
    '33': {
        'm': [
            [6, 8, 4],
            [1, 0, 3],
            [5, 2, 7]
        ],
        'dim': 3,
        'size': 9
    },
    'bayer': {
        'm': [
            [0, 12, 3, 15],
            [8, 4, 11, 7],
            [2, 14, 1, 13],
            [10, 6, 9, 5]
        ],
        'dim': 4,
        'size': 16
    }
}


def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def normalize(m, size):
    return np.round(m / np.max(m) * size).astype(np.uint8)


def create_half_toning_matrix(image, ht_m, ht_dim):
    h, w = image.shape

    m = np.array(w * h * ht_m)
    m = m.reshape(h, w, ht_dim, ht_dim) \
         .swapaxes(1, 2) \
         .reshape(h*ht_dim, w*ht_dim)

    return m


def dithering(image, out, ht_dim):
    h, w = out.shape

    for i in range(h):
        for j in range(w):
            if out[i, j] < image[floor(i / ht_dim), floor(j / ht_dim)]:
                out[i, j] = 255
            else:
                out[i, j] = 0

    return out.astype(np.uint8)


def main():
    input_path, output_path = _get_args()
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)

    for ht in HALF_TONING_MATRIX:
        img = normalize(img, HALF_TONING_MATRIX[ht]['size'])
        out = create_half_toning_matrix(img, HALF_TONING_MATRIX[ht]['m'], HALF_TONING_MATRIX[ht]['dim'])
        out = dithering(img, out, HALF_TONING_MATRIX[ht]['dim'])

        filename = os.path.basename(output_path)
        path = os.path.dirname(output_path)
        cv2.imwrite(os.path.join(path, '%s-%s' % (ht, filename)), out)
    

if __name__ == '__main__':
    main()
