from math import floor
import cv2
import numpy as np
import sys
import os

def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def dithering(img, zigzag=True):
    img = img.astype(np.int16)
    out = np.zeros_like(img).astype(np.uint8)
    h, w = img.shape

    apply_thr = lambda x: 255 * floor(x / 128)

    for i in range(h):
        if zigzag:
            rev = i % 2 == 1
        else:
            rev = False

        for j in range(w)[::-1 if rev else 1]:
            # set pixel in output image
            out[i,j] = apply_thr(img[i,j])

            # propagates the error
            err = img[i,j] - out[i,j]

            if not rev:
                if j+1 < w:
                    img[i,j+1] += round(7/16 * err)
                if j+1 < w and i+1 < h:
                    img[i+1,j+1] += round(1/16 * err)
                if i+1 < h:
                    img[i+1,j] += round(5/16 * err)
                if i+1 < h and j-1 >= 0:
                    img[i+1,j-1] += round(3/16 * err)
            else:
                if j-1 >= 0:
                    img[i,j-1] += round(7/16 * err)
                if j+1 < w and i+1 < h:
                    img[i+1,j+1] += round(3/16 * err)
                if i+1 < h:
                    img[i+1,j] += round(5/16 * err)
                if i+1 < h and j-1 >= 0:
                    img[i+1,j-1] += round(1/16 * err)

    return out


def main():
    input_path, output_path = _get_args()
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    out = dithering(img, zigzag=True)
    cv2.imwrite(output_path, out)
    

if __name__ == '__main__':
    main()
