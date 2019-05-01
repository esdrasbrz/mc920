from skimage import io, exposure
from scipy import ndimage
from math import floor
import numpy as np
import sys

HALF_TONING_MATRIX = [
    [6, 8, 4],
    [1, 0, 3],
    [5, 2, 7]
]
HALF_TONING_SIZE = 9
HALF_TONING_DIM = 3

def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def normalize(m):
    return np.round(m / np.max(m) * HALF_TONING_SIZE).astype(np.uint8)


def create_half_toning_matrix(image):
    h, w = image.shape

    m = np.array(w * h * HALF_TONING_MATRIX)
    m = m.reshape(h, w, HALF_TONING_DIM, HALF_TONING_DIM) \
         .swapaxes(1, 2) \
         .reshape(h*HALF_TONING_DIM, w*HALF_TONING_DIM)

    return m


def dithering(image, out):
    h, w = out.shape

    for i in range(h):
        for j in range(w):
            if out[i, j] < image[floor(i / HALF_TONING_DIM), floor(j / HALF_TONING_DIM)]:
                out[i, j] = 255
            else:
                out[i, j] = 0

    return out.astype(np.uint8)


def main():
    input_path, output_path = _get_args()
    img = io.imread(input_path)
    print(img.shape)

    img = normalize(img)
    out = create_half_toning_matrix(img)
    out = dithering(img, out)

    io.imsave(output_path, out)
    

if __name__ == '__main__':
    main()
