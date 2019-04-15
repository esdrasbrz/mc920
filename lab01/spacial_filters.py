from skimage import io
from scipy import ndimage
import numpy as np
import sys


## Filters to be applied
H1 = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
])

H2 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256

H3 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

H4 = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 script.py in_image_path out_image_path filter(h1 | h2 | h3 | h4 | h34)")
        exit(1)

    return sys.argv[1], sys.argv[2], sys.argv[3].lower()


def _normalize(img):
    img = img-img.min()
    img = 255*img / img.max()

    return img.astype(np.uint8)


def _conv2d(img, h):
    return ndimage.convolve(img, h, mode='constant')


def main():
    # read input
    input_path, output_path, filter_sel = _get_args()
    img = io.imread(input_path).astype(np.float32)
    
    filters = {
        'h1': lambda img: _normalize(_conv2d(img, H1)),
        'h2': lambda img: _normalize(_conv2d(img, H2)),
        'h3': lambda img: _normalize(_conv2d(img, H3)),
        'h4': lambda img: _normalize(_conv2d(img, H4)),
        'h34': lambda img: _normalize(np.sqrt(_conv2d(img, H3)**2 + _conv2d(img, H4)**2))
    }

    # apply filter
    out = filters[filter_sel](img)
    # save output image
    io.imsave(output_path, out)


if __name__ == '__main__':
    main()
