from skimage import io
from scipy import ndimage
import numpy as np
import sys


def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def main():
    input_path, output_path = _get_args()
    img = io.imread(input_path).astype(np.uint8)
    print(img.shape)



if __name__ == '__main__':
    main()
