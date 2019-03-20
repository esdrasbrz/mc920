from skimage import io
import numpy as np
import sys

def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 script.py in_image_path out_image_path gamma")
        exit(1)

    return sys.argv[1], sys.argv[2], float(sys.argv[3])

def _normalize(A):
    return A.astype(np.float32) / 255

def _denormalize(A):
    return (A * 255).astype(np.uint8)

def _transform(A, gamma):
    return A**(1/gamma)

def main():
    input_path, output_path, gamma = _get_args()
    input_img = io.imread(input_path)

    A = _normalize(input_img)
    A = _transform(A, gamma)
    img = _denormalize(A)

    io.imsave(output_path, img)

if __name__ == '__main__':
    main()
