from skimage import io
import numpy as np
import sys

def _get_args():
    if len(sys.argv) != 5:
        print("Usage python3 script.py in_image_path in2_image_path out_image_path alpha")
        exit(1)

    return sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

def _combine(img1, img2, alpha):
    out = img1 * alpha + img2 * (1-alpha)
    return out.astype(np.uint8)

def main():
    in1_path, in2_path, output_path, alpha = _get_args()
    img1 = io.imread(in1_path)
    img2 = io.imread(in2_path)
    
    out = _combine(img1, img2, alpha)
    io.imsave(output_path, out)

if __name__ == '__main__':
    main()
