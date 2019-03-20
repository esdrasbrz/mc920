from skimage import io
import numpy as np
import sys

def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 script.py in_image_path out_image_path nbit")
        exit(1)

    nbit = int(sys.argv[3])
    assert nbit >= 0 and nbit < 8
    return sys.argv[1], sys.argv[2], nbit

def _bit_plan(img, nbit):
    # apply mask
    mask = 1 << nbit
    img = (img & mask) / mask * 255
    img = img.astype(np.uint8)

    return img

def main():
    input_path, output_path, nbit = _get_args()
    img = io.imread(input_path)
    
    img = _bit_plan(img, nbit)

    io.imsave(output_path, img)

if __name__ == '__main__':
    main()
