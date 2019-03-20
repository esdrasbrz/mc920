from skimage import io
import numpy as np
import sys

MOSAIC_SEQUENCE = [6, 11, 13, 3, 8, 16, 1, 9, 12, 14, 2, 7, 4, 15, 10, 5]
MOSAIC_QTY = 4

def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]

def _get_tile_images(image, nrows=4, ncols=4):
    height, width = image.shape
    strides = image.strides 

    height //= nrows
    width //= ncols

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows * ncols, height, width),
        strides=(height * width * strides[0], *strides),
        writeable=False
    )

def _get_slice(img, nblock):
    height, width = img.shape
    slice_h, slice_w = height // MOSAIC_QTY, width // MOSAIC_QTY
    nblock = nblock - 1 # normalize to begin in 0

    xblock = nblock % MOSAIC_QTY
    yblock = nblock // MOSAIC_QTY

    xslice = (xblock * slice_w, xblock * slice_w + slice_w)
    yslice = (yblock * slice_h, yblock * slice_h + slice_h)
    return xslice, yslice

def main():
    input_path, output_path = _get_args()
    img = io.imread(input_path)
    
    out = np.zeros_like(img)
    for i, block in enumerate(MOSAIC_SEQUENCE):
        xslice, yslice = _get_slice(img, block) 
        xslice_out, yslice_out = _get_slice(img, i+1)
    
        out[yslice_out[0]:yslice_out[1], xslice_out[0]:xslice_out[1]] = img[yslice[0]:yslice[1], xslice[0]:xslice[1]]
    
    io.imsave(output_path, out)

if __name__ == '__main__':
    main()
