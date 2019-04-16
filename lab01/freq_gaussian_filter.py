from skimage import io
from scipy import fftpack
import numpy as np
import os
import sys


def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 script.py in_image_path out_image_path d0")
        exit(1)

    return sys.argv[1], sys.argv[2], int(sys.argv[3])


def _normalize(img):
    img = img-img.min()
    img = 255*img / img.max()

    return img.astype(np.uint8)


def _output_images(output_path, img_freq, out_freq, out_spacial):
    filename = os.path.basename(output_path)
    path = os.path.dirname(output_path)

    io.imsave(os.path.join(path, '{}-{}'.format('in-freq', filename)), img_freq)
    io.imsave(os.path.join(path, '{}-{}'.format('out-freq', filename)), out_freq)
    io.imsave(os.path.join(path, '{}-{}'.format('out', filename)), out_spacial)


def _generate_gaussian_transfer_function(shape, d0):
    h, w = shape 
    
    gtransfer = np.zeros(shape)
    dist2 = lambda x, y: (x-w//2)**2 + (y-h//2)**2
    for y in range(h):
        for x in range(w):
            gtransfer[y][x] = -dist2(x, y)**2 / (2 * d0**2)
    
    return np.exp(gtransfer)


def _apply_gaussian(img_freq, d0):
    transfer = _generate_gaussian_transfer_function(img_freq.shape, d0)
    return np.multiply(transfer, img_freq)


def main():
    # utils image functions
    freq2spacial = lambda freq: _normalize(np.abs(fftpack.ifft2(fftpack.ifftshift(freq))))
    freq2plot = lambda freq: _normalize(20 * np.log(np.abs(freq)))

    # read input
    input_path, output_path, d0 = _get_args()
    img = io.imread(input_path).astype(np.float32)
    
    # calculate fft of input image
    img_freq = fftpack.fftshift(fftpack.fft2(img))

    # apply gaussian filter in frequency domain
    out_freq = _apply_gaussian(img_freq, d0)

    _output_images(output_path, freq2plot(img_freq), freq2plot(out_freq), freq2spacial(out_freq))

if __name__ == '__main__':
    main()
