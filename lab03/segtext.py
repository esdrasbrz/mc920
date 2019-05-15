from skimage.morphology import binary_opening, binary_closing
import cv2
import numpy as np
import sys
import os


_img2bin = lambda i: ~((i / 255).astype(np.bool))
_bin2img = lambda i: ((~i) * 255).astype(np.uint8)


def _get_args():
    if len(sys.argv) != 3:
        print("Usage python3 script.py in_image_path out_image_path")
        exit(1)

    return sys.argv[1], sys.argv[2]


def _imshow(imgs):
    for label, img in imgs:
        cv2.imshow(label, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _preprocess(img):
    row = binary_closing(img, selem=np.ones((1,100)))
    column = binary_closing(img, selem=np.ones((200,1)))

    img = row & column
    return img


def _connected_components(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(_bin2img(img), 4, cv2.CV_32S)
    return nlabels, stats


def main():
    # read image
    input_path, output_path = _get_args()
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = _img2bin(img)

    # processing phase
    out = _preprocess(img)
    out = binary_closing(out, selem=np.ones((1,30)))
    _connected_components(out)

    cv2.imwrite(output_path, _bin2img(out))
    

if __name__ == '__main__':
    main()
