from skimage import io
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import time
import sys


def _get_args():
    if len(sys.argv) != 4:
        print("Usage python3 reduce_colors.py in_image_path out_image_path n_colors")
        exit(1)

    return sys.argv[1], sys.argv[2], int(sys.argv[3])


def _extract_features(img):
    h, w, ch = img.shape
    features = img.reshape((h*w, ch))
    return features


def _clusterize(colors, n_clusters):
    t = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0,
                             batch_size=2**12).fit(colors)
    print('Time to clusterize: {} s'.format(time.time() - t))

    return kmeans.labels_, kmeans.cluster_centers_


def _reduce_colors(output_shape, labels, labels_colors):
    out = np.zeros((labels.size, 3))
    
    for label, color in enumerate(labels_colors):
        out[labels == label] = color

    out = out.reshape(output_shape).astype(np.uint8)
    return out


def main():
    input_path, output_path, n_colors = _get_args()
    img = io.imread(input_path).astype(np.float32)
    colors = _extract_features(img)

    labels, labels_colors = _clusterize(colors, n_colors)
    out = _reduce_colors(img.shape, labels, labels_colors)
    io.imsave(output_path, out)


if __name__ == '__main__':
    main()
