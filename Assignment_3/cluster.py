import cv2 as cv
import numpy as np

from assignment import set_voxel_positions


def set_voxel_colors(positions, colors):

    data = np.array(positions, dtype=np.float32)

    # set K-means params
    k = 4  # num of cluster
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # get labels
    _, labels, centers = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    for i, label in enumerate(labels):
        colors[i] = [label, label, label]

    return colors