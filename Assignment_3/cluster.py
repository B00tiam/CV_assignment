import cv2 as cv
import numpy as np

from assignment import set_voxel_positions


def set_voxel_colors(positions, colors):

    # positions: x * block_size - width / 2, y * block_size, z * block_size - depth / 2
    # Get x * block_size - width / 2 & z * block_size - depth / 2 for clustering
    data = [[row[0], row[-1]] for row in positions]
    data_np = np.array(data, dtype=np.float32)

    # set K-means params
    k = 4  # num of cluster
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # get labels, centers(4-elements)
    _, labels, centers = cv.kmeans(data_np, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    print(labels)
    # define the color map
    color_map = {0: [255, 0, 0],  # red
                 1: [0, 255, 0],  # green
                 2: [0, 0, 255],  # blue
                 3: [255, 255, 0]}  # yellow

    for i, label in enumerate(labels):
        colors[i] = color_map[label]

    return colors