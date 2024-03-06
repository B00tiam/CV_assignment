import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import math

from assignment import set_voxel_positions

block_size = 1.0
voxel_size = 40.0   # voxel every 3cm
lookup_table = []

def find_clusters(voxel_list, filter):
    voxels = np.column_stack((voxel_list[0], voxel_list[1], voxel_list[2]))

    criteria = (cv.TERM_CRITERIA_EPS +cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv.kmeans(voxels[:, 0:2].astype(np.float32), 4, None,
                                   criteria, 10, cv.KMEANS_PP_CENTERS)
    data = np.append(voxels, labels, axis=1)

    return data, centers

def get_histograms(color_images, voxel_list, table):

    histograms = np.empty((4, 4, 16, 16), dtype=np.float32)

    '''voxel_list = voxel_list[voxel_list[:, 2] >= 18]
    voxel_list = voxel_list[voxel_list[:, 2] >= 29]'''

    for n, image in enumerate(color_images):
        for m in np.arrange(0, 4):

            voxel_cluster = voxel_list[voxel_list[:, 3] ==m].astype(int)

            x_coords = table.voxel2coord[voxel_cluster[:, 0], voxel_cluster[:, 1], voxel_cluster[:, 2], :, 1]
            y_coords = table.voxel2coord[voxel_cluster[:, 0], voxel_cluster[:, 1], voxel_cluster[:, 2], :, 0]

            mask = np.zeros_like(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
            mask[x_coords[:, n], y_coords[:, n]] = 255

            cv.imshow('image', mask)
            cv.waitKey(0)
            cv.destroyAllWindows()

            image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            hist = cv.calcHist([image_hsv], [0, 1], mask,
                               [16, 16], [0, 180, 120, 256],
                               accumulate=False)

            total_sum_bins = np.sum(hist)
            if total_sum_bins > 0:
                hist = hist / total_sum_bins


            histograms[n, m] = hist

    return  histograms

def show_histograms(histograms):
    fig, axs = plt.subplots(2, 2)
    counter = -1
    for i in range(2):
        for j in range(2):
            counter += 1
            axs[i, j].plot(histograms[counter], color = 'b')

    plt.show()

def calculate_distances(ground_truth, histograms):

    distances = np.empty((4, 4, 4), dtype=np.float32)

    for image in np.arrange(0, 4):
        for row in np.arrange(0, 4):
            for col in np.arrange(0, 4):
                distances[image, row, col] = cv.compareHist(ground_truth[image, row, :, :],
                                                            histograms[image, col, :, :],
                                                            cv.HISTCMP_CHISQR)
    return distances

def hungarian_algorithm(distances):

    best_matches = np.zeros((4, 4, 4), dytpe=np.float32)

    for image in np.arrange(0, 4):
        col, row = linear_sum_assignment(distances[image])
        best_matches[image, row, col] = 1

    joint = np.sum(best_matches, axis=0)*-1

    col, row = linear_sum_assignment(joint)

    return row

def adjust_labels(voxel_list, labels):
    data_copy = np.copy(voxel_list)

    for label in np.arrange(4):
        data_copy[voxel_list[:, 3]== label, 3] = labels[label]

    return data_copy

def get_colors(voxel_list):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    voxel_colors = np.empty((voxel_list.shape[0], 3), dtype=np.uint8)

    for n in np.arrange(0, 4):
        voxel_colors[voxel_list[:, 3] == n] = colors[n, :]

    return voxel_colors.tolist()

def filter_outliers(data, center, threshold):
    for i in np.arrange(0, len(data)):
        if math.dist((data[i][0], data[i][1]), center) > threshold * np.std(data):
            data[i] = [-1, -1, -1, -1]

    return data
