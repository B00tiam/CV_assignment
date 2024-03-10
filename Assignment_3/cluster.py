import cv2 as cv
import numpy as np

from assignment import *

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
    # print(labels)
    # get labeled voxel data
    
    # define the color map
    color_map = {0: [255, 0, 0],  # red
                 1: [0, 255, 0],  # green
                 2: [0, 0, 255],  # blue
                 3: [255, 255, 0]}  # yellow

    for i, label in enumerate(labels):
        colors[i] = color_map[label]

    return colors

def cluster_project(projects, colors, index):


    cam_path = './data/cam' + str(index + 1) + '/video.avi'
    # take the 1st frame (frame1) of each cam:
    # get the video
    video_capture = cv.VideoCapture(cam_path)
    if not video_capture.isOpened():
        print("Cannot open file")
        exit()
    ret, frame1 = video_capture.read()
    if not ret:
        print("Cannot get frame")
        exit()

    for i in range(len(projects)):
        x, y = projects[i]
        cv.circle(frame1, (x, y), 1, colors[i], -1)

    cv.imshow('Projected Image', frame1)
    cv.waitKey(0)
    cv.destroyAllWindows()