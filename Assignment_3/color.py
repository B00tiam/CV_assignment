# color model
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

def paint_histogram(hist_list):
    fig, axs = plt.subplots(2, 2)
    counter = -1
    for i in range(2):
        for j in range(2):
            counter += 1
            axs[i, j].plot(hist_list[counter], color='b')

    plt.title("Histograms")
    plt.show()

    return

def histogram(org_img, project_list, color_list):

    # list of histograms
    # get pic
    # origin_path = img_path + '/origin_img.jpg'
    # org_img = cv.imread(origin_path)

    # get hsv and gray pics
    org_gray = cv.cvtColor(org_img, cv.COLOR_BGR2GRAY)
    org_hsv = cv.cvtColor(org_img, cv.COLOR_BGR2HSV)

    hist_list = []

    color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    for l in range(4):
        # get mask using projects from set_voxel_positions
        mask = np.zeros_like(org_gray)
        for i in range(len(project_list)):
            if color_list[i] == color_map[l]:
                mask[project_list[i][1], project_list[i][0]] = 255

        # cv.imshow('Mask', cv.bitwise_and(org_img, org_img, mask=mask))
        # cv.waitKey(0)
        # get histogram
        # color = ('b', 'g', 'r')
        hist = cv.calcHist([org_hsv], [0, 1], mask, [16, 16], [0, 180, 120, 256], accumulate=False)
        total_sum_bins = np.sum(hist)
        # if total_sum_bins > 0:
            # hist = hist / total_sum_bins
        cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
        hist_list.append(hist)

    # cv.destroyAllWindows()
    return hist_list


# hungarian func used for deciding the best match
def hungarian_algorithm(distance_matrix):
    # col_ind is the true order
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    print(row_ind)
    print(col_ind)

    matchs = []     # show the circumstance of order matching
    for i in range(len(row_ind)):
        if row_ind[i] == col_ind[i]:
            matchs.append(-1)
        else:
            matchs.append(col_ind[i])   # the real label order

    return matchs

# calculate distance
def compute_distance(hist_before, hist_after):

    distance = cv.compareHist(hist_before, hist_after, cv.HISTCMP_INTERSECT)   # cv.HISTCMP_CHISQR

    # return float distance
    return distance

def live_matching(video_path, curr_time, curr_frame, video, project_list, color_list_old, hist_list_old):

    # read the video
    if curr_time == 0:
        video = cv.VideoCapture(video_path)
        if not video.isOpened():
            print("Cannot open file")
            exit()
    tar_frame = curr_time + 1
    while curr_frame < tar_frame:
        ret, frame = video.read()
        if not ret:
            print("Cannot get frame")
            exit()
        curr_frame += 1

    # cv.imshow('Frame', frame)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # get the frame
    if curr_time == 0:
        hist_list_new = histogram(frame, project_list, color_list_old)

        return curr_time + 50, curr_frame, video, color_list_old, hist_list_new

    else:
        hist_list_new = histogram(frame, project_list, color_list_old)
        distance_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                # calculate the distance between old & new
                distance_matrix[i, j] = compute_distance(hist_list_old[i], hist_list_new[j])

        print(distance_matrix)
        # use hungarian algorithm:
        matchs = hungarian_algorithm(distance_matrix)
        print(matchs)

        # matching:
        if all(element == -1 for element in matchs):
            return curr_time + 50, curr_frame, video, color_list_old, hist_list_new
        else:
            # match the real labels:
            color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
            for index1 in range(len(color_list_old)):
                for color in range(4):
                    if color_list_old[index1] == color_map[color]:
                        color_list_old[index1] = color
            for index2 in range(len(color_list_old)):
                for label in range(4):
                    if color_list_old[index2] == label and matchs[label] != -1:
                        color_list_old[index2] = -1 * matchs[label]
            color_list_new = []
            for index3 in range(len(color_list_old)):
                # get the absolute number
                color_list_new.append(color_map[abs(color_list_old[index3])])

        return curr_time + 50, curr_frame, video, color_list_new, hist_list_new
