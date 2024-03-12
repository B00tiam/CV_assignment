# color model
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogram(img_path, project_list, color_list):

    # get pic
    origin_path = img_path + '/origin_img.jpg'

    org_img = cv.imread(origin_path)

    # get hsv and gray pics
    org_gray = cv.cvtColor(org_img, cv.COLOR_BGR2GRAY)
    org_hsv = cv.cvtColor(org_img, cv.COLOR_BGR2HSV)

    color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    for l in range(4):
        # get mask using projects from set_voxel_positions
        mask = np.zeros_like(org_gray)
        for i in range(len(project_list)):
            if color_list[i] == color_map[l]:
                mask[project_list[i][1], project_list[i][0]] = 255

        cv.imshow('Mask', cv.bitwise_and(org_img, org_img, mask=mask))
        cv.waitKey(0)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([org_img], [i], mask, [256], [0, 256])

            plt.plot(hist, color=col)
            plt.xlim([0, 256])

        plt.title("Histogram")
        plt.show()

    cv.destroyAllWindows()

    # get histogram
    '''   
    '''

# hungarian func used for deciding the best match
# def hungarian_algorithm(distances):