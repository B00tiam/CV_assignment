import numpy as np
import cv2 as cv
import glob

from click import manual_process
from calibrate import calibrate

# Pre-process of img:
# Evaluate the quality of img
def quality_evaluate(img, gray):
    # calculate blur score and brightness
    blur = cv.Laplacian(gray, cv.CV_64F).var()
    bright = np.mean(img)

    # compare
    if blur < 100 or bright < 100:
        return False  # low quality
    else:
        return True  # high quality

# Promote the quality of pics
def promotion(img):
    # enhance edges
    img = cv.Laplacian(img, cv.CV_8U)
    # reduce blur
    aver_kernel = np.ones((3, 3), dtype=np.float32) / 9.0   # define average filter
    img = cv.filter2D(img, -1, aver_kernel)
    return img

def get_corners(path):
    # Specify the path to the images
    image_path_pattern = path + '/*.jpg'

    # Create a list of image file paths
    pre_images = glob.glob(image_path_pattern)
    images = []
    images_invalid = []

    # Evaluate quality

    for fname in pre_images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if quality_evaluate(img, gray) == True:
            images.append(fname)
        else:
            print("low quality: " + fname)

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (6, 5, 0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv.imread(fname)

        # Promotion
        # img = promotion(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)


        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # print(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            # Adjust the size of window
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.resizeWindow('img', img.shape[1], img.shape[0])

            print(fname + "-valid")   # Get the path of valid pics
            '''
            valid pics (temp)
            board1.jpg
            board10.jpg
            board11.jpg
            board12.jpg
            board13.jpg
            board14.jpg
            board16.jpg
            board18.jpg
            board19.jpg
            board2.jpg
            board20.jpg
            board21.jpg
            board22.jpg
            board24.jpg
            board25.jpg
            board3.jpg
            board4.jpg
            board5.jpg
            board7.jpg
            board9.jpg
            
            invalid pics (temp)
            board15.jpg
            board17.jpg
            board23.jpg
            board6.jpg
            board8.jpg
            '''
            # Show the imgs
            cv.imshow('img', img)
            cv.waitKey(1000)   # Adjustable according to user's device


        else:
            print(fname + "-invalid")   # Get the path of valid pics

            # collect the invalid path
            images_invalid.append(fname)

    cv.destroyAllWindows()
    # judge the len of images_invalid
    if len(images_invalid) == 0:
        # Interface to calibration
        calibrate(objpoints, imgpoints)
    else:
        # Interface to click functions
        manual_process(images_invalid, objpoints, imgpoints)


