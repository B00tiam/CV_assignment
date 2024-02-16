import numpy as np
import cv2 as cv
import glob

from click import click_event
from calibrate import calibrate
from functools import partial



def get_corners():
    # Specify the path to the images
    image_path_pattern = 'chessboards/*.jpg'

    # Create a list of image file paths
    images = glob.glob(image_path_pattern)

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
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)


        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            # Adjust the size of window
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.resizeWindow('img', img.shape[1], img.shape[0])

            print(fname + "-valid")   # Get the path of valid pics
            '''
            valid pics (temp)
            chessboards\board1.jpg
            chessboards\board10.jpg
            chessboards\board11.jpg
            chessboards\board12.jpg
            chessboards\board13.jpg
            chessboards\board14.jpg
            chessboards\board15.jpg
            chessboards\board16.jpg
            chessboards\board17.jpg
            chessboards\board18.jpg
            chessboards\board19.jpg
            chessboards\board2.jpg
            chessboards\board20.jpg
            chessboards\board21.jpg
            chessboards\board22.jpg
            chessboards\board24.jpg
            chessboards\board25.jpg
            chessboards\board3.jpg
            chessboards\board4.jpg
            chessboards\board5.jpg
            chessboards\board7.jpg
            chessboards\board9.jpg
            
            invalid pics (temp)
            chessboards\board23.jpg
            chessboards\board6.jpg
            chessboards\board8.jpg
            '''
            # Show the imgs
            cv.imshow('img', img)
            cv.waitKey(15000)   # Adjustable according to user's device

            # Interface to calibration
            calibrate(objpoints, imgpoints, gray, img)

        else:
            print(fname + "-invalid")   # Get the path of valid pics

            # Interface to click event:
            # Reshape the window
            cv.namedWindow('image', cv.WINDOW_NORMAL)
            cv.resizeWindow('image', img.shape[1], img.shape[0])
            cv.imshow('img', img)
            # cv.setMouseCallback('img', partial(click_event, img=img))
            cv.setMouseCallback('img', lambda event, x, y, flags, img=img: click_event(event, x, y, flags, img))

            cv.waitKey(10000)

    cv.destroyAllWindows()
