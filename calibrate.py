import numpy as np
import cv2 as cv


def calibrate(objpoints, imgpoints, gray, img):
    # Load test image
    testpath = "chessboards/board8.jpg"
    img = cv.imread(testpath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the intrinsic matrix and distortion coefficient
    print("Intrinsic matrix: \n", mtx)
    print("Distortion coefficient: \n", dist)

    # Undistort the image
    undistorted_img = cv.undistort(img, mtx, dist, None, mtx)

    # Draw 3D axes

    # Show the image
    # Adjust the size of window
    cv.namedWindow('undistorted img', cv.WINDOW_NORMAL)
    cv.resizeWindow('undistorted img', undistorted_img.shape[1], undistorted_img.shape[0])
    # cv.imshow('Original Image', img)
    cv.imshow('undistorted img', undistorted_img)
    cv.waitKey(10000)
    cv.destroyAllWindows()


