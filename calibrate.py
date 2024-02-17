import numpy as np
import cv2 as cv


def calibrate(objpoints, imgpoints):
    # Load test image
    testpath = 'chessboard/board15.jpg'
    img = cv.imread(testpath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (6, 5, 0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    axis = np.float32([[5, 0, 0], [0, 5, 0], [0, 0, -5]]).reshape(-1, 3)

    # Calibration
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the intrinsic matrix and distortion coefficient
    print("Intrinsic matrix: \n", mtx)
    print("Distortion coefficient: \n", dist)

    # Undistort the image
    # undistorted_img = cv.undistort(img, mtx, dist, None, mtx)

    # Draw 3D axes
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        corner = tuple(np.round(corners2[0].ravel()).astype(int))
        img = cv.line(img, corner, tuple(np.round(imgpts[0].ravel()).astype(int)), (255, 0, 0), 10)
        img = cv.line(img, corner, tuple(np.round(imgpts[1].ravel()).astype(int)), (0, 255, 0), 10)
        img = cv.line(img, corner, tuple(np.round(imgpts[2].ravel()).astype(int)), (0, 0, 255), 10)
        # Show the image
        # Adjust the size of window
        cv.namedWindow('painted img', cv.WINDOW_NORMAL)
        cv.resizeWindow('painted img', img.shape[1], img.shape[0])
        # cv.imshow('Original Image', img)
        cv.imshow('painted img', img)
        cv.waitKey(20000)

    cv.destroyAllWindows()


