import numpy as np
import cv2 as cv

def undistort(img, objpoints, imgpoints):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Calibration
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print the intrinsic matrix and distortion coefficient
    print("Intrinsic matrix: \n", mtx)
    print("Distortion coefficient: \n", dist)

    # Undistort the image
    h, w = img.shape[:2]
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv.undistort(img, mtx, dist, None, new_mtx)
    x, y, w, h = roi

    return undistorted_img, mtx, dist, x, y, w, h


def calibrate(objpoints, imgpoints):
    # Load test image
    testpath = ('chessboards/board25.jpg')
    img = cv.imread(testpath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (6, 5, 0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    axis_3D = np.float32([[6, 0, 0], [0, 6, 0], [0, 0, -6]]).reshape(-1, 3)
    axis_cube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    undistorted_img, mtx, dist, x, y, w, h = undistort(img, objpoints, imgpoints)
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    cv.namedWindow('undistorted img', cv.WINDOW_NORMAL)
    cv.resizeWindow('undistorted img', undistorted_img.shape[1], img.shape[0])
    cv.imshow('undistorted img', undistorted_img)
    cv.waitKey(10000)

    # Draw 3D axes
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        # Draw the coordinate
        imgpts_3D, jac = cv.projectPoints(axis_3D, rvecs, tvecs, mtx, dist)
        corner = tuple(np.round(corners2[0].ravel()).astype(int))
        img = cv.line(img, corner, tuple(np.round(imgpts_3D[0].ravel()).astype(int)), (255, 0, 0), 10)
        img = cv.line(img, corner, tuple(np.round(imgpts_3D[1].ravel()).astype(int)), (0, 255, 0), 10)
        img = cv.line(img, corner, tuple(np.round(imgpts_3D[2].ravel()).astype(int)), (0, 0, 255), 10)

        # Draw the cube
        imgpts_cube, jac = cv.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)
        imgpts_cube = np.int32(imgpts_cube).reshape(-1, 2)
        # draw ground floor in green
        img = cv.drawContours(img, [imgpts_cube[:4]], -1, (0, 255, 0), -3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(imgpts_cube[i]), tuple(imgpts_cube[j]), (255), 3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts_cube[4:]], -1, (0, 0, 255), 3)

        # Show the image
        # Adjust the size of window
        cv.namedWindow('painted img', cv.WINDOW_NORMAL)
        cv.resizeWindow('painted img', img.shape[1], img.shape[0])
        # cv.imshow('Original Image', img)
        cv.imshow('painted img', img)
        cv.waitKey(10000)

    cv.destroyAllWindows()


