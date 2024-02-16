import numpy as np
import cv2 as cv
import glob

folder_path = 'C:\\Users\\luiho\\OneDrive\\Desktop\\AI\\AI sem 1 (periods 3-4)\\CompVis\\chessboards'
image_path_pattern = f'{folder_path}\\*.jpg'
# Create a list of image file paths
images = glob.glob(image_path_pattern)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners if the correct number of corners is detected
        if corners2.shape[0] == 54:  # Check if the correct number of corners is detected
            cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.resizeWindow('img', img.shape[1], img.shape[0])

            print(fname + " is valid")

            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(fname + " has an incorrect number of corners.")
    else:
        print(fname + " is invalid")

        def click_event(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                print(x, '', y)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(params['img'], str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
                cv.imshow('img', params['img'])
                params['stored_coordinates'].append((x, y))

                # Draw corners only if the correct number of corners is reached
                if len(params['stored_coordinates']) == 54:
                    cv.drawChessboardCorners(params['img'], (9, 6), np.array(params['stored_coordinates']), True)
                    cv.imshow('img', params['img'])
                    cv.waitKey(500)

        stored_coordinates = []

        cv.setMouseCallback('img', click_event, {'stored_coordinates': stored_coordinates, 'img': img})

        cv.imshow('img', img)
        cv.waitKey(0)

cv.destroyAllWindows()
