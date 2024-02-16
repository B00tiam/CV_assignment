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
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners and lines
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', img.shape[1], img.shape[0])

        print(fname + " is valid")

        # Draw lines connecting the chessboard corners
        for i in range(1, len(corners2)):
            cv.line(img, tuple(corners2[i-1][0]), tuple(corners2[i][0]), (0, 255, 0), 2)

        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print(fname + " is invalid")

        def click_event(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                print(x, '', y)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(params['img'], str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
                params['stored_coordinates'].append((x, y))

                # Check if enough points are available to draw chessboard corners
                if len(params['stored_coordinates']) >= 6 * 9:
                    # Draw and display the corners and lines
                    cv.drawChessboardCorners(params['img'], (9,6), np.array(params['stored_coordinates']), True)
                    for i in range(1, len(params['stored_coordinates'])):
                        cv.line(params['img'], tuple(params['stored_coordinates'][i-1]), tuple(params['stored_coordinates'][i]), (0, 255, 0), 2)

                    cv.imshow('img', params['img'])
                    cv.waitKey(500)

        stored_coordinates = []

        cv.setMouseCallback('img', click_event, {'stored_coordinates': stored_coordinates, 'img': img})

        cv.imshow('img', img)
        cv.waitKey(0)

cv.destroyAllWindows()
