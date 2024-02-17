import numpy as np
import cv2 as cv
import glob

def interpolate_internal_corners(external_corners, rows, cols):
    external_corners = external_corners.reshape(4, 2)
    top_edge = np.linspace(external_corners[0], external_corners[1], num=cols)
    bottom_edge = np.linspace(external_corners[3], external_corners[2], num=cols)
    return np.vstack([np.linspace(top_edge[i], bottom_edge[i], num=rows) for i in range(cols)])

def determine_chessboard_orientation(external_corners):
    # Calculate distances between clicked corners
    distances = np.linalg.norm(np.diff(external_corners, axis=0), axis=1)

    # Determine orientation based on distances
    if np.allclose(distances[::2], distances[0]):
        return 6, 9  # Orientation: 6x9
    elif np.allclose(distances[1::2], distances[1]):
        return 9, 6  # Orientation: 9x6
    else:
        return None  # Unable to determine orientation


folder_path = 'chessboard'
image_path_pattern = f'{folder_path}\\*.jpg'
images = glob.glob(image_path_pattern)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

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

        stored_coordinates = []

        def click_event(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                print(x, '', y)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(params['img'], str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
                cv.imshow('img', params['img'])
                params['stored_coordinates'].append((x, y))

                # Draw corners only if enough corners are reached
                if len(params['stored_coordinates']) >= 4:
                    external_corners = np.array(params['stored_coordinates'], dtype=np.float32)
                    print('External Corners:', external_corners)

                    # Determine chessboard orientation
                    rows, cols = determine_chessboard_orientation(external_corners)
                    if rows and cols:
                        interpolated_coordinates = interpolate_internal_corners(external_corners, rows, cols)
                        print('Interpolated coordinates:', interpolated_coordinates)

                        interpolated_coordinates2 = cv.cornerSubPix(gray, interpolated_coordinates, (80, 80), (-1, -1), criteria)
                        print('Subpixel Corners:', interpolated_coordinates2)

                        cv.drawChessboardCorners(params['img'], (cols, rows), interpolated_coordinates, True)
                        cv.imshow('img', params['img'])
                        cv.waitKey(500)
                    else:
                        print('Unable to determine chessboard orientation.')

        cv.setMouseCallback('img', click_event, {'stored_coordinates': stored_coordinates, 'img': img})

        cv.imshow('img', img)
        cv.waitKey(0)

print('Manually selected coordinates:')
print(stored_coordinates)
cv.destroyAllWindows()
