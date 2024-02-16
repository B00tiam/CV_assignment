import numpy as np
import cv2 as cv
import glob

def get_corners():
    # Specify the path to the images
    folder_path = 'C:\\Users\\luiho\\OneDrive\\Desktop\\AI\\AI sem 1 (periods 3-4)\\CompVis\\chessboards'
    image_path_pattern = f'{folder_path}\\*.jpg'

    # Create a list of image file paths
    images = glob.glob(image_path_pattern)

    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (6, 5, 0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Create the window outside the loop
    cv.namedWindow('img', cv.WINDOW_NORMAL)

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
            cv.resizeWindow('img', img.shape[1], img.shape[0])

            print(fname +"-valid")

            # Show the image
            cv.imshow('img', img)
            cv.waitKey(1000)   # Adjustable according to user's device
        else:
            print(fname +"-invalid")

            cv2.setMouseCallback('img', click_event)

            #pop up click_event:
            #pass the manually clicked corners back into cv.drawChessboardCorners

    cv.destroyAllWindows()

# Call the function
get_corners()

import cv2
import numpy as np

manual_corners = []

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        manual_corners.append((x, y))
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        manual_corners.append((x, y))
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

    # driver function


if __name__ == "__main__":
    # reading the image
    img = cv2.imread('chessboards\\board8.jpg', 1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', img.shape[1], img.shape[0])

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)


    # wait for a key to be pressed to exit
    cv2.waitKey(0)#

    lengths = [5, 8, 5, 8]
    manual_corners.append(manual_corners[0])
    for i in range(0, len(manual_corners) - 1):
        l = lengths[i]
        a = manual_corners[i]
        b = manual_corners[i + 1]
        xy0 = np.array(a)
        xy1 = np.array(b)
        interp = np.linspace(start=xy0, stop=xy1, num=l + 1)
        print(interp)
        print('')

        for xy in interp:
            x, y = xy
            x = int(x.item()); y = int(y.item())
            print(x, y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow('image', img)
            cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
