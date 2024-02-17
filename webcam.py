import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real-world space
imgpoints = []  # 2d points in image plane.

# Capture video from webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Overlay chessboard corners on the original frame
        frame_with_corners = frame.copy()
        cv.drawChessboardCorners(frame_with_corners, (9, 6), corners2, ret)
        alpha = 0.5  # Adjust transparency (0: fully transparent, 1: fully opaque)
        cv.addWeighted(frame_with_corners, alpha, frame, 1 - alpha, 0, frame)

    # Display the result
    cv.imshow('Webcam with Corners', frame)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv.destroyAllWindows()
