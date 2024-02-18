import numpy as np
import cv2 as cv

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane.

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
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Overlay chessboard corners on the original frame
        frame_with_corners = frame.copy()
        cv.drawChessboardCorners(frame_with_corners, (9, 6), corners2, ret)
        alpha = 0.5  # Adjust transparency (0: fully transparent, 1: fully opaque)
        cv.addWeighted(frame_with_corners, alpha, frame, 1 - alpha, 0, frame)

        # Camera calibration
        ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Draw 3D axes with labels
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

            axis_3D = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
            imgpts_3D, _ = cv.projectPoints(axis_3D, rvecs, tvecs, mtx, dist)
            corner = tuple(np.round(corners2[0].ravel()).astype(int))

            # Draw lines representing the X, Y, and Z axes
            x_line_end = tuple(np.round(imgpts_3D[0].ravel()).astype(int))
            y_line_end = tuple(np.round(imgpts_3D[1].ravel()).astype(int))
            z_line_end = tuple(np.round(imgpts_3D[2].ravel()).astype(int))

            frame = cv.line(frame, corner, x_line_end, (255, 0, 0), 2)
            frame = cv.line(frame, corner, y_line_end, (0, 255, 0), 2)
            frame = cv.line(frame, corner, z_line_end, (0, 0, 255), 2)

            # Add labels for X, Y, and Z axes
            frame = cv.putText(frame, 'X', x_line_end, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            frame = cv.putText(frame, 'Y', y_line_end, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            frame = cv.putText(frame, 'Z', z_line_end, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            # Draw the cube
            axis_cube = np.float32(
                [[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
            imgpts_cube, _ = cv.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)
            imgpts_cube = np.int32(imgpts_cube).reshape(-1, 2)
            frame = cv.drawContours(frame, [imgpts_cube[:4]], -1, (0, 255, 0), -3)
            for i, j in zip(range(4), range(4, 8)):
                frame = cv.line(frame, tuple(imgpts_cube[i]), tuple(imgpts_cube[j]), (255), 2)
            frame = cv.drawContours(frame, [imgpts_cube[4:]], -1, (0, 0, 255), 2)

    # Display the result
    cv.imshow('Webcam', frame)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv.destroyAllWindows()
