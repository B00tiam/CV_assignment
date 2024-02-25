import numpy as np
import cv2
import os


def get_intrinsic(video_path):

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real-world space
    imgpoints = []  # 2D points in image plane.

    # Capture video from webcam
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Overlay chessboard corners on the original frame
            cv2.drawChessboardCorners(frame, (8, 6), corners2, ret)


        # Display the result
        cv2.imshow('Webcam with Axes, Labels, and Cube', frame)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    # Camera calibration
    # ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, 'data')
camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

for cam_dir in camera_dirs:
    video_path = os.path.join(root_dir, cam_dir, 'intrinsics.avi')
    print(video_path)
    get_intrinsic(video_path)
    print(f"Obtained intrinsic matrix for {cam_dir}")