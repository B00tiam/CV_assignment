import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import random


def read_checkerboard(path):
    # get the CheckerBoardSquareSize of checkerboard.xml
    tree_read = ET.parse(path)
    root_read = tree_read.getroot()

    element_read = root_read.find('CheckerBoardSquareSize')
    if element_read is not None:
        # get the content of element
        element_text_read = element_read.text
        print("Get CheckerBoardSquareSize:" + element_text_read)

    return int(element_text_read)

def write_intrinsic(path, mtx, dist):
    # input: mtx & dist
    # write the extrinsic params in intrinsics.xml

    # create root element
    root_write = ET.Element("intrinsic_params")

    # create elements
    array_element = ET.SubElement(root_write, "IntrinsicMatrix")
    array_element.text = np.array2string(mtx)
    array_element = ET.SubElement(root_write, "DistortionCoefficients")
    array_element.text = np.array2string(dist)

    # create XML tree
    tree_write = ET.ElementTree(root_write)

    # create XML file
    tree_write.write(path + "\intrinsics.xml")
    print("Written in intrinsics.xml")


def get_intrinsic(video_path, xml_path, board_path):
    # Get square size
    square_size = read_checkerboard(board_path)

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    # need to multiply the real size of square
    objp[:, :2] = (square_size * np.mgrid[0:8, 0:6]).T.reshape(-1, 2)

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
        image_size = (frame.shape[1], frame.shape[0])
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
        cv2.imshow('Cam calibration', frame)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Camera calibration (randomly use 30 of frames)
    combinedpoints = list(zip(objpoints, imgpoints))
    randomnum = random.sample(combinedpoints, k=50)
    objpoints, imgpoints = zip(*randomnum)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    # Write XML file
    write_intrinsic(xml_path, mtx, dist)

    # print(objpoints)
    # print(imgpoints)
    # print(mtx)
    # print(dist)

def run_intrinsic():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, 'data')
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

    index = int(input("Please choose the cam you want to start:(1, 2, 3 or 4) "))
    # determine start which run
    if index == 1:
        cam_dir = camera_dirs[index - 1]
    elif index == 2:
        cam_dir = camera_dirs[index - 1]
    elif index == 3:
        cam_dir = camera_dirs[index - 1]
    elif index == 4:
        cam_dir = camera_dirs[index - 1]
    else:
        cam_dir = -1
        print("False!")
        exit()


    video_path = os.path.join(root_dir, cam_dir, 'intrinsics.avi')
    xml_path = os.path.join(root_dir, cam_dir)
    board_path = os.path.join(root_dir, 'checkerboard.xml')
    print(f"Obtained cam path: {video_path}")
    get_intrinsic(video_path, xml_path, board_path)
    print(f"Obtained intrinsic matrix for {cam_dir}")

run_intrinsic()