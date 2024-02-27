import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

from processCorners import manual_process

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

def read_intrinsics(path):
    # get the params of checkerboard.xml
    tree_read = ET.parse(path)
    root_read = tree_read.getroot()

    # get two elements
    mtx = root_read.find('IntrinsicMatrix').text
    mtx = mtx.replace('\n', '').replace('[', '').replace(']', '').split()
    mtx = np.array(list(map(float, mtx))).reshape(3, 3)

    # 查找畸变系数元素
    dist = root_read.find('DistortionCoefficients').text
    dist = dist.replace('\n', '').replace('[', '').replace(']', '').split()
    dist = np.array(list(map(float, dist)))

    # print(mtx)
    # print(dist)
    return mtx, dist

def transform(old_co, R, T):
    # transform the coordinate
    new_co = np.dot(R, old_co) + T
    return new_co

def write_extrinsic(path, R, T):
    # input: R & T
    # write the extrinsic params in extrinsics.xml

    # create root element
    root_write = ET.Element("extrinsic_params")

    # create elements
    array_element = ET.SubElement(root_write, "RotationMatrix")
    array_element.text = np.array2string(R)
    array_element = ET.SubElement(root_write, "TranslationVector")
    array_element.text = np.array2string(T)

    # create XML tree
    tree_write = ET.ElementTree(root_write)

    # create XML file
    tree_write.write(path + "\extrinsics.xml")
    print("Written in extrinsics.xml")


# Calculate the extrinsic matrix
def get_extrinsic(video_path, xml_path, intrinsic_path, board_path):

    cap = cv2.VideoCapture(video_path)
    # Get square size
    square_size = read_checkerboard(board_path)
    # Get mtx & dist
    mtx, dist = read_intrinsics(intrinsic_path)
    # Capture the 1st frame of video
    ret, frame1 = cap.read()
    # image_size = (frame1.shape[1], frame1.shape[0])

    if not ret:
        print("Failed to capture frame from webcam")
        exit()
    cv2.imwrite(xml_path + '\\first_frame.jpg', frame1)
    cap.release()


    objpoints, imgpoints = manual_process(xml_path + '\\first_frame.jpg')

    # Get matrix R & T
    retval, rvec, tvec = cv2.solvePnP(objpoints[0], imgpoints[0], mtx, dist)
    T = np.squeeze(tvec)
    R, _ = cv2.Rodrigues(rvec)
    # Write R & T in xml file
    write_extrinsic(xml_path, R, T)
    # print(R)
    # print(T)

    # Paint the 3D axis
    paint(R, T, video_path, imgpoints[0], mtx)

def paint(R, T, video_path, corners, mtx):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break

        # Paint the 3D axis on video frames
        scale = 250  # control length
        thickness = 2  # control width
        line_type = cv2.LINE_AA  # type of lines

        # originpoint = np.round(corners[0].ravel()).astype(int)
        originpoint = transform(np.array([0, 0, 0]), R, T)
        s_originpoint = np.dot(mtx, originpoint)
        i_originpoint = np.round(s_originpoint[:2]/s_originpoint[2]).astype(int)

        # Paint X
        endpoint_x = transform(np.array([scale, 0, 0]), R, T)
        s_endpoint_x = np.dot(mtx, endpoint_x)
        i_endpoint_x = np.round(s_endpoint_x[:2]/s_endpoint_x[2]).astype(int)
        frame = cv2.line(frame, tuple(i_originpoint), tuple(i_endpoint_x), (0, 0, 255), thickness, line_type)

        # Paint Y
        endpoint_y = transform(np.array([0, scale, 0]), R, T)
        s_endpoint_y = np.dot(mtx, endpoint_y)
        i_endpoint_y = np.round(s_endpoint_y[:2] / s_endpoint_y[2]).astype(int)
        frame = cv2.line(frame, tuple(i_originpoint), tuple(i_endpoint_y), (0, 255, 0), thickness, line_type)

        # Paint Z
        endpoint_z = transform(np.array([0, 0, -1 * scale]), R, T)
        s_endpoint_z = np.dot(mtx, endpoint_z)
        i_endpoint_z = np.round(s_endpoint_z[:2] / s_endpoint_z[2]).astype(int)
        frame = cv2.line(frame, tuple(i_originpoint), tuple(i_endpoint_z), (255, 0, 0), thickness, line_type)

        # Show the frames
        cv2.imshow('Video with transformed coordinate system', frame)
        # cv2.waitKey(10000)
        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_extrinsic():
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

    video_path = os.path.join(root_dir, cam_dir, 'checkerboard.avi')
    xml_path = os.path.join(root_dir, cam_dir)
    intrinsic_path = os.path.join(root_dir, cam_dir, 'intrinsics.xml')
    board_path = os.path.join(root_dir, 'checkerboard.xml')
    print(f"Obtained cam path: {video_path}")
    get_extrinsic(video_path, xml_path, intrinsic_path, board_path)
    print(f"Obtained extrinsic matrix for {cam_dir}")

run_extrinsic()