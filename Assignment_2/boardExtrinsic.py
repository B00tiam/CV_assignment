import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

# from boardIntrinsic import get_intrinsic
'''
# Calculate the extrinsic matrix
def get_extrinsic():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, 'data')
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

    for cam_dir in camera_dirs:
        video_path1 = os.path.join(root_dir, cam_dir, 'intrinsics.avi')
        print(video_path1)
        get_intrinsic(video_path1)
        print(f"Obtained intrinsic matrix for {cam_dir}")
    # Get matrix R & T
'''
def read_intrinsics(path):
    # get the CheckerBoardSquareSize of checkerboard.xml
    tree_read = ET.parse(path)
    root_read = tree_read.getroot()

    element_read = root_read.find('CheckerBoardSquareSize')
    if element_read is not None:
        # get the content of element
        element_text_read = element_read.text
        print("Get CheckerBoardSquareSize:" + element_text_read)

    return int(element_text_read)

def write_extrinsic():
    # input: R & T
    # write the extrinsic params in extrinsics.xml
    # 创建示例NumPy数组
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # 创建根元素
    root_write = ET.Element("extrinsic_params")

    # 创建数组元素
    array_element = ET.SubElement(root_write, "RotationMatrix")
    array_element.text = np.array2string(arr)
    array_element = ET.SubElement(root_write, "TranslationVector")
    array_element.text = np.array2string(arr)

    # 创建XML树
    tree = ET.ElementTree(root_write)

    # 将XML树写入文件
    tree.write("data/extrinsics.xml")


path_checkerboard = './data/checkerboard.xml'

write_extrinsic()
