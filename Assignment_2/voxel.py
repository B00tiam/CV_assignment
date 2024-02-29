import glm
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET

block_size = 2.0
camera_ids = [1, 2, 3, 4]

def load_mask(camera_id):
    # Adjust the path according to your project structure and file format
    mask_path = f'C:/Users/luiho/PycharmProjects/CV_assignment/Assignment_2/data/cam{camera_id}/foreground_mask.jpg'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask for camera {camera_id} not found at {mask_path}")
    return mask

def load_intrinsics(camera_id):
    # Construct the path dynamically based on camera_id
    path = f'./data/cam{camera_id}/intrinsics.xml'
    tree = ET.parse(path)
    root = tree.getroot()

    # Assuming the XML structure is the same for all cameras and doesn't include the 'cam{camera_id}' part
    camera_matrix = root.find('IntrinsicMatrix').text
    camera_matrix = camera_matrix.replace('\n', '').replace('[', '').replace(']', '').split()
    camera_matrix = np.array(list(map(float, camera_matrix))).reshape(3, 3)

    dist_coeffs = root.find('DistortionCoefficients').text
    dist_coeffs = dist_coeffs.replace('\n', '').replace('[', '').replace(']', '').split()
    dist_coeffs = np.array(list(map(float, dist_coeffs)))

    return camera_matrix, dist_coeffs

# Function to load extrinsic parameters for a specific camera
def load_extrinsics(camera_id):
    # Construct the path dynamically based on camera_id
    path = f'./data/cam{camera_id}/extrinsics.xml'
    tree = ET.parse(path)
    root = tree.getroot()

    # Assuming the XML structure is the same for all cameras and doesn't include the 'cam{camera_id}' part
    rotation_matrix = root.find('RotationMatrix').text
    rotation_matrix = rotation_matrix.replace('\n', '').replace('[', '').replace(']', '').split()
    rotation_matrix = np.array(list(map(float, rotation_matrix))).reshape(3, 3)

    translation_vector = root.find('TranslationVector').text
    translation_vector = translation_vector.replace('\n', '').replace('[', '').replace(']', '').split()
    translation_vector = np.array(list(map(float, translation_vector)))

    return rotation_matrix, translation_vector

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def is_voxel_in_foreground(voxel, camera_matrix, dist_coeffs, rotation_matrix, translation_vector, mask):
    # Project voxel to 2D point
    voxel_np = np.array([voxel], dtype=np.float32).reshape(-1, 1, 3)
    projected_point, _ = cv2.projectPoints(voxel_np, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)
    x, y = int(projected_point[0][0][0]), int(projected_point[0][0][1])

    # Check if the projected point is within the mask's bounds and is part of the foreground
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
        cv2.circle(mask, (x, y), 2, (125, 125, 125), -1)
        if mask[y, x] == 255:
            return True
        else:
            return False

def set_voxel_positions(width, height, depth):
    lookup_table = {}

    # Fill in the lookup table based on the foreground masks

    for x in range(0, 60, int(block_size)):
        for y in range(0, 60, int(block_size)):
            for z in range(0, 60, int(block_size)):
                for cam_id in camera_ids:
                    camera_matrix, dist_coeffs = load_intrinsics(cam_id)
                    rotation_matrix, translation_vector = load_extrinsics(cam_id)
                    mask = load_mask(cam_id)  # Get the mask for the current camera
                    voxel = [x - width / 2, y, z - depth / 2]
                    lookup_table[cam_id] = {}
                    lookup_table[cam_id][f'{x}_{y}_{z}'] = {}
                    lookup_table[cam_id][f'{x}_{y}_{z}']['visible'] = is_voxel_in_foreground(voxel, camera_matrix, dist_coeffs, rotation_matrix, translation_vector, mask)
                    lookup_table[cam_id][f'{x}_{y}_{z}']['mask'] = mask

    # print()
    # cv2.imshow('',lookup_table[1][f'{0}_{0}_{0}']['mask'])
    # cv2.waitKey(0)
    data, colors = [], []
    # Now only add voxels to data and colors if they are marked as foreground in the lookup table
    # for cam_id in lookup_table:
    #     voxel_visible = True
    #     for voxel in lookup_table[voxel]:
    #         if lookup_table[cam_id][voxel]['visible'] == False or lookup_table[]:
    #             voxel_visible = False
    #         break
    #     if voxel_visible:
    #         data.append(voxel)
    #         colors.append([x / width, y / height, z / depth])
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam_position = []
    cam_direction = []
    for i in range(4):
        R, T = load_extrinsics(i + 1)
        p = -np.dot(np.linalg.inv(R), T)
        d = R[:, 2]
        print(d)
        cam_position.append(p.tolist())
        cam_direction.append(d.tolist())

    return cam_position, cam_direction
    '''
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    '''

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # New code
    cam_angles = []
    cam_rotations = []
    for i in range(4):
        R, T = load_extrinsics(i + 1)
        rotations = np.eye(4)
        rotations[:3, :3] = R
        rotations[:3, 3] = T.flatten()
        cam_rotations.append(rotations.tolist())

        angles = np.degrees(np.arctan2(R[2, 1], R[2, 2])), \
            np.degrees(np.arcsin(-R[2, 0])), \
            np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        # print(list(angles))
        cam_angles.append(list(angles))

    print(cam_rotations)
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    print(cam_rotations)
    return cam_rotations

# test
# get_cam_positions()
get_cam_rotation_matrices()
