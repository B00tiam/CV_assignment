import glm
import random
import numpy as np
import xml.etree.ElementTree as ET

block_size = 1.0

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
    # print(data)
    # print(colors)
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
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
        # print(p)
        # switch the val of y & z
        temp = p[1]
        p[1] = p[2]
        p[2] = temp
        p[1] = -1 * p[1]
        # print(p)
        # print(d)
        cam_position.append((p / 50).tolist())
        # cam_direction.append(d.tolist())

    cam_color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    return cam_position, cam_color
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
        d = []
        R, T = load_extrinsics(i + 1)
        rotations = np.eye(4)
        rotations[:3, :3] = R
        rotations[:3, 3] = T.flatten()
        cam_rotations.append(rotations.tolist())

        # get Rrot
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
        r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
        theta_x = np.degrees(np.arctan2(r32, r33))
        theta_y = np.degrees(np.arctan2(-r31, np.sqrt(r11 ** 2 + r21 ** 2)))
        theta_z = np.degrees(np.arctan2(r21, r11))
        theta_x = (theta_x + 180) % 360 - 180
        theta_y = (theta_y + 180) % 360 - 180
        theta_z = (theta_z + 180) % 360 - 180
        d.append(theta_x)
        d.append(theta_y)
        d.append(theta_z)
        cam_angles.append(d)


    # print(cam_rotations)
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    #print(cam_rotations)
    return cam_rotations

# test
get_cam_positions()
get_cam_rotation_matrices()