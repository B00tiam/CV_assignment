import glm
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

block_size = 1.0

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def load_foreground_masks(camera_dirs):
    masks = []
    for cam_dir in camera_dirs:
        mask_path = f"{cam_dir}/background_model.jpg"  # Adjust path as necessary
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks

def project_to_2d(voxel, camera_matrix, dist_coeffs, rvec, tvec):
    # Convert voxel to numpy array with shape (1, 1, 3).
    voxel_np = np.array(voxel, dtype=np.float).reshape(-1, 3)
    # Project 3D points to 2D.
    projected_points, _ = cv2.projectPoints(voxel_np, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points[0][0]  # Return the first 2D point as (x, y).

def check_voxel_in_foreground(voxel, camera_matrices, masks):
    for idx, (camera_matrix, dist_coeffs, rvec, tvec) in enumerate(camera_matrices):
        # Project voxel to each camera's 2D plane.
        x2d, y2d = project_to_2d(voxel, camera_matrix, dist_coeffs, rvec, tvec)
        # Check if the projection is within the image bounds and foreground mask.
        if 0 <= x2d < masks[idx].shape[1] and 0 <= y2d < masks[idx].shape[0]:
            if masks[idx][int(y2d), int(x2d)] > 0:  # Foreground pixel
                return True
    return False
def set_voxel_positions(width, height, depth, camera_dirs, camera_matrices):
    masks = load_foreground_masks(camera_dirs)
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel = [x*block_size - width/2, y*block_size, z*block_size - depth/2]
                if check_voxel_in_foreground(voxel, camera_matrices, masks):
                    data.append(voxel)
                    colors.append([x / width, z / depth, y / height])  # Example coloring
    return data, colors


def get_cam_positions(cam_dirs):
    positions = []
    orientations = []

    for cam_dir in cam_dirs:
        file_path = os.path.join(cam_dir, '')  # Construct the path to the extrinsics file
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Assuming the XML structure has camera elements with position and orientation children
        for camera in root.findall('camera'):
            position = camera.find('position').text.split(',')
            orientation = camera.find('orientation').text.split(',')

            # Convert string to float and append to the lists
            positions.append([float(x) for x in position])
            orientations.append([float(x) for x in orientation])

    return positions, orientations

# Define the camera directories
camera_dirs = [
    r'C:\Users\luiho\PycharmProjects\CV_assignment\Assignment_2\data\cam1',
    r'C:\Users\luiho\PycharmProjects\CV_assignment\Assignment_2\data\cam2',
    r'C:\Users\luiho\PycharmProjects\CV_assignment\Assignment_2\data\cam3',
    r'C:\Users\luiho\PycharmProjects\CV_assignment\Assignment_2\data\cam4'
]

positions, orientations = get_cam_positions(camera_dirs)


def get_cam_rotation_matrices(orientations):
    cam_rotations = []
    for orientation in orientations:
        # Assuming orientation is given as Euler angles [roll, pitch, yaw]
        rotation_matrix = glm.mat4(1)  # Identity matrix
        roll, pitch, yaw = orientation
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(roll), glm.vec3(1, 0, 0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(pitch), glm.vec3(0, 1, 0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(yaw), glm.vec3(0, 0, 1))
        cam_rotations.append(rotation_matrix)
    return cam_rotations

# Define camera directories

positions, orientations = get_cam_positions(camera_dirs)
cam_rotations = get_cam_rotation_matrices(orientations)

def visualize_voxels(voxel_positions, colors=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [v[0] for v in voxel_positions]
    ys = [v[1] for v in voxel_positions]
    zs = [v[2] for v in voxel_positions]

    if colors is None:
        ax.scatter(xs, ys, zs, c='b', marker='s')
    else:
        ax.scatter(xs, ys, zs, c=colors, marker='s')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# Example usage with your set_voxel_positions function
width, height, depth = 10, 10, 10  # Adjust these values based on your voxel grid size
camera_matrices = [...]  # This should be a list of tuples: (camera_matrix, dist_coeffs, rvec, tvec) for each camera -
#do you have any idea how to make it work?
voxel_positions, colors = set_voxel_positions(width, height, depth, camera_dirs, camera_matrices)

visualize_voxels(voxel_positions, colors)