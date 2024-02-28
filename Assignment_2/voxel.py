import cv2
import numpy as np
import glm
import random
import xml.etree.ElementTree as ET

block_size = 1.0

# Load camera configurations from XML
# Function to load intrinsic parameters from intrinsics.xml
import numpy as np
import xml.etree.ElementTree as ET

# Function to load intrinsic parameters for a specific camera
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

# Generate grid positions (no change needed as per your request)
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

# Adjusted voxel generation to match specific needs
def set_voxel_positions(width, height, depth):
    data, colors = [], []
    # Adjust this logic as per your requirement or input video analysis
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # Example condition, adjust as necessary
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, y / height, z / depth])
    return data, colors

camera_image_width = 1920
camera_image_height = 1080
# Function to project voxels onto camera views and filter based on FoV
def project_and_filter_voxels(voxels, camera_id):
    camera_matrix, dist_coeffs = load_intrinsics(camera_id)
    rotation_vector, translation_vector = load_extrinsics(camera_id)
    filtered_voxels = []
    for voxel in voxels:
        voxel_np = np.array([voxel], dtype = np.float32).reshape(-1, 1, 3)
        projected_point, _ = cv2.projectPoints(voxel_np, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        # Example filter condition, you'll need to adjust this based on actual camera image dimensions and FoV
        if 0 <= projected_point[0][0][0] < camera_image_width and 0 <= projected_point[0][0][1] < camera_image_height:
            filtered_voxels.append(voxel)
    return filtered_voxels

# Main function to orchestrate voxel reconstruction
def voxel_reconstruction(width, height, depth):
    voxels = set_voxel_positions(width, height, depth)[0]
    # Assuming you have 4 cameras based on the original code
    for camera_id in range(1, 5):
        voxels = project_and_filter_voxels(voxels, camera_id)
    # Implement further processing, such as noise reduction and hole filling here
    return voxels

# Example usage
# voxels = voxel_reconstruction(100, 100, 100)
load_extrinsics(1)
load_intrinsics(1)
