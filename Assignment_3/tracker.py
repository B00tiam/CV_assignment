import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import json

from scipy.interpolate import CubicSpline

from assignment import set_multi_voxel_positions, create_lookup_table
from cluster import set_voxel_colors

def get_data():
    # Load configurations from the config.json file
    with open('config.json') as config_file:
        config = json.load(config_file)

    # Initialization from assignment.py mimicked here
    create_lookup_table(config['world_width'], config['world_height'], config['world_depth'])

    # Store all cluster centers
    all_centers = []

    curr_time = 0
    max_frames = 54  # Example limit for the simulation (max: 54)

    while curr_time < max_frames:
        # Simulate getting voxel positions and colors for current frame
        # 27*50 = 54*25 times
        voxel_positions, voxel_colors, _ = set_multi_voxel_positions(config['world_width'], config['world_height'], config['world_depth'], curr_time, frame_cnt=curr_time * 25)

        # Simulate processing voxel data to obtain cluster centers
        _, centers = set_voxel_colors(voxel_positions[3], voxel_colors[3])

        print(f"Frame: {curr_time * 25}, Centers shape: {centers.shape}, Centers: {centers}")

        # Append current centers to the list
        all_centers.append(centers)

        curr_time += 1  # Move to the next frame

    return all_centers

def write_file(all_centers):
    # Write the list into XML file
    # create root element
    root = ET.Element('list_data')

    for index in range(len(all_centers)):
        array_element = ET.SubElement(root, "people" + str(index))
        array_element.text = np.array2string(all_centers[index])

    tree = ET.ElementTree(root)
    tree.write('./data/mapdata.xml')

def read_file(path):
    # Get the all_list from XML file
    return

def draw_map(all_centers):

    fig, ax = plt.subplots()
    # ax.set_xlim(-config['world_width'], config['world_width'])
    # ax.set_ylim(-config['world_height'], config['world_height'])
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_title('Track Over Time')
    ax.grid(True)
    # Plot all cluster centers
    colors = ['red', 'green', 'blue', 'orange']

    for index, centers in enumerate(all_centers):
        # print(index)
        for i in range(4):
            ax.scatter(centers[i, 0], centers[i, 1], alpha=0.5, color=colors[i])  # Adjust alpha for visualization
            if index > 0:
                # Paint the lines
                x = [all_centers[index - 1][i, 0], all_centers[index][i, 0]]
                y = [all_centers[index - 1][i, 1], all_centers[index][i, 1]]
                x_smooth = np.linspace(x[0], x[1], 100)
                y_smooth = np.linspace(y[0], y[1], 100)
                ax.plot(x_smooth, y_smooth, '-o', color=colors[i])

    plt.show()  # Display the final plot after the loop is finished

write_file(get_data())