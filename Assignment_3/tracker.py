import matplotlib.pyplot as plt
import numpy as np
import json
from assignment import set_multi_voxel_positions, create_lookup_table
from cluster import set_voxel_colors

# Load configurations from the config.json file
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialization from assignment.py mimicked here
create_lookup_table(config['world_width'], config['world_height'], config['world_depth'])

fig, ax = plt.subplots()
# ax.set_xlim(-config['world_width'], config['world_width'])
# ax.set_ylim(-config['world_height'], config['world_height'])
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_title('Track Over Time')
ax.grid(True)

# Store all cluster centers
all_centers = []

curr_time = 0
max_frames = 25  # Example limit for the simulation (max: 25)

while curr_time < max_frames:
    # Simulate getting voxel positions and colors for current frame
    # 54 times
    voxel_positions, voxel_colors, _ = set_multi_voxel_positions(config['world_width'], config['world_height'], config['world_depth'], curr_time, frame_cnt=curr_time * 50)

    # Simulate processing voxel data to obtain cluster centers
    _, centers = set_voxel_colors(voxel_positions[3], voxel_colors[3])

    print(f"Frame: {curr_time * 50}, Centers shape: {centers.shape}, Centers: {centers}")

    # Append current centers to the list
    all_centers.append(centers)

    curr_time += 1  # Move to the next frame

# Plot all cluster centers
colors = ['red', 'green', 'blue', 'orange']

for index, centers in enumerate(all_centers):
    # print(index)
    for i in range(4):
        ax.scatter(centers[i, 0], centers[i, 1], alpha=0.5, color=colors[i])  # Adjust alpha for visualization
        if index > 0:
            # Paint the lines
            ax.plot([all_centers[index - 1][i, 0], all_centers[index][i, 0]], [all_centers[index-1][i, 1], all_centers[index][i, 1]], '-o', color=colors[i])

plt.show()  # Display the final plot after the loop is finished
