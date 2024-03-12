import matplotlib.pyplot as plt
from assignment import set_voxel_positions, set_multi_voxel_positions
from cluster import set_voxel_colors
import json
import numpy as np

# Load configurations from the config.json file
with open('config.json') as config_file:
    config = json.load(config_file)

# Set up the plot for real-time visualization
fig, ax = plt.subplots()
ax.set_xlim(-config['world_width'], config['world_width'])
ax.set_ylim(-config['world_height'], config['world_height'])
ax.set_title('Cluster Centers Over Time')
ax.grid(True)

# Prepare scatter plot objects for each cluster color
scatter_plots = {
    'red': ax.scatter([], [], c='red', s=10),
    'green': ax.scatter([], [], c='green', s=10),
    'blue': ax.scatter([], [], c='blue', s=10),
    'yellow': ax.scatter([], [], c='yellow', s=10)
}

# The colors for each cluster center
cluster_colors = ['red', 'green', 'blue', 'yellow']

# Dummy data - Replace this with actual voxel positions and colors as per your data retrieval method
voxel_positions = np.zeros((config['world_width'], config['world_height'], config['world_depth']))
voxel_colors = np.zeros_like(voxel_positions)

plt.ion()  # Turn on interactive mode

curr_time = 0
while True:  # Replace with a suitable condition to stop
    # Update voxel data
    if curr_time % 24 == 0:
        voxel_positions, voxel_colors = set_multi_voxel_positions(
            config['world_width'], config['world_height'], config['world_depth'], curr_time, frame_cnt=curr_time
        )
    else:
        voxel_positions, voxel_colors, _ = set_voxel_positions(
            config['world_width'], config['world_height'], config['world_depth'], curr_time
        )

    # Update colors and get the centers from the clustering function
    _, centers = set_voxel_colors(voxel_positions, voxel_colors)

    # Since `centers` might be more than the number of colors, we use modulo to cycle through colors
    for i, center in enumerate(centers):
        color = cluster_colors[i % len(cluster_colors)]
        scatter_plots[color].set_offsets(np.vstack((scatter_plots[color].get_offsets(), [center[:2]])))  # Plot only x, y

    print(f"Current time: {curr_time}, centers: {centers}")

    # Update the display
    plt.pause(0.001)  # Pause briefly to allow the plot to update

    # Increment time
    curr_time += 1

    # Add a break condition here if necessary
    # ...

plt.ioff()  # Turn off interactive mode when finished plotting
plt.show()  # Display the final plot
