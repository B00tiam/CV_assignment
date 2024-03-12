import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cluster import set_voxel_colors
from assignment import set_multi_voxel_positions

num_clusters = 4
cluster_trajectories = {i: [] for i in range(num_clusters)}
lines = []
fig, ax = plt.subplots()

ax.set_xlim(0, 240)
ax.set_ylim(0, 240)
ax.set_title('Trajectories')

for i in range(num_clusters):
    ln, = plt.plot([], [], 'o-', label=f'Cluster {i+1}')
    lines.append(ln)
plt.legend()

def init():
    for line in lines:
        line.set_data([], [])
        return lines

def update(frame_cnt):

    voxel_positions, voxel_colors = set_multi_voxel_positions(240, 80, 240, frame_cnt, frame_cnt)

    _, centers = set_voxel_colors(voxel_positions, voxel_colors)

    for idx, center in enumerate(centers):
        cluster_trajectories[idx].append(center)
        x_data = [point[0] for point in cluster_trajectories[idx]]
        y_data = [point[1] for point in cluster_trajectories[idx]]
        lines[idx].set_data(x_data, y_data)

    return lines

ani = FuncAnimation(fig, update, frames=range(0, 648, 12), init_func=init, blit=True)

plt.show()
