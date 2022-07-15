import umap
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

tracking_data_file = 'nuclei_tracking_with_time.csv'

df = pd.read_csv(tracking_data_file)
segment_intensity_times = df[[
    'SegmentationID',
    'ImageNumber',
    'Intensity_MeanIntensityEdge_CorrResizedRedFUCCI',
    'Intensity_MeanIntensityEdge_CorrResizedGreenFUCCI'
]]

reducer = umap.UMAP()
scaler = StandardScaler()
embedded_data = reducer.fit_transform(scaler.fit_transform(segment_intensity_times[[
    'Intensity_MeanIntensityEdge_CorrResizedRedFUCCI',
    'Intensity_MeanIntensityEdge_CorrResizedGreenFUCCI'
]]))

# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
# plt.show()

gradients_T = [[], [], [], []]

# num_traj = 100
# traj_plt = 0
for cell in segment_intensity_times['SegmentationID'].unique():
    cell_data = segment_intensity_times[segment_intensity_times['SegmentationID'] == cell]
    embedded_trajectory = reducer.transform(scaler.transform(cell_data[[
        'Intensity_MeanIntensityEdge_CorrResizedRedFUCCI',
        'Intensity_MeanIntensityEdge_CorrResizedGreenFUCCI'
    ]]))
    # plt.plot(embedded_trajectory[:, 0], embedded_trajectory[:, 1])
    # traj_plt += 1
    # if traj_plt > num_traj:
        # break
# plt.show()
# exit()
    # time = cell_data['ImageNumber'].to_list()
    # green = cell_data['Intensity_MeanIntensityEdge_CorrResizedRedFUCCI'].to_list()
    # red = cell_data['Intensity_MeanIntensityEdge_CorrResizedGreenFUCCI'].to_list()

    time = cell_data['ImageNumber'].to_list()
    green = embedded_trajectory[:, 0]
    red = embedded_trajectory[:, 1]

    data_lists = np.asarray([[t, g, r] for t, g, r in sorted(zip(time, green, red))]).transpose()
    time, green, red = data_lists

    x = []
    y = []

    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        dg_dt = (green[i + 1] - green[i]) / dt
        dr_dt  = (red[i + 1] - red[i]) / dt

        """
        for delta_t in range(int(dt)):
            x.append(green[i] + dg_dt * delta_t)
            y.append(red[i] + dr_dt * delta_t)
        """

        gradients_T[0].append(green[i])
        gradients_T[1].append(red[i])
        gradients_T[2].append(dg_dt)
        gradients_T[3].append(dr_dt)

    x.append(green[-1])
    y.append(red[-1])

    # if traj_plt < num_traj:
        # plt.plot(x, y)
        # traj_plt += 1
    # else:
        # break

# plt.scatter(gradients_T[0], gradients_T[1])


pts_per_axis = 30
x_min, x_max = min(gradients_T[0]), max(gradients_T[0])
y_min, y_max = min(gradients_T[1]), max(gradients_T[1])
x_step = (x_max - x_min) / (pts_per_axis)
y_step = (y_max - y_min) / (pts_per_axis)

xx, yy = np.meshgrid(
    np.linspace(x_min + x_step / 2, x_max - x_step / 2, pts_per_axis),
    np.linspace(y_min + y_step / 2, y_max - y_step / 2, pts_per_axis)
)

U = np.zeros((pts_per_axis, pts_per_axis))
V = np.zeros((pts_per_axis, pts_per_axis))
Cts = np.ones((pts_per_axis, pts_per_axis)) * 1e-10

"""
for i in range(len(gradients_T[0])):
    x_bucket = int((gradients_T[0][i] - x_min) / x_step)
    if x_bucket == -1: x_bucket += 1
    if x_bucket == pts_per_axis: x_bucket -= 1

    y_bucket = int((gradients_T[1][i] - y_min)  / y_step)
    if y_bucket == -1: y_bucket += 1
    if y_bucket == pts_per_axis: y_bucket -= 1

    U[x_bucket][y_bucket] += gradients_T[2][i]
    V[x_bucket][y_bucket] += gradients_T[3][i]
    Cts[x_bucket][y_bucket] += 1
"""
smth = 2
for i in range(pts_per_axis):
    for j in range(pts_per_axis):
        for x, y, g, r in zip(gradients_T[0], gradients_T[1], gradients_T[2],
                gradients_T[3]):
            if abs(xx[i][j] - x) < smth * x_step and abs(yy[i][j] - y) < smth * y_step:
                U[i][j] += g
                V[i][j] += r
                Cts[i][j] += 1

U /= Cts
V /= Cts

plt.title(f'Trajectory Velocity Field ({smth}x Smoothing)')
plt.xlabel("Green Mean Intensity")
plt.ylabel("Red Mean Intensity")

plt.quiver(xx, yy, U, V)
plt.show()

