import sys
sys.path.append('ProposalNetwork/utils')
from plane import Plane, Plane_np

import open3d as o3d
import numpy as np

import torch

# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("ProposalNetwork/utils/caixa.ply")
# o3d.visualization.draw_geometries([pcd_load])
points = np.asarray(pcd_load.points)
import time


p1 = Plane()

p2 = Plane_np()

# points = np.array([[-2.11,1.38,0],[0,0,1.86],[1.44,1.27,0]])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
point_torch = torch.from_numpy(points).to(device)

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
t1 = time.perf_counter()
a1 = p1.fit(point_torch, thresh=0.01, maxIteration=1000)
t2 = time.perf_counter()
t3 = time.perf_counter()
a11 = p1.fit_parallel(point_torch, thresh=0.01, maxIteration=1000)
t4 = time.perf_counter()
t5 = time.perf_counter()
a2 = p2.fit(points, thresh=0.01, maxIteration=1000)
t6 = time.perf_counter()

print(f'time for pyransac3d: {t6-t5}')
print(f'time for torch: {t2-t1}')
print(f'time for torch parallel: {t4-t3}')

print(a1)
print(a2[0],torch.from_numpy(a2[1]))
print(a11)