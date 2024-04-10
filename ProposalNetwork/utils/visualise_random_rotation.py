import matplotlib.pyplot as plt
import numpy as np
import torch

from cubercnn import util

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1602779#1602779
# http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
# https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m
def qr_full(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot

def qr_full_torch(num_samples=1):
    z = torch.randn(num_samples, 3, 3)
    q, r = torch.linalg.qr(z)
    sign = 2 * (torch.diagonal(r, dim1=-2, dim2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= torch.linalg.det(rot)[..., None]
    return rot

def randn_orthobasis_torch(num_samples=1):
    z = torch.randn(num_samples, 3, 3)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    z[:, 0] = torch.cross(z[:, 1], z[:, 2], dim=-1)
    z[:, 0] = z[:, 0] / torch.norm(z[:, 0], dim=-1, keepdim=True)
    z[:, 1] = torch.cross(z[:, 2], z[:, 0], dim=-1)
    z[:, 1] = z[:, 1] / torch.norm(z[:, 1], dim=-1, keepdim=True)
    return z

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1288873#1288873
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44701#44701
def randn_orthobasis(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    z[:, 0] = np.cross(z[:, 1], z[:, 2], axis=-1)
    z[:, 0] = z[:, 0] / np.linalg.norm(z[:, 0], axis=-1, keepdims=True)
    z[:, 1] = np.cross(z[:, 2], z[:, 0], axis=-1)
    z[:, 1] = z[:, 1] / np.linalg.norm(z[:, 1], axis=-1, keepdims=True)
    return z

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4394036#4394036
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44701#44701
def randn_axis(num_samples=1, corrected=True):
    u = np.random.uniform(0, 1, size=num_samples)
    z = np.random.randn(num_samples, 1, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    if corrected:
        t = np.linspace(0, np.pi, 1024)
        cdf_psi = (t - np.sin(t)) / np.pi
        psi = np.interp(u, cdf_psi, t, left=0, right=np.pi)
    else:
        psi = 2 * np.pi * u

    return rot3x3_from_axis_angle(z, psi)

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/442423#442423
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44691#44691
def nbubis(num_samples=1, corrected=True):
    u1 = np.random.uniform(0, 1, size=num_samples)
    u2 = np.random.uniform(0, 1, size=num_samples)
    u3 = np.random.uniform(0, 1, size=num_samples)

    theta = np.arccos(2 * u1 - 1)
    phi = 2 * np.pi * u2
    axis_vector = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
    axis_vector = np.stack(axis_vector, axis=1).reshape(-1, 1, 3)

    if corrected:
        t = np.linspace(0, np.pi, 1024)
        cdf_psi = (t - np.sin(t)) / np.pi
        psi = np.interp(u3, cdf_psi, t, left=0, right=np.pi)
    else:
        psi = 2 * np.pi * u3

    return rot3x3_from_axis_angle(axis_vector, psi)

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1602779#1602779
def qr_half(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    return q

def euler_angles(num_samples=1):
    rx = np.random.rand(num_samples) * np.pi - np.pi/2
    ry = np.random.rand(num_samples) * np.pi - np.pi/2
    rz = np.random.rand(num_samples) * np.pi - np.pi/2
    # loop over all
    rotation_matrix = np.array([util.euler2mat([x,y,z]) for x, y, z in zip(rx,ry,rz)])
    return rotation_matrix

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
def rot3x3_from_axis_angle(axis_vector, angle):
    angle = np.atleast_1d(angle)[..., None, None]
    K = np.cross(np.eye(3), axis_vector)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def plot_scatter(pointses, filename, kwargses):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    for points, kwargs in zip(pointses, kwargses):
        ax.scatter(*np.asarray(points).T, marker=".", **kwargs)
    ax.view_init(elev=45, azim=-45, roll=0)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_aspect("equal", adjustable="box")
    fig.savefig('ProposalNetwork/output/random_rotation/'+filename, dpi=300, bbox_inches="tight", pad_inches=0)
    # plt.show()
    # exit()
    plt.close(fig)

METHODS = {
    "randn_orthobasis_torch": randn_orthobasis_torch,
    "qr_full_torch": qr_full_torch,
    "euler": euler_angles,
    "randn_orthobasis": randn_orthobasis,
    "randn_axis": randn_axis,
    "randn_axis_incorrect": lambda **kwargs: randn_axis(corrected=False, **kwargs),
    "nbubis": nbubis,
    "nbubis_incorrect": lambda **kwargs: nbubis(corrected=False, **kwargs),
    "qr_half": qr_half,
    "qr_full": qr_full,
}

import time
import os
os.makedirs('ProposalNetwork/output/random_rotation',exist_ok=True)
# x is the starting point; y contains various sample rotated points.
# x = np.array([1.0, 0.0, 0.0])
x = np.array([1 / 9, -4 / 9, 8 / 9], dtype=np.float32)
x /= np.linalg.norm(x)  # Normalize to unit vector, just in case.
for name, func in METHODS.items():
    t1 = time.perf_counter()
    rot = func(num_samples=5000 // (2 if "_half" in name else 1))
    t2 = time.perf_counter()
    print(f'{name}\t\t Time: {t2-t1:.4f}')
    if 'torch' in name:
        y = rot @ torch.from_numpy(x)
    else:
        y = rot @ x
    plot_scatter(
        [y, [x]],
        f"rot3x3_{name}.png",
        [{"s": 1, "alpha": 0.5}, {"s": 64, "color": "#ff77cc"}],
    )