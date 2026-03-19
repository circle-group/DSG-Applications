import torch
from torch import Tensor
import numpy as np
from typing import Tuple
import potpourri3d as pp3d
from digeo import Mesh, MeshPointBatch
from digeo.optim import MeshLossFunc

EPS = 1e-6


def compute_log_map(mesh, face, b_coords, solver):
    vertices = mesh.faces[face]
    T1, T2, N = solver.get_tangent_frames()
    logmaps = np.zeros((3, mesh.vertices.shape[0], 3), dtype=np.float32)
    for k, v in enumerate(vertices):
        logmap = solver.compute_log_map(v.item(), strategy="VectorHeat")
        logmaps[k] = logmap[:, 0:1] * T1[v : v + 1] + logmap[:, 1:2] * T2[v : v + 1]

    min_lengths = np.linalg.norm(logmaps, axis=(2)).min(axis=0)
    bary_logmap = (
        b_coords[0] * logmaps[0] + b_coords[1] * logmaps[1] + b_coords[2] * logmaps[2]
    )
    log_lengths = np.linalg.norm(bary_logmap, axis=1)

    # Find opposing log maps and select closest ones
    opposite_v = log_lengths < min_lengths
    closest_v = np.argmax(b_coords)
    bary_logmap[opposite_v] = logmaps[closest_v][opposite_v]
    log_lengths = np.linalg.norm(bary_logmap, axis=1)

    # Project on tangent space while maintaining correct lengths
    n = mesh.triangle_normals[face].cpu().numpy()
    n = n[np.newaxis, :]
    bary_logmap = bary_logmap - (bary_logmap * n).sum(axis=1, keepdims=True) * n
    bary_logmap = bary_logmap / (
        np.linalg.norm(bary_logmap, axis=1, keepdims=True) + EPS
    )
    bary_logmap = log_lengths[:, np.newaxis] * bary_logmap

    return bary_logmap


def get_karcher_mean(solver, mesh, x):
    x_faces = x.faces.detach().cpu().numpy()  # (N)
    x_bary = x.get_barycentric_coords().detach().cpu().numpy()  # (N, 3)

    e1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
    e2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]
    triangle_areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=-1), dim=1)
    vertex_areas = torch.zeros(
        (mesh.vertices.shape[0],), device=mesh.device
    ).index_add_(0, mesh.faces[:, 0], triangle_areas / 3.0)
    vertex_areas = vertex_areas.index_add_(
        0, mesh.faces[:, 1], triangle_areas / 3.0
    )
    vertex_areas = vertex_areas.index_add_(
        0, mesh.faces[:, 2], triangle_areas / 3.0
    )
    vertex_areas = vertex_areas.detach().cpu().numpy()

    logmaps = np.zeros(
        (x_faces.shape[0], mesh.vertices.shape[0], 3), dtype=np.float32
    )  # (n_points, V, 3)
    for i in range(len(x_faces)):
        logmaps[i] = compute_log_map(mesh, x_faces[i], x_bary[i], solver)

    log_lengths = np.linalg.norm(logmaps, axis=(2))  # (n_points, V)
    idx = np.argmin(log_lengths, axis=0)  # (V)
    res = np.zeros((x_faces.shape[0], 3), dtype=np.float32)  # (n_points, 3)
    logmaps = logmaps * vertex_areas[np.newaxis, :, np.newaxis]
    for i in range(log_lengths.shape[0]):
        res[i] = logmaps[i, idx == i].sum(axis=0) / (
            np.sum(vertex_areas[idx == i]) + EPS
        )

    res = torch.from_numpy(res).to(mesh.device).to(mesh.dtype)
    loss = (vertex_areas * (log_lengths.min(axis=0) ** 2)).sum()
    return res, idx, loss


class LossGCVT(MeshLossFunc):
    def __init__(self, mesh: Mesh):
        super().__init__()
        self.solver = pp3d.MeshVectorHeatSolver(
            mesh.vertices.detach().cpu().numpy(),
            mesh.faces.detach().cpu().numpy(),
            t_coef=0.25,
        )
        self.regions = None

    def compute(self, mesh: Mesh, x: MeshPointBatch) -> Tuple[Tensor, Tensor]:
        grad, regions, loss = get_karcher_mean(self.solver, mesh, x)
        self.regions = regions
        return loss, grad

    def get_logs(self):
        return {"regions": self.regions}
