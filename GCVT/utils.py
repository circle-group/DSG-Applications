import torch
import numpy as np
from digeo import MeshPointBatch
from digeo.ops import trace_geodesics


def sanitize_results(results):
    if isinstance(results, np.number):
        return float(results)
    elif isinstance(results, np.ndarray):
        return results.tolist()
    elif isinstance(results, list):
        return [sanitize_results(r) for r in results]
    elif isinstance(results, dict):
        return {k: sanitize_results(v) for k, v in results.items()}
    else:
        return results


@torch.no_grad()
def sample_cluster(mesh, source_vertex, sigma, num_samples=50):
    face = mesh.v2t[source_vertex, 1].item()
    uv = torch.zeros(2).to(mesh.device, dtype=torch.float32)
    for k in range(2):
        if mesh.faces[face, k+1].item() == source_vertex:
            uv[k] = 1.0

    start_points = MeshPointBatch(
        faces=torch.tensor([face], device=mesh.device, dtype=torch.int32).repeat(num_samples),
        uvs=uv.unsqueeze(0).repeat(num_samples, 1)
    )

    dirs = torch.randn((num_samples, 3), device=mesh.device)
    normal = mesh.vertex_normals[source_vertex].unsqueeze(0)

    dirs -= (dirs * normal).sum(dim=1, keepdim=True) * normal
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

    dirs_length = torch.normal(mean=0.0, std=sigma, size=(num_samples, 1), device=mesh.device).abs()
    dirs = dirs * dirs_length

    end_points, _ = trace_geodesics(mesh, start_points, dirs, gradient="none")

    return end_points
