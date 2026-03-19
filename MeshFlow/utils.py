import torch
import numpy as np
import os
import scipy
from typing import Tuple
import scipy.sparse.linalg
import igl
import math
from torch import Tensor
from torch.func import vmap, jacrev
from digeo import Mesh, MeshPointBatch
from digeo.ops import trace_geodesics


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tri_bary_coords(p0: Tensor, p1: Tensor, p2: Tensor, p: Tensor) -> Tensor:
    """Compute barycentric coordinates of p in the triangle (p0, p1, p2)."""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return torch.stack([u, v, w], dim=-1)


def convert_to_meshpointbatch(mesh: Mesh, points) -> MeshPointBatch:
    """
    Convert a numpy array of points to a MeshPointBatch object.

    Args:
        mesh (Mesh): The mesh to which the points belong.
        points (np.ndarray): An array of shape (N, 3) representing the points.

    Returns:
        MeshPointBatch: A MeshPointBatch object containing the points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be a 2D array with shape (N, 3).")

    triangle_centroids = mesh.vertices[mesh.faces].mean(dim=1)
    points_tensor = torch.as_tensor(points).to(mesh.device)

    # find the closest triangle for each point
    faces = torch.zeros(points_tensor.shape[0], dtype=torch.int32, device=mesh.device)
    for i, point in enumerate(points_tensor):
        distances = torch.norm(triangle_centroids - point, dim=1)
        faces[i] = torch.argmin(distances)

    # compute barycentric coordinates
    closest_triangles = mesh.faces[faces]
    v0 = mesh.vertices[closest_triangles[:, 0]]
    v1 = mesh.vertices[closest_triangles[:, 1]]
    v2 = mesh.vertices[closest_triangles[:, 2]]
    bary_coords = tri_bary_coords(v0, v1, v2, points_tensor)

    bary_coords = bary_coords.clamp(min=0, max=1)
    bary_coords = bary_coords / bary_coords.sum(dim=-1, keepdim=True)

    return MeshPointBatch(faces=faces, uvs=bary_coords[:, 1:])


def sample_eigen(
    mesh: Mesh, n_points: int, k: int
) -> Tuple[MeshPointBatch, torch.Tensor]:
    """
    Samples points on the mesh surface according to a distribution
    derived from the top k Laplacian eigenfunctions (largest eigenvalues).

    Args:
        mesh (Mesh): Input mesh with .vertices and .faces (PyTorch tensors).
        n_points (int): Number of points to sample.
        k (int): Number of top Laplacian eigenvectors to use (excluding the trivial one).

    Returns:
        MeshPointBatch: Sampled barycentric points on the mesh.
    """
    V = mesh.vertices.detach().cpu().numpy()  # (n_vertices, 3)
    F = mesh.faces.detach().cpu().numpy()  # (n_faces, 3)

    # Compute cotangent Laplacian and mass matrix
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)

    # Compute top k + 1 eigenvectors (get largest magnitude eigenvalues)
    evals, evecs = scipy.sparse.linalg.eigsh(
        L, M=M, k=k, sigma=1e-5
    )  # 'LA' = largest algebraic

    scalar_field = evecs[:, -10]

    # Scalar field: squared L2 norm across selected eigenvectors
    # scalar_field = np.mean(evecs, axis=1)  # shape: (n_vertices,)
    scalar_field = (scalar_field - scalar_field.min()) / (
        scalar_field.max() - scalar_field.min()
    )

    # Compute per-face average of the scalar field
    face_values = scalar_field[F].mean(axis=1)

    # Compute face areas
    v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    # Compute sampling weights and normalize
    face_weights = face_areas * face_values**2
    face_probs = face_weights / face_weights.sum()

    # Sample face indices
    face_indices = np.random.choice(len(F), size=n_points, p=face_probs)

    # Sample barycentric coordinates on each face
    u = np.random.rand(n_points)
    v = (1 - u) * np.random.rand(n_points)  # ensures u + v <= 1

    # Return mesh points as barycentric coordinates
    sampled_points = MeshPointBatch(
        faces=torch.as_tensor(face_indices, dtype=torch.int32),
        uvs=torch.as_tensor(np.stack((u, v), axis=-1), dtype=torch.float32),
    )

    return sampled_points.to(mesh.device), torch.as_tensor(
        face_probs, dtype=torch.float32, device=mesh.device
    )


@torch.no_grad()
def kde_vertex(mesh, X, sigma=0.1):
    vertex_densities_X = torch.empty_like(mesh.vertices[:, 0])
    for i in range(mesh.vertices.shape[0]):
        diff = mesh.vertices[i, :] - X
        dist_sq = torch.sum(diff**2, dim=1)
        weights = (1 / math.sqrt(torch.pi)) * torch.exp(-dist_sq / (sigma**2))
        vertex_densities_X[i] = torch.sum(weights)
    vertex_densities_X = vertex_densities_X / torch.sum(vertex_densities_X)
    return vertex_densities_X


@torch.no_grad()
def kld(mesh, X, Y, sigma=0.1):
    vertex_densities_X = kde_vertex(mesh, X, sigma)
    vertex_densities_Y = kde_vertex(mesh, Y, sigma)

    # Compute KLD
    kld_value = torch.sum(
        vertex_densities_X
        * torch.log(vertex_densities_X / (vertex_densities_Y + 1e-10))
    )  # Add small constant
    return kld_value.item()


def div_fn(u):
    J = jacrev(u)
    return lambda x, n: torch.trace(J(x, n))


def output_and_div(mesh: Mesh, model, points: MeshPointBatch):
    x = points.interpolate(mesh)
    dx = -model(points)
    n = mesh.triangle_normals[points.faces]
    div = vmap(div_fn(model.vecfield))(x, n)
    return dx, div


def mesh_logprob(mesh, x):
    e1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
    e2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]
    areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=-1), dim=-1)
    tot_area = torch.sum(areas).item()
    return torch.full_like(x.faces, -math.log(tot_area), dtype=torch.float32)


@torch.no_grad()
def compute_exact_loglikelihood(
    mesh: Mesh,
    model,
    target_meshpoints: MeshPointBatch,
    t1: float = 1.0,
    num_steps=1000,
):
    """Computes the negative log-likelihood of a batch of data."""
    x = target_meshpoints
    logdetjac = torch.zeros_like(target_meshpoints.uvs[:, 0])

    t_tensor = torch.linspace(t1, 0, num_steps + 1).to(logdetjac.device)
    for t in t_tensor:
        dx, div = output_and_div(mesh, model, x)

        x, _ = trace_geodesics(mesh, x, -dx * (t1 - t) / num_steps, gradient="none")
        logdetjac += -div * (t1 - t) / num_steps

    logp0 = mesh_logprob(mesh, x)
    logp1 = logp0 + logdetjac

    return logp1


def chamfer_distance(D):
    """
    Compute Chamfer distance from distance matrix D.
    D[i,j] = squared distance between point i in A and point j in B
    """
    # nearest-neighbor distances
    min_a_to_b = torch.min(D, dim=1)[0]
    min_b_to_a = torch.min(D, dim=0)[0]
    # averages
    return torch.mean(min_a_to_b) + torch.mean(min_b_to_a)
