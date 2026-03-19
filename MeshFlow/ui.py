import numpy as np
import plotly.graph_objects as go
from digeo import Mesh, MeshPointBatch
from torch import Tensor


def visualize_mesh_and_points(
    mesh: Mesh,
    meshpoints: MeshPointBatch | Tensor,
    sigma: float = 0.1,
    desc: str = "Points",
    show_points=True,
) -> None:
    # Interpolate meshpoints to get 3D coordinates
    if isinstance(meshpoints, MeshPointBatch):
        points = meshpoints.interpolate(mesh).detach().cpu().numpy()
    elif isinstance(meshpoints, Tensor):
        points = meshpoints.detach().cpu().numpy()
    else:
        raise TypeError("meshpoints must be a MeshPointBatch or Tensor")

    # Extract mesh data
    vertices = mesh.vertices.detach().cpu().numpy()
    triangles = np.array(
        [[tri[0].item(), tri[1].item(), tri[2].item()] for tri in mesh.faces]
    )

    # Compute squared distances from each vertex to each point
    diff = (
        vertices[:, np.newaxis, :] - points[np.newaxis, :, :]
    )  # (n_vertices, n_points, 3)
    dist_sq = np.sum(diff**2, axis=2)  # (n_vertices, n_points)

    # Apply Gaussian kernel
    weights = np.exp(-dist_sq / (2 * sigma**2))  # (n_vertices, n_points)
    vertex_densities = np.sum(weights, axis=1)  # (n_vertices,)

    # Normalize to [0, 1]
    vertex_densities /= np.max(vertex_densities)

    # Create heatmap mesh
    heatmap_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=vertex_densities,
        colorscale="Viridis",
        showscale=True,
        opacity=1.0,
        name="Gaussian Heatmap",
    )

    # Also plot the original points
    points_scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=1, color="red"),
        name=desc,
    )

    data = [heatmap_mesh]
    if show_points:
        data.append(points_scatter)

    # Final figure
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        title=f"Mesh Heatmap with Gaussian-Smoothed {desc}",
    )
    fig.show()


def visualize_vecfield(mesh: Mesh, meshpoints, vecs) -> None:
    if isinstance(meshpoints, MeshPointBatch):
        points = meshpoints.interpolate(mesh).detach().cpu().numpy()
    elif isinstance(meshpoints, Tensor):
        points = meshpoints.detach().cpu().numpy()
    else:
        raise TypeError("meshpoints must be a MeshPointBatch or Tensor")

    # Convert vecs to numpy if it's a tensor
    if isinstance(vecs, Tensor):
        vecs = vecs.detach().cpu().numpy()

    vertices = mesh.vertices.detach().cpu().numpy()
    triangles = mesh.faces.detach().cpu().numpy()

    # Create mesh
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="lightblue",
        opacity=0.5,
        name="Mesh",
    )

    vecs = go.Cone(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=vecs[:, 0],
        v=vecs[:, 1],
        w=vecs[:, 2],
        sizemode="scaled",
        sizeref=0.5,
        anchor="tail",
        colorscale="Viridis",
        showscale=True,
    )

    # Create a 3D quiver plot
    fig = go.Figure(data=[mesh_plot, vecs])

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        title="Vector Field Visualization",
    )
    fig.show()


def visualize_paths(mesh: Mesh, paths) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    triangles = mesh.faces.detach().cpu().numpy()

    # Create mesh
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color="lightblue",
        opacity=0.5,
        name="Mesh",
    )

    start_points = np.array([path[0].detach().cpu().numpy() for path in paths])
    start_points_scatter = go.Scatter3d(
        x=start_points[:, 0],
        y=start_points[:, 1],
        z=start_points[:, 2],
        mode="markers",
        marker=dict(size=2, color="red"),
        name="start points",
    )

    # Create paths as lines
    paths_lines = []
    for path in paths:
        if isinstance(path, Tensor):
            path = path.detach().cpu().numpy()
        paths_lines.append(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line=dict(width=2, color="black"),
                name="Path",
            )
        )

    data = [mesh_plot, start_points_scatter] + paths_lines
    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        title="Paths Visualization",
    )
    fig.show()
