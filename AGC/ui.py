import numpy as np
import plotly.graph_objects as go
from digeo import Mesh
from typing import List
from torch import Tensor
from matplotlib import cm


def visualize_mesh(mesh: Mesh):
    """
    Visualizes a mesh using Plotly.

    Args:
        mesh (Mesh): The mesh to visualize.
    """
    vertices = mesh.vertices.detach().cpu().numpy()

    # Create list of triangles (faces)
    i, j, k = [], [], []
    for tri in mesh.faces:
        i.append(tri[0].item())
        j.append(tri[1].item())
        k.append(tri[2].item())

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                color="lightblue",
                opacity=0.5,
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # makes the aspect ratio equal
        ),
        title="Mesh Visualization",
    )

    fig.show()


def visualize_mesh_vectors(mesh: Mesh, starts: Tensor, vectors: Tensor):
    """
    Visualizes the normals of a mesh.

    Args:
        mesh (Mesh): The mesh whose normals are to be visualized.
    """
    start_arr = starts.detach().cpu().numpy()
    vectors_arr = vectors.detach().cpu().numpy()
    vertices = mesh.vertices.detach().cpu().numpy()

    # Create list of triangles (faces)
    i, j, k = [], [], []
    for tri in mesh.faces:
        i.append(tri[0].item())
        j.append(tri[1].item())
        k.append(tri[2].item())

    # Create a 3D mesh plot
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color="lightblue",
        opacity=0.5,
        name="Mesh",
    )

    ends = start_arr + vectors_arr * 0.1  # [N, 3]

    # Interleave start and end points with None to separate lines
    x = np.empty(start_arr.shape[0] * 3, dtype=float)
    y = np.empty_like(x)
    z = np.empty_like(x)

    x[0::3] = start_arr[:, 0]
    x[1::3] = ends[:, 0]
    x[2::3] = np.nan  # separator

    y[0::3] = start_arr[:, 1]
    y[1::3] = ends[:, 1]
    y[2::3] = np.nan

    z[0::3] = start_arr[:, 2]
    z[1::3] = ends[:, 2]
    z[2::3] = np.nan

    # One single trace for all lines
    normal_lines = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="green", width=2),
    )

    fig = go.Figure(data=[mesh_plot, normal_lines])
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # makes the aspect ratio equal
        ),
        title="Mesh Normals",
    )

    fig.show()


def visualize_agcn(
    mesh: Mesh, patches: List[Tensor], start_dirs: List[Tensor], end_dirs: List[Tensor]
) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()

    # Create list of triangles (faces)
    i, j, k = [], [], []
    for tri in mesh.faces:
        i.append(tri[0].item())
        j.append(tri[1].item())
        k.append(tri[2].item())

    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color="lightblue",
        opacity=0.5,
    )

    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "pink",
        "brown",
    ]
    patches_data = []
    k = 0
    for i in range(0, len(patches)):
        points = patches[i].detach().cpu().numpy()
        point_scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors[k % len(colors)]),
            name="Patch Points",
        )
        patches_data.append(point_scatter)
        k += 1

    dirs_data = []
    k = 0  # counter for colors

    for i in range(len(start_dirs)):
        start_points = start_dirs[i].detach().cpu().numpy()
        end_points = end_dirs[i].detach().cpu().numpy()

        # Create lines between start and end points
        xs, ys, zs = [], [], []
        for s, e in zip(start_points, end_points):
            xs.extend([s[0], e[0], None])
            ys.extend([s[1], e[1], None])
            zs.extend([s[2], e[2], None])

        dir_scatter = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=2, color=colors[k % len(colors)]),
            name="Direction Vectors",
        )
        dirs_data.append(dir_scatter)
        k += 1

    fig = go.Figure(data=[mesh_plot] + patches_data + dirs_data)

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # makes the aspect ratio equal
        ),
        title="Mesh Visualization",
    )

    fig.show()


def visualize_mesh_segmentation(mesh: Mesh, logits: Tensor) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()

    # Create list of triangles (faces)
    i, j, k = [], [], []
    for tri in mesh.faces:
        i.append(tri[0].item())
        j.append(tri[1].item())
        k.append(tri[2].item())

    labels_arr = logits.argmax(dim=-1).detach().cpu().numpy()
    colormap = cm.get_cmap(
        "tab20", labels_arr.max() + 1
    )  # tab20 or Set3 has many distinct colors
    face_colors = np.array([colormap(i)[:3] for i in labels_arr])

    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        facecolor=face_colors,
        opacity=1.0,
        flatshading=True,
        name="Mesh with Segmentation",
        showscale=False,
    )

    fig = go.Figure(data=[mesh_plot])
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # makes the aspect ratio equal
        ),
        title="Mesh Segmentation Visualization",
    )

    fig.show()
