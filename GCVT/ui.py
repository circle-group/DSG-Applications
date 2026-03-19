import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from digeo import Mesh, MeshPointBatch
from matplotlib import cm


def visualize_points(mesh: Mesh, meshpoints: MeshPointBatch) -> None:
    # Interpolate seeds
    seeds = meshpoints.interpolate(mesh).detach().cpu().numpy()

    # Extract mesh data
    vertices = mesh.vertices.detach().cpu().numpy()            # [V, 3]
    triangles = np.array([[tri[0].item(), tri[1].item(), tri[2].item()] for tri in mesh.faces])  # [F, 3]

    # Plotly expects colors per vertex per face (one color per vertex)
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=0.6,
        showscale=False
    )

    # Plot seed points
    seeds_scatter = go.Scatter3d(
        x=seeds[:, 0],
        y=seeds[:, 1],
        z=seeds[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Seeds'
    )

    # Combine and plot
    fig = go.Figure(data=[mesh_plot, seeds_scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Mesh with points",
    )
    fig.show()

def visualize_voronoi(mesh: Mesh, seeds_meshpoints: MeshPointBatch, vertex_regions: NDArray) -> None:
    # Convert seeds to numpy
    seeds = seeds_meshpoints.interpolate(mesh).detach().cpu().numpy()

    # Extract mesh data
    vertices = mesh.vertices.detach().cpu().numpy()            # [V, 3]
    triangles = np.array([[tri[0].item(), tri[1].item(), tri[2].item()] for tri in mesh.faces])  # [F, 3]

    # Create a color map for the regions
    region_ids = np.unique(vertex_regions)
    num_regions = len(region_ids)
    colormap = cm.get_cmap('tab20', num_regions)  # tab20 or Set3 has many distinct colors
    region_to_color = {region: colormap(i)[:3] for i, region in enumerate(region_ids)}  # RGB
    vertex_colors = np.array([region_to_color[region] for region in vertex_regions])  # [V, 3]

    # Plot mesh with vertex colors
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        vertexcolor=vertex_colors,
        opacity=1.0,
        flatshading=True,
        name='Voronoi Regions',
        showscale=False
    )

    # Plot seed points
    seeds_scatter = go.Scatter3d(
        x=seeds[:, 0],
        y=seeds[:, 1],
        z=seeds[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Seeds'
    )

    # Combine and plot
    fig = go.Figure(data=[mesh_plot, seeds_scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Voronoi Regions with Discrete Colors",
    )
    fig.show()

