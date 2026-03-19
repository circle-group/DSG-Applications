import numpy as np
import trimesh
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import open3d as o3d


SAMPLING_RATIO = 0.25


def subsample_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    target_vertex_count = len(mesh.vertices) // 4
    simplified_mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_vertex_count * 2
    )
    mesh_trimesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_mesh.vertices),
        faces=np.asarray(simplified_mesh.triangles),
    )
    return mesh_trimesh


def get_sig17_paths() -> Tuple[List[str], List[str]]:
    train_dir = Path("data/sig17_seg_benchmark/meshes/train")
    val_dir = Path("data/sig17_seg_benchmark/meshes/test")
    train_paths = [str(p) for p in train_dir.rglob("*.off")]
    val_paths = [str(p) for p in val_dir.rglob("*.off")]
    return sorted(train_paths), sorted(val_paths)


def rescale_meshes(mesh_paths):
    for path in tqdm(mesh_paths, desc="Rescaling meshes"):
        mesh = trimesh.load_mesh(path)
        mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.export(path)


def subsample_path(mesh_paths) -> None:
    for path in tqdm(mesh_paths, desc="Subsampling meshes"):
        subsampled_mesh_path = path.replace(".off", "_subsampled.obj")
        subsampled_mesh = subsample_mesh(path)
        subsampled_mesh.export(subsampled_mesh_path)


def main():
    train_mesh_paths, val_mesh_paths = get_sig17_paths()
    mesh_paths = train_mesh_paths + val_mesh_paths
    rescale_meshes(mesh_paths)
    subsample_path(mesh_paths)


if __name__ == "__main__":
    main()
