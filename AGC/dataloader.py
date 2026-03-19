import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from digeo import load_mesh_from_file

from utils import heat_kernel_signature


def random_rotation_matrix(device, config):

    if config["mesh"]["rotate"]:
        # Random quaternion
        rand = torch.randn(4, device=device, dtype=torch.float32)
        rand = rand / rand.norm()  # scalar norm
        w, x, y, z = rand  # unpack

        # Convert quaternion to rotation matrix
        R = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        R = torch.eye(3, device=device, dtype=torch.float32)

    if config["mesh"]["scale"]:
        scale_var = config["mesh"]["max_scale"] - config["mesh"]["min_scale"]
        scale = (
            torch.rand(1, device=device, dtype=torch.float32) * scale_var
            + config["mesh"]["min_scale"]
        )
        R = R * scale

    return R


class MeshDataSet(Dataset):
    def __init__(self, config, mesh_paths, labels):
        self.config = config
        self.labels = labels

        self.input = []
        self.mesh_list = []
        self.subsampled_mesh_list = []

        for path in tqdm(mesh_paths, desc="Loading meshes"):
            mesh = load_mesh_from_file(path, device="cpu")
            self.mesh_list.append(mesh)

            subsampled_mesh_path = path.replace(".off", "_subsampled.obj")
            if not os.path.exists(subsampled_mesh_path):
                raise RuntimeError(
                    f"Subsampled mesh file {subsampled_mesh_path} does not exist. Run data_setup.py beforehand."
                )

            subsampled_mesh = load_mesh_from_file(subsampled_mesh_path, device="cpu")
            self.subsampled_mesh_list.append(subsampled_mesh)

            if self.config["model"]["input"] == "hks":
                hks = heat_kernel_signature(mesh, n_eig=128, n_scales=16)
                self.input.append(torch.tensor(hks, dtype=torch.float32))
            elif self.config["model"]["input"] == "xyz":
                self.input.append(None)

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        mesh = self.mesh_list[idx]
        subsampled_mesh = self.subsampled_mesh_list[idx]
        label = self.labels[idx]
        R = random_rotation_matrix(mesh.device, self.config)
        mesh.vertices = mesh.vertices @ R.T
        subsampled_mesh.vertices = subsampled_mesh.vertices @ R.T
        model_input = self.input[idx]
        return mesh, subsampled_mesh, model_input, label


def collate_no_batch(batch):
    # batch is a list of size batch_size (here always 1)
    return batch[0]


def get_dataloader(config, mesh_paths, labels, shuffle=False):
    dataset = MeshDataSet(config, mesh_paths, labels)
    return DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        collate_fn=collate_no_batch,
        shuffle=shuffle,
    )
