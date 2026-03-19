import torch
import torch.nn as nn
import torch.nn.functional as F
from digeo import Mesh, MeshPointBatch


class VecfieldModel(nn.Module):
    def __init__(self, input_dim=3) -> None:
        super(VecfieldModel, self).__init__()
        self.num_layers = 3
        self.hidden_dim = 512

        self.scale = torch.nn.Parameter(torch.tensor(1e-4))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, self.hidden_dim))
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.output_layer = nn.Linear(self.hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.scale * self.output_layer(x)

        return x


class MeshFlow(nn.Module):
    def __init__(self, mesh: Mesh):
        super(MeshFlow, self).__init__()
        self.mesh = mesh
        self.vecfield_model = VecfieldModel()

    def vecfield(self, x, normals):
        x = self.vecfield_model(x)
        x = x - normals * torch.sum(x * normals, dim=-1, keepdim=True)
        return x

    def forward(self, points: MeshPointBatch) -> torch.Tensor:
        x = points.interpolate(self.mesh)
        normals = self.mesh.triangle_normals[points.faces]
        return self.vecfield(x, normals)
