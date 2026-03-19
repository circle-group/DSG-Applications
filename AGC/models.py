import torch
import torch.nn as nn
import torch.nn.functional as F
from digeo import Mesh
from digeo.nn import AGC
from typing import Dict, Optional


class MeshPoolBlock(nn.Module):
    def __init__(self):
        super(MeshPoolBlock, self).__init__()
        self.dist:Optional[torch.Tensor] = None

    def pool(self, mesh:Mesh, subsampled_mesh:Mesh, X:torch.Tensor):
        self.dist = torch.cdist(mesh.vertices, subsampled_mesh.vertices, p=2) # (N, S)
        nearest_idx = torch.argmin(self.dist, dim=0) # (S,)
        return X[nearest_idx]

    def unpool(self, X:torch.Tensor):
        if self.dist is None:
            raise RuntimeError("Pool must be called before unpool.")
        # Use the distance matrix to unpool the features
        nearest_idx = torch.argmin(self.dist, dim=1) # (N,)
        self.dist = None
        return X[nearest_idx]


class ResNetBlock(nn.Module):
    def __init__(self, in_filter:int, out_filter:int, rho_init:float=0.1, n_rho:int=1, learn_rho=True, n_patches:int=4):
        super(ResNetBlock, self).__init__()
        self.conv1 = AGC(in_filter, out_filter, rho_init=rho_init, n_patches=n_patches, n_rho=n_rho, learn_rho=learn_rho)
        self.conv2 = AGC(out_filter, out_filter, rho_init=rho_init, n_patches=n_patches, n_rho=n_rho, learn_rho=learn_rho)
        self.lin = nn.Linear(in_filter*n_patches, out_filter*n_patches, bias=False) if in_filter != out_filter else None

    def forward(self, mesh:Mesh, x):
        identity = self.lin(x) if self.lin else x
        x = F.relu(self.conv1(mesh, x))
        x = self.conv2(mesh, x)
        x = x + identity
        return F.relu(x)


class ResNet(torch.nn.Module):
    def __init__(self, config:Dict):
        super(ResNet, self).__init__()

        filter_sizes = config["model"]["filter_sizes"]
        n_patches = config["model"]["n_patches"]
        rho_init = config["model"]["rho_init"]
        learn_rho = config["model"]["learn_rho"]
        n_rho = config["model"]["n_rho"]
        output_size = config["model"]["output_size"]
        self.task = config["task"]
        input_type = config["model"]["input"]

        if self.task not in ["classification", "face_segmentation", "vertex_segmentation"]:
            raise ValueError(f"Unknown task type: {self.task}")

        channel_sizes = [filter_sizes[k] * n_patches[k] for k in range(len(filter_sizes))]

        input_size = 3 if input_type=="xyz" else 16
        self.lin0 = nn.Linear(input_size, channel_sizes[0])

        # Stack 1
        self.resnet_block11 = ResNetBlock(filter_sizes[0], filter_sizes[0], rho_init=rho_init[0], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[0])
        self.resnet_block12 = ResNetBlock(filter_sizes[0], filter_sizes[0], rho_init=rho_init[0], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[0])

        # Pool
        self.pooler = MeshPoolBlock()
        self.pool_mlp = nn.Sequential(
            nn.Linear(channel_sizes[0], channel_sizes[0]),
            nn.ReLU(),
            nn.Linear(channel_sizes[0], filter_sizes[0] * n_patches[1])
        )

        # Stack 2
        self.resnet_block21 = ResNetBlock(filter_sizes[0], filter_sizes[1], rho_init=rho_init[1], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[1])
        self.resnet_block22 = ResNetBlock(filter_sizes[1], filter_sizes[1], rho_init=rho_init[1], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[1])


        if self.task == "classification":
            self.classification_layer = nn.Sequential(
                nn.Linear(channel_sizes[1], channel_sizes[1]),
                nn.ReLU(),
                nn.Linear(channel_sizes[1], output_size)
            )
            return

        # Stack 3
        self.resnet_block31 = ResNetBlock(filter_sizes[1], filter_sizes[1], rho_init=rho_init[1], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[1])
        self.resnet_block32 = ResNetBlock(filter_sizes[1], filter_sizes[1], rho_init=rho_init[1], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[1])

        # Unpool
        self.unpool_mlp = nn.Sequential(
            nn.Linear(channel_sizes[1]+channel_sizes[0], channel_sizes[1]),
            nn.ReLU(),
            nn.Linear(channel_sizes[1], channel_sizes[0])
        )

        # Stack 4
        if (filter_sizes[1] * n_patches[1]) % n_patches[0] != 0:
            raise ValueError("The number of channels after unpooling must be divisible by the number of patches in the previous layer.")
        self.resnet_block41 = ResNetBlock(filter_sizes[0], filter_sizes[0], rho_init=rho_init[0], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[0])
        self.resnet_block42 = ResNetBlock(filter_sizes[0], filter_sizes[0], rho_init=rho_init[0], n_rho=n_rho, learn_rho=learn_rho, n_patches=n_patches[0])

        self.lin_final = nn.Sequential(
            nn.Linear(channel_sizes[0], output_size)
        )


    def forward(self, mesh:Mesh, mesh_subsampled:Mesh, model_input:Optional[torch.Tensor]):
        if model_input is None:
            x = mesh.vertices # [V, 3]
        else:
            x = model_input

        # Linear transformation from input positions to channel_sizes[0] features
        x = F.relu(self.lin0(x)) # [V, C0]

        # Stack 1
        x = self.resnet_block11(mesh, x)
        x_prepool = self.resnet_block12(mesh, x)

        # Pooling
        x = self.pooler.pool(mesh, mesh_subsampled, x)
        x = self.pool_mlp(x)  # [S, C0]

        # Stack 2
        x = self.resnet_block21(mesh_subsampled, x)
        x = self.resnet_block22(mesh_subsampled, x)

        if self.task == "classification":
            x = self.classification_layer(x)
            x = torch.sum(x, dim=0, keepdim=True)
            return x

        # Stack 3
        x = self.resnet_block31(mesh_subsampled, x)
        x = self.resnet_block32(mesh_subsampled, x)

        # Unpooling
        x = self.pooler.unpool(x)

        # Concatenate pre-pooling x with post-pooling x
        x = torch.cat((x, x_prepool), dim=1)

        x = self.unpool_mlp(x)

        # Stack 4
        x = self.resnet_block41(mesh, x)
        x = self.resnet_block42(mesh, x)

        x = self.lin_final(x)

        if self.task == "face_segmentation":
            x = x[mesh.faces]
            x = x.sum(dim=1)

        return x
