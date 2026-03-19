import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from digeo import load_mesh_from_file
from digeo.nn import BiharmonicDistance
from digeo.ops import trace_geodesics, uniform_sampling
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from model import MeshFlow
from utils import chamfer_distance, convert_to_meshpointbatch, kld

checkpoint_dir = None


def main(config_file, mesh_file, eigfn_file, device):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    n_points = config["n_points"]
    n_points_test = config["n_points_test"]
    epochs = config["epochs"]
    batch_size_OT = config["batch_size_OT"]
    batch_size = config["batch_size"]
    k_step = config["k_step"]
    base_lr = config["optim"]["lr"]
    save_every = config["save_every"]
    eval_every = config["eval_every"]
    avoid_holes = config["avoid_holes"]

    mesh = load_mesh_from_file(mesh_file).to(device)

    output_dir = os.path.join(os.path.dirname(__file__), "output")

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_dir, Path(eigfn_file).stem + "_" + dt_string)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(eigfn_file, "rb") as f:
        point_arr = np.load(f)
        target_points = torch.from_numpy(point_arr[:n_points, :]).to(device)
        test_points = torch.from_numpy(
            point_arr[n_points : n_points + n_points_test, :]
        ).to(device)
    target_meshpoints = convert_to_meshpointbatch(mesh, target_points)
    test_meshpoints = convert_to_meshpointbatch(mesh, test_points)

    model = MeshFlow(mesh)
    if checkpoint_dir is not None:
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model_last.pth"))
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    if config["optim"]["scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config["optim"]["scheduler"]["eta_min"]
        )
    else:
        lr_scheduler = None

    criterion = BiharmonicDistance(mesh, k_eig=128)

    plot_dict = defaultdict(list)
    best_kld = float("inf")
    best_cd = float("inf")

    progress_bar = tqdm(range(epochs), unit="epoch")
    for epoch in progress_bar:
        idx = torch.randperm(n_points)
        running_loss = 0.0

        model.train()
        for k in range(0, len(idx), batch_size_OT):
            with torch.no_grad():
                k_end = min(k + batch_size_OT, len(idx))
                x1 = target_meshpoints[idx[k:k_end]].detach().clone()
                x0 = uniform_sampling(mesh, len(x1)).to(device)
                pairwise_distances = criterion(x0, x1, pairwise=True)

                pairwise_distances = pairwise_distances**2

                row_idx, col_idx = linear_sum_assignment(
                    pairwise_distances.detach().cpu().numpy()
                )
                row_idx = torch.from_numpy(row_idx).to(device)
                col_idx = torch.from_numpy(col_idx).to(device)

                x0 = x0[row_idx]
                x1 = x1[col_idx]

            for i in range(batch_size_OT // batch_size + 1):
                k_start = i * batch_size
                k_end = min((i + 1) * batch_size, batch_size_OT, len(row_idx))
                if k_start >= k_end:
                    continue

                x_batch = x0[k_start:k_end]
                x1_batch = x1[k_start:k_end]

                for _ in range(k_step):
                    v = model(x_batch) / k_step
                    x_batch, _ = trace_geodesics(
                        mesh,
                        x_batch,
                        v,
                        gradient="gfd",
                        avoid_holes=avoid_holes,
                    )

                loss = criterion(x_batch, x1_batch)
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item() * len(x1_batch)

        if epoch % eval_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                model.eval()
                x = uniform_sampling(mesh, n_points_test).to(device)
                v = model(x)
                for _ in range(k_step):
                    x, _ = trace_geodesics(
                        mesh, x, v / k_step, gradient="none", avoid_holes=avoid_holes
                    )
                    v = model(x)

                kld_value = kld(
                    mesh, x.interpolate(mesh), test_meshpoints.interpolate(mesh)
                )

                D = criterion(x, test_meshpoints, pairwise=True)
                cd_value = chamfer_distance(D).item()

            if kld_value < best_kld:
                best_kld = kld_value
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "model_best.pth")
                )

            best_cd = min(best_cd, cd_value)
            print(
                f"Eval Epoch {epoch}: KLD {kld_value:.4f}, Chamfer Distance {cd_value:.4f}"
            )

            plot_dict["kld"].append(kld_value)
            plot_dict["chamfer_distance"].append(cd_value)

        train_loss = running_loss / n_points
        progress_bar.set_description(
            f"Loss {train_loss:.4f}, KLD {kld_value:.4f}, CD {cd_value:.4f}"
        )
        plot_dict["train_loss"].append(train_loss)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(output_dir, "model_last.pth"))

            plt.figure()
            plt.plot(plot_dict["train_loss"])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(output_dir, "train_loss_plot.png"))
            plt.close()

            plt.figure()
            plt.plot(eval_every * np.arange(len(plot_dict["kld"])), plot_dict["kld"])
            plt.xlabel("Epoch")
            plt.ylabel("KLD")
            plt.savefig(os.path.join(output_dir, "kld_plot.png"))
            plt.close()

            plt.figure()
            plt.plot(
                eval_every * np.arange(len(plot_dict["chamfer_distance"])),
                plot_dict["chamfer_distance"],
            )
            plt.xlabel("Epoch")
            plt.ylabel("Chamfer Distance")
            plt.savefig(os.path.join(output_dir, "cd_plot.png"))
            plt.close()

            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(plot_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_config = "configs/ot.yaml"
    default_mesh_file = "data/bunny_simp.obj"
    default_eigfn_file = "data/bunny_eigfn009.npy"

    parser.add_argument(
        "--config", type=str, default=default_config, help="Path to the config file"
    )
    parser.add_argument(
        "--mesh_file", type=str, default=default_mesh_file, help="Path to the mesh file"
    )
    parser.add_argument(
        "--eigfn_file",
        type=str,
        default=default_eigfn_file,
        help="Path to the eigenfunction file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    main(args.config, args.mesh_file, args.eigfn_file, args.device)
