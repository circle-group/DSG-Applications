import os
import torch
from digeo import load_mesh_from_file
from digeo.ops import uniform_sampling
from digeo.optim import mesh_lbfgs, mesh_gd
from collections import defaultdict
import json
import argparse

from utils import sample_cluster, sanitize_results
from loss import LossGCVT

vertex_dict = {"scorpion": 2849, "spot": 1855, "skull": 21936}

sigma_dict = {"scorpion": 0.4, "spot": 0.2, "skull": 0.8}


def main(mesh_name, distribution, device):
    num_samples = 50
    total_epochs = 20
    mesh = load_mesh_from_file(f"data/{mesh_name}.obj", device=device)
    loss_func = LossGCVT(mesh)

    models = {
        "Mesh-LBFGS": lambda x: mesh_lbfgs(
            mesh=mesh,
            x=x,
            loss_func=loss_func,
            max_iter=50,
            tol=1e-9,
            patience=2000,
            list_size=10,
            lr=0.5,
        ),
        "Lloyd": lambda x: mesh_gd(
            mesh=mesh,
            x=x,
            loss_func=loss_func,
            max_iter=50,
            tol=1e-9,
            patience=2000,
            use_line_search=False,
            lr=1.0,
        ),
    }
    results = {name: defaultdict(list) for name in models.keys()}
    total_losses = defaultdict(list)
    total_gradient = defaultdict(list)
    total_function_calls = defaultdict(list)

    for epoch in range(total_epochs):
        if distribution == "cluster":
            samples = sample_cluster(
                mesh,
                source_vertex=vertex_dict[mesh_name],
                sigma=sigma_dict[mesh_name],
                num_samples=num_samples,
            )
        elif distribution == "uniform":
            samples = uniform_sampling(mesh, num_samples)
        else:
            raise ValueError("Unknown distribution")

        x0 = samples.to(device).detach()
        print(f"Epoch {epoch + 1} / {total_epochs}")
        for name in models.keys():
            result, logs = models[name](x0.clone())
            results[name]["loss"].append(logs["loss"][-1])
            results[name]["time"].append(logs["time"])
            results[name]["iterations"].append(len(logs["loss"]))
            total_losses[name].append(logs["loss"])
            total_gradient[name].append(logs["mean_dir"])
            total_function_calls[name].append(logs["function_calls"])

    for name, result in results.items():
        print(f"{name} mean loss:", torch.tensor(result["loss"]).mean().item())

    json_data = {
        "result": results,
        "loss": total_losses,
        "gradient": total_gradient,
        "function_calls": total_function_calls,
    }
    json_data = sanitize_results(json_data)

    run_dir = "./runs"
    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/results_{mesh_name}_{distribution}.json", "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mesh_name",
        type=str,
        help="Name of the mesh file to process. Should be in the data/ folder",
    )
    parser.add_argument(
        "distribution",
        type=str,
        choices=["uniform", "cluster"],
        default="cluster",
        help="Initial sampling distribution to use.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the computations on."
    )

    args = parser.parse_args()
    main(args.mesh_name, args.distribution, args.device)
