import argparse
import json
import os
import matplotlib.pyplot as plt
import torch


plt.rcParams.update(
    {
        "font.size": 20,  # default font size
        "axes.labelsize": 26,
        "xtick.labelsize": 16,  # x-axis tick labels
        "ytick.labelsize": 16,  # y-axis tick labels
    }
)


def pad_to_max_length(loss_list):
    max_len = max(len(l) for l in loss_list)
    padded = []
    for l in loss_list:
        if len(l) < max_len:
            pad_tensor = l + l[-1:] * (max_len - len(l))
        else:
            pad_tensor = l[:max_len]
        padded_l = torch.tensor(pad_tensor, dtype=torch.float32)
        padded.append(padded_l)
    return torch.stack(padded)


def get_mean_std(tensor_dict):
    loss_tensors = []
    for name in tensor_dict.keys():
        loss_tensor = pad_to_max_length(tensor_dict[name])
        loss_tensors.append(loss_tensor)

    mean_tensors = [t.mean(dim=0).cpu() for t in loss_tensors]
    std_tensors = [t.std(dim=0).cpu() for t in loss_tensors]

    return mean_tensors, std_tensors


def get_median_iqr(tensor_dict):
    loss_tensors = []
    for name in tensor_dict.keys():
        loss_tensor = pad_to_max_length(tensor_dict[name])
        loss_tensors.append(loss_tensor)

    medians = [t.median(dim=0).values.cpu() for t in loss_tensors]
    p25 = [t.quantile(0.25, dim=0).cpu() for t in loss_tensors]
    p75 = [t.quantile(0.75, dim=0).cpu() for t in loss_tensors]

    return medians, p25, p75


def main(mesh_name, distribution, min_y, max_y):
    json_path = f"./runs/results_{mesh_name}_{distribution}.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    fig_dir = f"./runs/{mesh_name}_{distribution}"
    os.makedirs(fig_dir, exist_ok=True)

    results = data["result"]
    models = list(results.keys())
    total_losses = data["loss"]
    total_gradient = data["gradient"]
    total_function_calls = data["function_calls"]

    median_loss, iqr25_loss, iqr75_loss = get_median_iqr(total_losses)
    mean_gradient, std_gradient = get_mean_std(total_gradient)
    mean_function_calls, std_function_calls = get_mean_std(total_function_calls)

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(models):
        plt.plot(
            torch.arange(median_loss[i].shape[0]),
            median_loss[i],
            label=f"{name}",
            alpha=0.6,
            color=colors[i % len(colors)],
        )
        plt.fill_between(
            torch.arange(median_loss[i].shape[0]),
            iqr25_loss[i],
            iqr75_loss[i],
            alpha=0.2,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_dir}/loss_comparison.svg")

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(models):
        plt.plot(
            torch.arange(median_loss[i].shape[0]),
            median_loss[i],
            label=f"{name}",
            alpha=0.6,
            color=colors[i % len(colors)],
        )
        plt.fill_between(
            torch.arange(median_loss[i].shape[0]),
            iqr25_loss[i],
            iqr75_loss[i],
            alpha=0.2,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.ylim(top=max_y, bottom=min_y)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_dir}/loss_comparison_zoomed.svg")

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(models):
        plt.plot(
            torch.arange(mean_gradient[i].shape[0]),
            mean_gradient[i],
            label=f"{name}",
            alpha=0.6,
            color=colors[i % len(colors)],
        )
        plt.fill_between(
            torch.arange(mean_gradient[i].shape[0]),
            mean_gradient[i] - std_gradient[i],
            mean_gradient[i] + std_gradient[i],
            alpha=0.2,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_dir}/gradient_comparison.svg")

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(models):
        plt.plot(
            torch.arange(mean_function_calls[i].shape[0]),
            mean_function_calls[i],
            label=f"{name}",
            alpha=0.6,
            color=colors[i % len(colors)],
        )
        plt.fill_between(
            torch.arange(mean_function_calls[i].shape[0]),
            mean_function_calls[i] - std_function_calls[i],
            mean_function_calls[i] + std_function_calls[i],
            alpha=0.2,
            color=colors[i % len(colors)],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Function Calls")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_dir}/function_calls_comparison.svg")


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
        "--min", type=float, default=0.0, help="Minimum y value for zoomed loss plot."
    )
    parser.add_argument(
        "--max", type=float, default=10.0, help="Maximum y value for zoomed loss plot."
    )

    args = parser.parse_args()
    main(args.mesh_name, args.distribution, args.min, args.max)
