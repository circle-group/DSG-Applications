from digeo import load_mesh_from_file
from digeo.ops import uniform_sampling
from digeo.optim import mesh_lbfgs, mesh_gd
import matplotlib.pyplot as plt
import argparse

from loss import LossGCVT
from ui import visualize_voronoi, visualize_points


def main(mesh_file, optimizer="lbfgs", device="cpu"):
    mesh = load_mesh_from_file(mesh_file, device=device)
    x0 = uniform_sampling(mesh, 50)
    loss_func = LossGCVT(mesh)

    print("Starting optimization...")
    if optimizer == "lbfgs":
        result, logs = mesh_lbfgs(
            mesh=mesh,
            x=x0,
            loss_func=loss_func,
            max_iter=20,
            tol=1e-8,
            list_size=10,
            lr=1.0,
            patience=10,
        )
    elif optimizer == "gd":
        result, logs = mesh_gd(
            mesh=mesh,
            x=x0,
            loss_func=loss_func,
            max_iter=20,
            tol=1e-8,
            lr=1.0,
            use_line_search=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    print(f"Final loss: {logs['loss'][-1]}")
    print(f"Time taken: {logs['time']} seconds")

    visualize_points(mesh, x0)
    visualize_voronoi(mesh, result, logs["regions"])

    plt.figure()
    plt.plot(logs["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{optimizer} Loss")

    plt.figure()
    plt.plot(logs["step_size"])
    plt.xlabel("Iteration")
    plt.ylabel("Step Size")
    plt.yscale("log")
    plt.title(f"{optimizer} Step Size")

    plt.figure()
    plt.plot(logs["mean_dir"])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Direction Norm")
    plt.title(f"{optimizer} Mean Direction Norm")

    plt.figure()
    plt.plot(logs["function_calls"])
    plt.xlabel("Iteration")
    plt.ylabel("Function Calls")
    plt.title(f"{optimizer} Function Calls")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mesh_path",
        type=str,
        help="Path to the mesh file to process.",
    )
    parser.add_argument(
        "optim",
        type=str,
        choices=["lbfgs", "gd"],
        default="lbfgs",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the computations on."
    )

    args = parser.parse_args()
    main(args.mesh_path, args.optim, device=args.device)
