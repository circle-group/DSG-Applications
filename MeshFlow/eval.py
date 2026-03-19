from pathlib import Path
import argparse

import numpy as np
import torch
from digeo import load_mesh_from_file
from digeo.nn import BiharmonicDistance
from digeo.ops import trace_geodesics, uniform_sampling

from ui import visualize_mesh_and_points, visualize_paths
from model import MeshFlow
from utils import (
    chamfer_distance,
    compute_exact_loglikelihood,
    convert_to_meshpointbatch,
    kld,
)

def main(checkpoint_path, mesh_path, eigenfn_path, device, num_steps, num_samples, visualize):
    mesh = load_mesh_from_file(mesh_path, device=device)
    model = MeshFlow(mesh)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model = model.to(device)

    with open(eigenfn_path, "rb") as f:
        target_points = np.load(f)[50000 : 50000 + num_samples]
    target_meshpoints = convert_to_meshpointbatch(mesh, target_points)

    nll = compute_exact_loglikelihood(mesh, model, target_meshpoints, num_steps=1000)
    print("NLL on target points (1000 steps):", -nll.mean().item())

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(f"\n=== Evaluating with {num_steps} steps ===")
    with torch.no_grad():
        model.eval()
        torch.cuda.synchronize()
        start.record()
        x_train = uniform_sampling(mesh, num_samples).to(device)
        v_init = model(x_train)
        v = v_init.clone()
        x = x_train.clone()
        for k in range(num_steps):
            x, info = trace_geodesics(mesh, x, v / num_steps, gradient="none")
            v = model(x)

        end.record()
        torch.cuda.synchronize()
        print(f"Geodesic Tracing Time: {start.elapsed_time(end) / 1000} s")

        print(
            "KLD:", kld(mesh, x.interpolate(mesh), target_meshpoints.interpolate(mesh))
        )

        criterion = BiharmonicDistance(mesh)

        D = criterion(x, target_meshpoints, pairwise=True)
        cd = chamfer_distance(D)
        print("Chamfer Distance:", cd.item())

    if not visualize:
        return

    visualize_mesh_and_points(mesh, x_train, desc="Train Points")
    visualize_mesh_and_points(mesh, x, desc="Predicted Points")
    visualize_mesh_and_points(mesh, target_meshpoints, desc="Target Points", show_points=False)

    paths = None
    with torch.no_grad():
        model.eval()
        x_train = target_meshpoints[:500].clone()
        v_init = model(x_train)
        v = v_init.clone()
        x = x_train.clone()
        for k in range(num_steps):
            x, info = trace_geodesics(mesh, x, -v / num_steps, gradient="none", debug=True)
            v = model(x)

            if paths is None:
                paths = [info.get_path(k) for k in range(len(x))]
            else:
                paths = [torch.cat((paths[k], info.get_path(k))) for k in range(len(x))]

    visualize_mesh_and_points(mesh, x_train, desc="Train Points")
    visualize_mesh_and_points(mesh, x, desc="Predicted Points")
    visualize_paths(mesh, paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MeshFlow model")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--mesh_path", type=str, help="Path to input mesh")
    parser.add_argument("--eigenfn_path", type=str, help="Path to eigenfunction samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of steps for geodesic tracing")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of points to sample for evaluation")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize results")
    args = parser.parse_args()

    main(
        checkpoint_path=Path(args.checkpoint_path),
        mesh_path=Path(args.mesh_path),
        eigenfn_path=Path(args.eigenfn_path),
        device=args.device,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        visualize=args.visualize,
    )