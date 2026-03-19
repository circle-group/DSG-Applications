import torch
from typing import List, Tuple
from pathlib import Path
import yaml
import argparse

from dataloader import get_dataloader
from training import train


def get_labels() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    train_labels = []
    train_dir = Path("data/sig17_seg_benchmark/segs/train/adobe")
    for f in sorted(train_dir.rglob("*.txt")):
        with open(f, "r") as file:
            labels_idx = [int(line.strip()) for line in file.readlines()]
            train_labels.append(torch.tensor(labels_idx, dtype=torch.int64) - 1)

    train_dir = Path("data/sig17_seg_benchmark/segs/train/faust")
    mesh_count = len(
        list(Path("data/sig17_seg_benchmark/meshes/train/faust").rglob("*.off"))
    )
    for f in sorted(train_dir.rglob("*.txt")):
        with open(f, "r") as file:
            labels_idx = [int(line.strip()) for line in file.readlines()]
            for _ in range(mesh_count):
                train_labels.append(torch.tensor(labels_idx, dtype=torch.int64) - 1)

    train_dir = Path("data/sig17_seg_benchmark/segs/train/MIT_animation")
    mesh_root = Path("data/sig17_seg_benchmark/meshes/train/MIT_animation")
    train_paths = sorted(list(train_dir.rglob("*.txt")))
    mesh_paths = [p for p in mesh_root.iterdir()]
    for k, f in enumerate(train_paths):
        with open(f, "r") as file:
            labels_idx = [int(line.strip()) for line in file.readlines()]
            mesh_count = len(list(mesh_paths[k].rglob("*.off")))
            for _ in range(mesh_count):
                train_labels.append(torch.tensor(labels_idx, dtype=torch.int64) - 1)

    train_dir = Path("data/sig17_seg_benchmark/segs/train/scape")
    mesh_count = len(
        list(Path("data/sig17_seg_benchmark/meshes/train/scape").rglob("*.off"))
    )
    for f in sorted(train_dir.rglob("*.txt")):
        with open(f, "r") as file:
            labels_idx = [int(line.strip()) for line in file.readlines()]
            for _ in range(mesh_count):
                train_labels.append(torch.tensor(labels_idx, dtype=torch.int64) - 1)

    val_labels = []
    val_dir = Path("data/sig17_seg_benchmark/segs/test")
    for f in sorted(val_dir.rglob("*.txt")):
        with open(f, "r") as file:
            labels_idx = [int(line.strip()) for line in file.readlines()]
            val_labels.append(torch.tensor(labels_idx, dtype=torch.int64) - 1)
    return train_labels, val_labels


def get_mesh_paths() -> Tuple[List[str], List[str]]:
    train_dir = Path("data/sig17_seg_benchmark/meshes/train")
    val_dir = Path("data/sig17_seg_benchmark/meshes/test")
    train_paths = [str(p) for p in train_dir.rglob("*.off")]
    val_paths = [str(p) for p in val_dir.rglob("*.off")]
    return sorted(train_paths), sorted(val_paths)


def main(config_path, save_dir, device):
    train_mesh_paths, val_mesh_paths = get_mesh_paths()
    train_labels, test_labels = get_labels()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train_dataloader = get_dataloader(
        config, train_mesh_paths, train_labels, shuffle=True
    )
    test_dataloader = get_dataloader(config, val_mesh_paths, test_labels, shuffle=False)

    train(
        config_path=config_path,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        save_dir=save_dir,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script with arguments.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    main(args.config, args.save, args.device)
