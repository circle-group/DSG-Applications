from pathlib import Path
import torch
import matplotlib.pyplot as plt
import json
import yaml
import numpy as np

from models import ResNet

output_dirs = [
    "sig17_resunet19",
    "sig17_resunet20"
]

root_dir = Path(__file__).parent / "output"

def vis_loss():
    data = {}

    for output_dir in output_dirs:
        output_path = root_dir / output_dir
        info_path = output_path / "train_info.json"

        with open(info_path, 'r') as f:
            data[output_dir] = json.load(f)

    plt.figure()
    for output_dir in output_dirs:
        test_acc = np.array(data[output_dir]['test_acc'])
        window = 5
        smoothed = np.convolve(test_acc, np.ones(window)/window, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, label=output_dir)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()

    plt.figure()
    for output_dir in output_dirs:
        plt.plot(range(len(data[output_dir]['train_acc'][10:])), data[output_dir]['train_acc'][10:], label=output_dir)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Epoch")
    plt.legend()


def vis_rho():
    for output_dir in output_dirs:
        output_path = root_dir / output_dir
        config_path = output_path / "config.yaml"
        config = yaml.safe_load(open(config_path))
        model = ResNet(config)
        model_path = output_path / "model.pth"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        group_names = []
        k = 0
        colors = ["red", "blue", "green", "orange", "purple", "pink", "brown", "gray"]
        plt.figure()
        for name, layer in model.named_modules():
            if layer.__class__.__name__ == "AGCN":
                layer_name = name.removeprefix("model.")
                group_names.append(layer_name)
                for i in range(layer.rho.shape[3]):
                    y1 = abs(layer.rho[0, 0, 0, i, 0].item())
                    y2 = abs(layer.rho[0, 1, 0, i, 0].item())
                    plt.scatter([k, k], [y1, y2], color=colors[i%len(colors)], s=10)
                k += 1

        plt.xticks(ticks=range(len(group_names)), labels=group_names)
        plt.xticks(rotation=90)
        plt.ylabel("Param Value")
        plt.tight_layout()
        plt.title(f"AGCN Ring Size Parameters for {output_dir}")

        group_names = []
        k = 0
        plt.figure()
        for name, layer in model.named_modules():
            if layer.__class__.__name__ == "AGCN":
                layer_name = name.removeprefix("model.")
                group_names.append(layer_name)

                # Collect all rho values for this layer
                values = []
                for i in range(layer.rho.shape[3]):
                    y1 = abs(layer.rho[0, 0, 0, i, 0].item())
                    y2 = abs(layer.rho[0, 1, 0, i, 0].item())
                    values.extend([y1, y2])  # add both values

                # Draw a box plot for this layer
                plt.boxplot(values, positions=[k], widths=0.6, patch_artist=True)
                k += 1
        plt.xticks(ticks=range(len(group_names)), labels=group_names)
        plt.xticks(rotation=90)
        plt.ylabel("Param Value")
        plt.tight_layout()
        plt.title(f"AGCN Ring Size Parameters for {output_dir}")


if __name__ == "__main__":
    vis_loss()
    vis_rho()
    plt.show()