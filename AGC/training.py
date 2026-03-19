import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import yaml
import json
import os

from models import ResNet


def train(config_path, train_dataloader, test_dataloader, save_dir, device):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"))

    epochs = config["optim"]["epochs"]
    label_smoothing = config["optim"]["label_smoothing"]
    gradient_clipping = config["optim"]["gradient_clipping"]
    test_every = config["optim"]["test_every"]

    train_info_path = os.path.join(save_dir, "train_info.json")

    model = ResNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    lr_scheduler = None
    if config["optim"]["lr_scheduler"]["type"] == "cosine":
        warmup = config["optim"]["lr_scheduler"].get("warmup", 0)
        eta = config["optim"]["lr_scheduler"].get("min_lr")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=eta
        )
    if config["optim"]["lr_scheduler"]["type"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["optim"]["lr_scheduler"]["step"],
            gamma=config["optim"]["lr_scheduler"]["gamma"],
        )
    if "warmup" in config["optim"]["lr_scheduler"]:
        warmup = config["optim"]["lr_scheduler"]["warmup"]
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup]
        )

    total_train_loss = []
    total_train_acc = []
    total_test_acc = []
    total_test_epochs = []

    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}, Training"
        )
        for mesh, subsampled_mesh, model_input, label in progress_bar:
            optimizer.zero_grad()

            mesh = mesh.to(device)
            subsampled_mesh = subsampled_mesh.to(device)
            label = label.to(device)
            model_input = model_input.to(device) if model_input is not None else None

            outputs = model(mesh, subsampled_mesh, model_input)

            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=-1) == label).float().mean().item()
            progress_bar.set_postfix({"loss": loss.item()})

        if epoch % test_every == 0 or epoch == epochs - 1:
            model.eval()
            test_acc = 0.0
            with torch.no_grad():
                for mesh, subsampled_mesh, model_input, label in tqdm(
                    test_dataloader, desc=f"Epoch {epoch + 1}/{epochs}, Testing"
                ):
                    model_input = (
                        model_input.to(device) if model_input is not None else None
                    )
                    outputs = model(
                        mesh.to(device), subsampled_mesh.to(device), model_input
                    )
                    label = label.to(device)
                    loss = criterion(outputs, label)
                    test_acc += (outputs.argmax(dim=-1) == label).float().mean().item()

            total_test_acc.append(test_acc / len(test_dataloader))
            total_test_epochs.append(epoch)

            model_path = os.path.join(save_dir, "model.pth")
            torch.save(model.state_dict(), model_path)

        total_train_loss.append(train_loss / len(train_dataloader))
        total_train_acc.append(train_acc / len(train_dataloader))

        if test_acc / len(test_dataloader) > best_test_acc:
            best_test_acc = test_acc / len(test_dataloader)

        with open(train_info_path, "w") as f:
            json.dump(
                {
                    "best_test_acc": best_test_acc,
                    "train_loss": total_train_loss,
                    "train_acc": total_train_acc,
                    "test_epochs": total_test_epochs,
                    "test_acc": total_test_acc,
                },
                f,
            )

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_dataloader):.4f}, Train Acc: {train_acc / len(train_dataloader):.4f}, Best Test Acc: {best_test_acc}"
        )
        if epoch % test_every == 0 or epoch == epochs - 1:
            print(f"Test Acc: {test_acc / len(test_dataloader):.4f}")

        plt.figure()
        plt.plot(range(len(total_train_loss)), total_train_loss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(range(len(total_train_acc)), total_train_acc, label="Train Accuracy")
        plt.plot(total_test_epochs, total_test_acc, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Test Accuracy")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
        plt.close()
