import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from model import get_model
from dataset import get_dataloaders


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    config_path = "/app/configs/training_config.yaml"
    exists = Path(config_path).exists()
    if not exists:
        config_path = "configs/training_config.yaml"

    config = load_config(config_path)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    arch = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    model = get_model(arch, num_classes)
    model = model.to(device)

    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    learning_rate = config["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir_path = config["output"]["checkpoint_dir"]
    checkpoint_dir = Path(checkpoint_dir_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["output"]["model_name"]
    total_epochs = config["training"]["epochs"]

    for epoch in range(total_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_number = epoch + 1
        log = {
            "epoch": epoch_number,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log), flush=True)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / model_name
            checkpoint = {"model_state_dict": model.state_dict()}
            torch.save(checkpoint, save_path)

            saved_log = {"event": "checkpoint_saved", "path": str(save_path)}
            print(json.dumps(saved_log), flush=True)
        else:
            patience_counter = patience_counter + 1
            no_patience_left = patience_counter >= patience
            if no_patience_left:
                stop_log = {"event": "early_stopping", "epoch": epoch_number}
                print(json.dumps(stop_log), flush=True)
                break

    final_loss = round(best_val_loss, 4)
    done_log = {"event": "training_complete", "best_val_loss": final_loss}
    print(json.dumps(done_log), flush=True)


if __name__ == "__main__":
    main()
