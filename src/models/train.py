from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DroneAudioDataset
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed
from src.utils.logger import get_log_file, log_line


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the proxy drone audio classifier.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)

    seed = train_cfg["seed"]["value"]
    set_seed(seed)

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = get_log_file(str(log_dir), prefix="train")
    def logger_info(msg):
        log_line(msg, logfile)
    logger_info("Starting training run")

    # Device
    device_cfg = train_cfg["train"]["device"]
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger_info(f"Using device: {device}")

    # Dataset
    split_csv = data_cfg["paths"]["master_metadata_csv"].replace(
        "master_metadata.csv", "split_metadata.csv"
    )

    train_dataset = DroneAudioDataset(
        metadata_csv=split_csv,
        config_path=args.data_config,
        split="train",
        fixed_duration_sec=5.0,
    )
    val_dataset = DroneAudioDataset(
        metadata_csv=split_csv,
        config_path=args.data_config,
        split="val",
        fixed_duration_sec=5.0,
    )

    logger_info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    batch_size = train_cfg["train"]["batch_size"]
    num_workers = train_cfg["train"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = ProxyAudioCNN(
        input_channels=model_cfg["model"]["input_channels"],
        num_classes=model_cfg["model"]["num_classes"],
        dropout=model_cfg["model"]["dropout"],
    ).to(device)
    logger_info(f"Model: {model_cfg['model']['name']}")

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["optimizer"]["lr"],
        weight_decay=model_cfg["optimizer"]["weight_decay"],
    )

    epochs = train_cfg["train"]["epochs"]
    log_every = train_cfg["train"]["log_every"]
    save_best_only = train_cfg["train"]["save_best_only"]

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for step, batch in enumerate(train_loader, 1):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if step % log_every == 0:
                running_acc = correct / total
                logger_info(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {running_acc:.4f}"
                )

        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        logger_info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if save_best_only:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                logger_info(f"Saved best model with val_acc={val_acc:.4f} to {ckpt_path}")
        else:
            ckpt_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger_info(f"Saved checkpoint to {ckpt_path}")

    logger_info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()