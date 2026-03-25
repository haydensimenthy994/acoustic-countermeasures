from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits from master metadata.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    master_csv = Path(cfg["paths"]["master_metadata_csv"])
    metadata_dir = Path(cfg["paths"]["metadata_dir"])

    train_ratio = cfg["splits"]["train_ratio"]
    val_ratio = cfg["splits"]["val_ratio"]
    test_ratio = cfg["splits"]["test_ratio"]
    seed = cfg["splits"]["random_seed"]

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = pd.read_csv(master_csv)

    df = df.dropna(subset=["label"]).copy()

   # Balance classes by undersampling the majority class
    min_count = df["label"].value_counts().min()
    balanced_parts = []
    for label, group in df.groupby("label"):
        balanced_parts.append(group.sample(n=min_count, random_state=seed))
    df = pd.concat(balanced_parts).reset_index(drop=True)
    print(f"Balanced dataset: {min_count} samples per class ({min_count * 2} total)")
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    split_csv = metadata_dir / "split_metadata.csv"
    split_df.to_csv(split_csv, index=False)

    print(f"Saved split metadata to {split_csv}")
    print(split_df["split"].value_counts())
    print(split_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()