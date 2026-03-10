from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_label(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]

    if "drone" in parts:
        return "drone"
    if "no_drone" in parts or "nodrone" in parts or "background" in parts:
        return "no_drone"

    return None


def collect_audio_files(root: Path, dataset_name: str) -> list[dict]:
    rows = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue

        label = infer_label(path)
        rows.append(
            {
                "filepath": str(path).replace("\\", "/"),
                "filename": path.name,
                "dataset": dataset_name,
                "label": label,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master metadata manifest from processed audio directories.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--uav-dir", default="data/interim/audio/uav_db")
    parser.add_argument("--audioset-dir", default="data/interim/audio/audioset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_dir = Path(cfg["paths"]["metadata_dir"])
    output_csv = Path(cfg["paths"]["master_metadata_csv"])

    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    uav_dir = Path(args.uav_dir)
    audioset_dir = Path(args.audioset_dir)

    if uav_dir.exists():
        rows.extend(collect_audio_files(uav_dir, "uav_db"))

    if audioset_dir.exists():
        rows.extend(collect_audio_files(audioset_dir, "audioset"))

    df = pd.DataFrame(rows)

    if df.empty:
        print("No audio files found. Manifest not created.")
        return

    df["label_source"] = df["label"].apply(lambda x: "folder_name" if pd.notna(x) else "unknown")
    df.to_csv(output_csv, index=False)

    print(f"Saved manifest to {output_csv}")
    print(df["dataset"].value_counts(dropna=False))
    print(df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()