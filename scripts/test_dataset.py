from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DroneAudioDataset


def main() -> None:
    metadata_csv = Path("data/metadata/split_metadata.csv")

    if not metadata_csv.exists():
        print("split_metadata.csv not found yet.")
        print("That is expected until you preprocess audio and build the manifest.")
        return

    ds = DroneAudioDataset(
        metadata_csv=str(metadata_csv),
        config_path="configs/data.yaml",
        split="train",
        fixed_duration_sec=5.0,
    )

    print(f"Dataset size: {len(ds)}")

    if len(ds) == 0:
        print("Dataset is empty.")
        return

    sample = ds[0]
    print("Feature shape:", sample["features"].shape)
    print("Label:", sample["label"].item())
    print("File:", sample["filepath"])


if __name__ == "__main__":
    main()