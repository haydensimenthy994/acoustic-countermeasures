"""
Build a cross-dataset test manifest from SWARM-AUDIO-DATASET.

Why a separate manifest:
  Our existing models were trained on the legacy dataset whose
  random file-level split is suspected of leaking source recordings
  across train/val/test (see scripts/check_duration_leakage.py and
  the project README). Evaluating those trained models on
  SWARM-AUDIO-DATASET — a corpus they have never seen — gives an
  honest cross-dataset generalisation estimate without re-training.

Output schema:
  filepath        absolute path to a 4 s WAV (auto-resampled at load)
  filename        leaf file name (for traceability)
  label           "drone" | "no_drone"
  source_dataset  "alemadi" | "trident" | "wildlife_xenocanto"
                  | "wildlife_other" | "alemadi_background"
  drone_type      "bebop" | "membo" | "drone_alemadi" | "trident"
                  | NaN (for backgrounds)
  group_id        identifier of the parent recording — for any future
                  GroupShuffleSplit retraining. Multiple chunks
                  cut from the same parent recording share group_id.
  split           always "test" — this manifest is held out by design.

Usage:
  python scripts/build_swarm_manifest.py \\
      --root  C:/Users/hayde/Downloads/SWARM-AUDIO-DATASET/SWARM-AUDIO-DATASET \\
      --out   data/metadata/swarm_test_manifest.csv

The script only walks `2_segments_4s/` because those are uniform 4 s
clips that drop straight into the existing DroneAudioDataset (5 s
target → padded by 1 s of zeros, no truncation).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Source-group extraction
# ---------------------------------------------------------------------------
# Different sub-corpora use different filename conventions. Examples:
#
#   Alemadi drone segments    : B_S2_D1_067-bebop_000__0.wav
#                               → parent rec = "B_S2_D1_067"
#   Trident drone segments    : trident_d1_0_seg1.wav
#                               → parent rec = "trident_d1_0"
#   Alemadi background (ESC50): 1-100032-A-00.wav
#                               → parent rec = "100032"
#   Wildlife (Xeno-Canto)     : XC1015738 - House Sparrow - X_0.wav
#                               → parent rec = "XC1015738"
#   Wildlife (other)          : doing_the_dishes000_0.wav
#                               → parent rec = "doing_the_dishes000"
#
# The regex patterns below extract the parent-recording identifier so
# every chunk derived from the same source ends up in the same group.

ALEMADI_DRONE_RE  = re.compile(r"^(B_S\d+_D\d+_\d+)-")
TRIDENT_DRONE_RE  = re.compile(r"^(trident_d\d+_\d+)_")
ESC50_BG_RE       = re.compile(r"^\d+-(\d+)-")
XENO_CANTO_RE     = re.compile(r"^(XC\d+)")


def _extract_group_id(filename: str, fallback: str) -> str:
    """Best-effort parent-recording id from filename. Falls back to
    `fallback` (typically the filename minus a trailing _<digit>)."""
    for pattern in (ALEMADI_DRONE_RE, TRIDENT_DRONE_RE, XENO_CANTO_RE):
        m = pattern.match(filename)
        if m:
            return m.group(1)

    # ESC-50 style "<fold>-<clip>-<take>-<class>"
    m = ESC50_BG_RE.match(filename)
    if m:
        return m.group(1)

    # Generic: drop a trailing "_<digit>+\.wav" so chunks of the same
    # parent recording share an id.
    stem = Path(filename).stem
    return re.sub(r"_\d+$", "", stem)


# ---------------------------------------------------------------------------
# Path → (label, source_dataset, drone_type)
# ---------------------------------------------------------------------------

def _classify(rel_parts: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    """Map the relative path parts under `2_segments_4s/` to
    (label, source_dataset, drone_type). Returns None to skip."""
    if len(rel_parts) < 2:
        return None

    top = rel_parts[0]   # "drones" or "background"
    second = rel_parts[1]

    if top == "drones":
        if second == "alemadi" and len(rel_parts) >= 3:
            sub = rel_parts[2]                  # bebop / membo / drone
            if sub in {"bebop", "membo"}:
                return "drone", "alemadi", sub
            if sub == "drone":
                return "drone", "alemadi", "drone_alemadi"
            return None
        if second == "trident":
            return "drone", "trident", "trident"
        return None

    if top == "background":
        if second == "alemadi":
            return "no_drone", "alemadi_background", None
        if second == "wildlife":
            sub = rel_parts[2] if len(rel_parts) >= 3 else "unknown"
            if sub.startswith("birds"):
                return "no_drone", "wildlife_xenocanto", None
            return "no_drone", "wildlife_other", None
        return None

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-dataset SWARM test manifest.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to SWARM-AUDIO-DATASET root (the folder containing "
             "1_standardized, 2_segments_4s, ...).",
    )
    parser.add_argument(
        "--out",
        default="data/metadata/swarm_test_manifest.csv",
        help="Output manifest CSV path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    seg_root = root / "2_segments_4s"
    if not seg_root.is_dir():
        print(f"ERROR: {seg_root} does not exist", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in seg_root.rglob("*.wav"):
        rel_parts = path.relative_to(seg_root).parts
        info = _classify(rel_parts)
        if info is None:
            continue
        label, source, drone_type = info
        group_id = _extract_group_id(path.name, fallback=path.stem)
        rows.append({
            "filepath":       str(path).replace("\\", "/"),
            "filename":       path.name,
            "label":          label,
            "source_dataset": source,
            "drone_type":     drone_type,
            "group_id":       group_id,
            "split":          "test",
        })

    if not rows:
        print("No matching audio files found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary -----------------------------------------------------------------
    print(f"Wrote {len(df)} rows to {out_path}")
    print()
    print("Per source / per label:")
    print(
        df.groupby(["source_dataset", "label"]).size()
          .unstack(fill_value=0)
    )
    print()
    print("Per drone type:")
    print(df.loc[df["label"] == "drone", "drone_type"].value_counts())
    print()
    print(f"Distinct group_id values: {df['group_id'].nunique()}  "
          f"(used by future GroupShuffleSplit)")


if __name__ == "__main__":
    main()
