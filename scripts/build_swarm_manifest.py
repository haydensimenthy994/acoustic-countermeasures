"""Build the cross-dataset test manifest from SWARM-AUDIO-DATASET.

The trained models have never seen any of these clips, so this gives an
honest generalisation read without retraining. Walks only `2_segments_4s/`
since those drop straight into DroneAudioDataset (5 s target = 1 s of pad,
no truncation). `group_id` is preserved for any future GroupShuffleSplit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Each sub-corpus has its own filename convention; pull the parent-recording
# id out so chunks of the same source share a group_id. Examples:
#   Alemadi drone:       B_S2_D1_067-bebop_000__0.wav     -> B_S2_D1_067
#   Trident drone:       trident_d1_0_seg1.wav            -> trident_d1_0
#   ESC-50 background:   1-100032-A-00.wav                -> 100032
#   Xeno-Canto:          XC1015738 - House Sparrow_0.wav  -> XC1015738
#   Other wildlife:      doing_the_dishes000_0.wav        -> doing_the_dishes000

ALEMADI_DRONE_RE  = re.compile(r"^(B_S\d+_D\d+_\d+)-")
TRIDENT_DRONE_RE  = re.compile(r"^(trident_d\d+_\d+)_")
ESC50_BG_RE       = re.compile(r"^\d+-(\d+)-")
XENO_CANTO_RE     = re.compile(r"^(XC\d+)")


def _extract_group_id(filename: str, fallback: str) -> str:
    """Best-effort parent-recording id from a filename."""
    for pattern in (ALEMADI_DRONE_RE, TRIDENT_DRONE_RE, XENO_CANTO_RE):
        m = pattern.match(filename)
        if m:
            return m.group(1)

    # ESC-50 layout: <fold>-<clip>-<take>-<class>
    m = ESC50_BG_RE.match(filename)
    if m:
        return m.group(1)

    # Generic: strip a trailing _<digit>.
    stem = Path(filename).stem
    return re.sub(r"_\d+$", "", stem)


def _classify(rel_parts: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    """Path parts → (label, source_dataset, drone_type), or None to skip."""
    if len(rel_parts) < 2:
        return None

    top = rel_parts[0]
    second = rel_parts[1]

    if top == "drones":
        if second == "alemadi" and len(rel_parts) >= 3:
            sub = rel_parts[2]
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
    print(f"Distinct group_id values: {df['group_id'].nunique()}")


if __name__ == "__main__":
    main()
