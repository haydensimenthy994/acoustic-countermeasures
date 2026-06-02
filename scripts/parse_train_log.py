"""Scrape per-epoch metrics out of a train_*.log into JSON for fig 1."""
import argparse, json, re
from pathlib import Path


# Format example:
#   Epoch 1/10 | Train Loss: 0.5867 | Train Acc: 0.6658 | Val Loss: 0.5329 | Val Acc: 0.7751 | Time: 128.8s
PATTERN_EPOCH = re.compile(
    r"Epoch\s+(\d+)\s*/\s*\d+\s*\|\s*"
    r"Train\s+Loss:\s*([0-9.]+)\s*\|\s*"
    r"Train\s+Acc:\s*([0-9.]+)\s*\|\s*"
    r"Val\s+Loss:\s*([0-9.]+)\s*\|\s*"
    r"Val\s+Acc:\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_log(path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = PATTERN_EPOCH.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            train_acc.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))
            val_acc.append(float(m.group(5)))
    if not epochs:
        raise ValueError(
            f"No epoch summary lines parsed from {path}. "
            "Expected lines like 'Epoch 1/10 | Train Loss: ... | Val Acc: ...'"
        )
    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    return {
        "epochs":      epochs,
        "train_loss":  train_loss,
        "train_acc":   train_acc,
        "val_loss":    val_loss,
        "val_acc":     val_acc,
        "best_val_acc": val_acc[best_idx],
        "best_epoch":   epochs[best_idx],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",  required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--name", required=True, help="cnn14 or scratch")
    args = ap.parse_args()

    data = parse_log(args.log)
    data["run_name"] = args.name
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"parsed {len(data['epochs'])} epochs  ->  {args.out}")
    print(f"best val_acc = {data['best_val_acc']:.4f} at epoch {data['best_epoch']}")


if __name__ == "__main__":
    main()