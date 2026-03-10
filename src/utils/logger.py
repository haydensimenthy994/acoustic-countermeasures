from __future__ import annotations

from pathlib import Path
from datetime import datetime


def get_log_file(log_dir: str = "outputs/logs", prefix: str = "train") -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{prefix}_{timestamp}.log"


def log_line(message: str, logfile: Path | None = None) -> None:
    print(message)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(message + "\n")