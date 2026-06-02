from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, strict: bool = True) -> None:
    """Seed Python, NumPy and torch RNGs. Call this at the top of main()."""
    if strict:
        # cuBLAS only picks this up if it's set before the first CUDA op.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[set_seed] could not enable deterministic algorithms: {e}")
