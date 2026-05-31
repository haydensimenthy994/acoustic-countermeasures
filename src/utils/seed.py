from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, strict: bool = True) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs.

    With `strict=True` (default), additionally:
      - sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` so cuBLAS GEMM is reproducible,
      - enables `torch.use_deterministic_algorithms(True, warn_only=True)`
        so any non-deterministic op falls back to a deterministic kernel
        when one exists and only warns when one does not.

    Note: `CUBLAS_WORKSPACE_CONFIG` must be set before any CUDA op runs.
    Calling `set_seed` at the top of `main()` (before model loading) is
    enough; calling it after a CUDA op has already executed has no effect
    on cuBLAS determinism for that op.

    Without strict mode, behaviour matches the original implementation
    (deterministic cuDNN only) — kept for reverse compatibility with
    code that prefers performance over reproducibility.
    """
    if strict:
        # Must be set before the first CUDA op for cuBLAS to honour it.
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
            # Older torch versions or some ops may not support this.
            print(f"[set_seed] could not enable deterministic algorithms: {e}")
