"""Deterministic seeding -- mirrors DL_WITH_SSL_GA/tcn_utils.py:set_seed.

Call set_seed(SEED) at the top of every main() AND immediately before model
weight initialisation, so every run starts from identical parameters.

Note on resume: as in the baseline, RNG state is NOT checkpointed, so a resumed
run is not bit-identical to an uninterrupted one (the difference is negligible;
early stopping dominates). See STUDY_REPORT S13.
"""
from __future__ import annotations

import random

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # torch not importable in a lightweight (e.g. audit) context
    _HAVE_TORCH = False

from src.common.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy, and (if available) torch CPU+CUDA; force determinism."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAVE_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derived_rng(offset: int = 0) -> random.Random:
    """A fixed, independent RNG stream (e.g. for a val-monitor subset), matching
    the baseline's `random.Random(SEED + k)` pattern."""
    return random.Random(SEED + offset)
