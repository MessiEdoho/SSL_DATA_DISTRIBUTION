"""Datasets and manifest loading.

Two consumers:
  * SSL pretraining  -> SSLWindowDataset: unlabeled 5 s windows, sanitised;
    masking is applied downstream by the masked autoencoder, not here.
  * Supervised fine-tuning -> the vendored EEGSegmentDataset (labels).

Manifest facts (audit 2026-07-25): the full D_full manifest enumerates all
28,752,220 train segments individually, so we read it directly. Loading that
JSON is heavy (~GB, ~1 min); we parse once, keep only the fields we need, and
drop the raw structure to bound memory.  Optional deterministic subsampling
([TBD-3]) is provided for tractable SSL runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # torch is required for the datasets, not for loaders
    torch = None
    Dataset = object  # type: ignore

from src.common.config import SEED, WIN_LEN


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #
def load_records(manifest_path: Path, partition: str) -> List[dict]:
    """Load one partition's records from a split manifest JSON.

    Returns the raw record dicts (filepath, label, mouse_id, and possibly
    chrono_idx / t_start_sec).  For huge partitions prefer load_filepaths().
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get(partition)
    if not isinstance(recs, list):
        raise KeyError(f"Partition '{partition}' not found in {manifest_path}")
    return recs


def load_filepaths(
    manifest_path: Path,
    partition: str,
    labels: Optional[Tuple[int, ...]] = None,
    subsample: Optional[float] = None,
    max_n: Optional[int] = None,
    seed: int = SEED,
    return_mouse: bool = False,
):
    """Memory-lean extraction of segment paths from a manifest partition.

    labels    : keep only these label values (None = all -> full distribution).
    subsample : keep this fraction (0<f<=1), deterministic.
    max_n     : hard cap on the number returned (applied after subsample).
    return_mouse : also return a parallel list of mouse_ids.

    Returns list[str] of filepaths, or (filepaths, mouse_ids) if return_mouse.
    """
    recs = load_records(manifest_path, partition)
    fps: List[str] = []
    mice: List[str] = []
    for r in recs:
        if labels is not None and int(r.get("label", 0)) not in labels:
            continue
        fps.append(r["filepath"])
        if return_mouse:
            mid = r.get("mouse_id")
            if not mid:
                stem = Path(r["filepath"]).stem
                mid = stem.split("_", 1)[0] if "_" in stem else stem
            mice.append(str(mid))
    del recs  # free the parsed structure

    if subsample is not None and 0 < subsample < 1:
        rng = random.Random(seed)
        idx = list(range(len(fps)))
        rng.shuffle(idx)
        keep = set(idx[: int(round(len(fps) * subsample))])
        fps = [fps[i] for i in range(len(fps)) if i in keep]
        if return_mouse:
            mice = [mice[i] for i in range(len(mice)) if i in keep]

    if max_n is not None and len(fps) > max_n:
        rng = random.Random(seed + 1)
        idx = sorted(rng.sample(range(len(fps)), max_n))
        fps = [fps[i] for i in idx]
        if return_mouse:
            mice = [mice[i] for i in idx]

    return (fps, mice) if return_mouse else fps


# --------------------------------------------------------------------------- #
# SSL dataset (unlabeled)
# --------------------------------------------------------------------------- #
class SSLWindowDataset(Dataset):
    """Unlabeled 5 s windows for self-supervised pretraining.

    Returns a float32 tensor of shape (1, WIN_LEN).  Segments are already robust
    z-scored offline (baseline preprocessing); we additionally SANITISE against
    the rare preprocessing-corrupted segments in the un-filtered full corpus
    (non-finite -> 0, |x| clipped to `clip`) so a stray |x|~1e17 cannot turn the
    reconstruction loss into NaN.
    """

    def __init__(self, filepaths: List[str], clip: float = 20.0, sanitise: bool = True):
        if torch is None:
            raise ImportError("torch is required for SSLWindowDataset")
        self.filepaths = filepaths
        self.clip = float(clip)
        self.sanitise = sanitise

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, i: int):
        x = np.load(self.filepaths[i]).astype(np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        if self.sanitise:
            x = np.nan_to_num(x, nan=0.0, posinf=self.clip, neginf=-self.clip)
            np.clip(x, -self.clip, self.clip, out=x)
        if x.shape[0] != WIN_LEN:  # defensive: enforce expected length
            x = _fix_length(x, WIN_LEN)
        return torch.from_numpy(x).unsqueeze(0)  # (1, WIN_LEN)


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] > n:
        return x[:n]
    return np.pad(x, (0, n - x.shape[0]))


# --------------------------------------------------------------------------- #
# Supervised dataset (labeled) -- reuse the vendored baseline dataset
# --------------------------------------------------------------------------- #
def make_supervised_dataset(pairs: List[Tuple[str, int]]):
    """Build the vendored EEGSegmentDataset from (filepath, label) pairs, so
    fine-tuning uses the exact same loading path as the R0 baseline."""
    from src.vendor.tcn_utils import EEGSegmentDataset
    return EEGSegmentDataset(pairs)


def load_supervised_pairs(manifest_path: Path, partition: str) -> List[Tuple[str, int]]:
    """(filepath, label) pairs for supervised training/eval."""
    recs = load_records(manifest_path, partition)
    return [(r["filepath"], int(r["label"])) for r in recs]
