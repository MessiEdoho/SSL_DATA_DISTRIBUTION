"""Central configuration: cluster paths, signal constants, and the run registry.

All paths default to the cluster layout but are OVERRIDABLE via environment
variables so the same code runs unchanged locally or on the cluster:

    SSL_DD_SCRATCH           base scratch dir   (default /home/people/22206468/scratch)
    SSL_DD_PARENT_MANIFESTS  parent manifest dir to reuse (D0, D_full, val/test)
    SSL_DD_INPUT             this project's derived-input dir (D_broad, strata)
    SSL_DD_OUTPUT            this project's output dir

Nothing here has side effects (no directory creation) -- import is always safe.
Use ensure_dirs() explicitly when a script is about to write.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# Signal / segmentation constants -- identical to the baseline (do not change).
# --------------------------------------------------------------------------- #
FS: int = 500            # sampling rate (Hz)
WIN_LEN: int = 2500      # window length in samples (5 s @ 500 Hz)
STEP: int = 1250         # hop in samples (2.5 s, 50% overlap)
WIN_SEC: float = WIN_LEN / FS
STEP_SEC: float = STEP / FS

SEED: int = 42           # single seed throughout (see STUDY_REPORT S13)

# M3 selected hyperparameters (num_filters, kernel_size, dropout, fusion) --
# loaded from the baseline tuning output so the SSL encoder is IDENTICAL to R0.
# Override the path via SSL_DD_M3_PARAMS if it lives elsewhere on the cluster.
M3_PARAMS_PATH = os.environ.get("SSL_DD_M3_PARAMS", "")  # resolved in load_m3_hparams()

# --------------------------------------------------------------------------- #
# Base paths (overridable via environment).
# --------------------------------------------------------------------------- #
SCRATCH = Path(os.environ.get("SSL_DD_SCRATCH", "/home/people/22206468/scratch"))

# Parent-project manifests, reused READ-ONLY (identical split to R0).
PARENT_MANIFEST_DIR = Path(os.environ.get(
    "SSL_DD_PARENT_MANIFESTS", str(SCRATCH / "INPUT_DATA" / "data_splits_outputs")))

# This project's derived inputs (written and read back): D_broad, artefact strata.
INPUT_DIR = Path(os.environ.get(
    "SSL_DD_INPUT", str(SCRATCH / "INPUT_SSL_DATA_DISTRIBUTION")))

# This project's outputs.
OUTPUT_DIR = Path(os.environ.get(
    "SSL_DD_OUTPUT", str(SCRATCH / "OUTPUT_SSL_DATA_DISTRIBUTION")))

# --------------------------------------------------------------------------- #
# Manifest filenames.  Parent manifests live in PARENT_MANIFEST_DIR; the D_broad
# manifest we build ourselves lives in INPUT_DIR.
# --------------------------------------------------------------------------- #
MANIFEST_FULL = "data_splits.json"                                  # D_full (raw, full corpus)
MANIFEST_FULL_ENRICHED = "data_splits_full_train_enriched.json"     # D_full (+ chronology)
MANIFEST_D0 = "data_splits_nonictal_sampled.json"                   # D0 (proximity-downsampled)
MANIFEST_VALTEST = "data_splits_nonictal_sampled_filtered_enriched.json"  # val/test (enriched)
MANIFEST_BROAD = "data_splits_broad.json"                           # D_broad (built by us -> INPUT_DIR)

# Distribution -> (directory, filename, partition-key).  partition None = all.
DATA_MANIFESTS = {
    "D0":      (PARENT_MANIFEST_DIR, MANIFEST_D0,           "train"),
    "D_full":  (PARENT_MANIFEST_DIR, MANIFEST_FULL_ENRICHED, "train"),
    "D_broad": (INPUT_DIR,           MANIFEST_BROAD,        "train"),
    "valtest": (PARENT_MANIFEST_DIR, MANIFEST_VALTEST,      None),
}


def manifest_path(distribution: str) -> Path:
    """Absolute path to the manifest backing a data distribution."""
    directory, fname, _ = DATA_MANIFESTS[distribution]
    return directory / fname


def manifest_partition(distribution: str) -> Optional[str]:
    return DATA_MANIFESTS[distribution][2]


# --------------------------------------------------------------------------- #
# Run registry.  R1-R4 are DEVELOPED here; R0 is reused from the parent project.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSpec:
    name: str
    encoder_init: str            # "random" | "ssl"
    ssl_data: Optional[str]      # "D0" | "D_full" | None  (SSL pretraining corpus)
    finetune_data: str           # "D0" | "D_broad"
    protocols: Tuple[str, ...]   # ("frozen", "full") or ("full",)
    developed_here: bool


RUNS = {
    "R0": RunSpec("R0", "random", None,     "D0",      ("full",),          False),
    "R1": RunSpec("R1", "ssl",    "D0",     "D0",      ("frozen", "full"), True),
    "R2": RunSpec("R2", "ssl",    "D_full", "D0",      ("frozen", "full"), True),
    "R3": RunSpec("R3", "ssl",    "D_full", "D_broad", ("frozen", "full"), True),
    "R4": RunSpec("R4", "random", None,     "D_broad", ("full",),          True),
}

# Stage folder names (match the cluster output layout).
STAGE_SSL = "SSL_PRETRAINING"
STAGE_FROZEN = "FROZEN_FINE_TUNING"
STAGE_FULL = "FULL_FINE_TUNING"
STAGE_ANALYSIS = "ANALYSIS"
PROTOCOL_STAGE = {"frozen": STAGE_FROZEN, "full": STAGE_FULL}


def run_stage_dir(run: str, stage: str) -> Path:
    """Output dir for a run x stage, e.g. OUTPUT/R1/FULL_FINE_TUNING."""
    return OUTPUT_DIR / run / stage


def analysis_dir() -> Path:
    return OUTPUT_DIR / STAGE_ANALYSIS


def ssl_shared_dir(ssl_data: str) -> Path:
    """Canonical location of a SHARED SSL-pretrained encoder, keyed by the SSL
    corpus.  R2 and R3 both pretrain on D_full -> the encoder is trained ONCE
    and stored here; each run's own SSL_PRETRAINING/ folder keeps its logs and a
    provenance pointer to this directory (STUDY_REPORT S13)."""
    return OUTPUT_DIR / "SSL_SHARED" / ssl_data


def load_m3_hparams(path: Optional[Path] = None) -> dict:
    """Load M3's selected architecture hyperparameters (num_filters, kernel_size,
    dropout, fusion) so the SSL encoder matches R0 exactly.

    Resolution order: explicit `path` -> SSL_DD_M3_PARAMS -> a few known cluster
    locations.  Raises FileNotFoundError with guidance if none is found.
    """
    import json

    candidates = []
    if path:
        candidates.append(Path(path))
    if M3_PARAMS_PATH:
        candidates.append(Path(M3_PARAMS_PATH))
    # Known baseline tuning-output locations (best-effort).
    candidates += [
        SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCNtuning_outputs" / "best_multiscale_params.json",
        SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "best_multiscale_params.json",
    ]
    for c in candidates:
        if c and Path(c).exists():
            with open(c, "r", encoding="utf-8") as f:
                hp = json.load(f)
            return {
                "num_filters": int(hp["num_filters"]),
                "kernel_size": int(hp["kernel_size"]),
                "dropout": float(hp["dropout"]),
                "fusion": hp.get("fusion", "concat"),
            }
    raise FileNotFoundError(
        "best_multiscale_params.json not found. Pass --m3-params <path> or set "
        "SSL_DD_M3_PARAMS to the baseline tuning output. Tried: "
        + ", ".join(str(c) for c in candidates if c))


def ensure_dirs(*paths: Path) -> None:
    """Create the given directories (parents included).  Call from scripts that
    write; never at import time."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable snapshot of the resolved configuration (for logs)."""
    lines = [
        "Resolved configuration:",
        f"  SCRATCH             = {SCRATCH}",
        f"  PARENT_MANIFEST_DIR = {PARENT_MANIFEST_DIR}",
        f"  INPUT_DIR           = {INPUT_DIR}",
        f"  OUTPUT_DIR          = {OUTPUT_DIR}",
        f"  FS/WIN_LEN/STEP     = {FS} / {WIN_LEN} / {STEP}",
        f"  SEED                = {SEED}",
        "  Runs developed here = " + ", ".join(r for r, s in RUNS.items() if s.developed_here),
    ]
    return "\n".join(lines)
