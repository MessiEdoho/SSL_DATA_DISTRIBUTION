"""build_broad_manifest.py -- construct the D_broad fine-tuning manifest ([TBD-4]).

D_broad = D0 (all ictal + all D0 non-ictal) PLUS extra FAR-from-seizure
background, so the classification head and decision see diverse artefacts.

Construction (set-difference; no EDF/distance recomputation needed):
  full_nonictal  = every non-ictal segment in data_splits.json train
  d0_nonictal    = non-ictal segments already in D0
  extra_pool     = full_nonictal - d0_nonictal      (the far background D0 dropped)
  D_broad train  = D0 train  +  a deterministic sample of extra_pool
  D_broad val/test = copied unchanged from the filtered+enriched manifest.

The broadening level is controlled by --target-nonictal-per-ictal R:
  target non-ictal = R * n_ictal ; extra added = target - (D0 non-ictal count).
  [TBD-4] the default R below is PROVISIONAL -- confirm the value before use.

By default m254 (0 seizures; excluded from D0) is also excluded from the extra
pool, so D_broad broadens BACKGROUND within the SAME 71 training mice as D0/R0.

CPU-only; memory-heavy (loads the full ~28.7M-record manifest).  Writes to
INPUT_SSL_DATA_DISTRIBUTION/data_splits_broad.json.

Usage:
    python scripts/build_broad_manifest.py --target-nonictal-per-ictal 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as cfg                 # noqa: E402
from src.common.logging_utils import setup_logging    # noqa: E402
from src.data.datasets import load_records            # noqa: E402

# [TBD-4] PROVISIONAL default broadening level (non-ictal per ictal). D0 = 2.37.
DEFAULT_TARGET_RATIO = 6.0
DEFAULT_EXCLUDE_MOUSE = "m254"


def _mouse_of(rec: dict) -> str:
    mid = rec.get("mouse_id")
    if mid:
        return str(mid)
    stem = Path(rec["filepath"]).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-manifest", type=Path,
                    default=cfg.PARENT_MANIFEST_DIR / cfg.MANIFEST_FULL)
    ap.add_argument("--d0-manifest", type=Path,
                    default=cfg.PARENT_MANIFEST_DIR / cfg.MANIFEST_D0)
    ap.add_argument("--valtest-manifest", type=Path,
                    default=cfg.PARENT_MANIFEST_DIR / cfg.MANIFEST_VALTEST)
    ap.add_argument("--out", type=Path,
                    default=cfg.INPUT_DIR / cfg.MANIFEST_BROAD)
    ap.add_argument("--target-nonictal-per-ictal", type=float,
                    default=DEFAULT_TARGET_RATIO,
                    help="R: target non-ictal:ictal ratio for D_broad [TBD-4].")
    ap.add_argument("--exclude-mouse", default=DEFAULT_EXCLUDE_MOUSE,
                    help="Mouse id excluded from the extra pool (match D0).")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    cfg.ensure_dirs(args.out.parent, cfg.analysis_dir() / "build_broad")
    logger = setup_logging("build_broad_manifest",
                           cfg.analysis_dir() / "build_broad" / "build_broad_manifest.log")
    logger.info("=" * 70)
    logger.info("build_broad_manifest.py | %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("Target non-ictal:ictal ratio R = %.2f (D0 is ~2.37) [TBD-4]",
                args.target_nonictal_per_ictal)

    # ---- D0 train (kept in full) --------------------------------------- #
    d0 = load_records(args.d0_manifest, "train")
    d0_ictal = [r for r in d0 if int(r.get("label", 0)) == 1]
    d0_nonictal = [r for r in d0 if int(r.get("label", 0)) == 0]
    d0_nonictal_paths = {r["filepath"] for r in d0_nonictal}
    n_ictal = len(d0_ictal)
    logger.info("D0 train: %d ictal, %d non-ictal", n_ictal, len(d0_nonictal))

    # ---- full non-ictal pool minus D0, minus excluded mouse ------------ #
    logger.info("Loading full manifest (large; ~1 min)...")
    full = load_records(args.full_manifest, "train")
    extra_pool = [
        r for r in full
        if int(r.get("label", 0)) == 0
        and r["filepath"] not in d0_nonictal_paths
        and _mouse_of(r) != args.exclude_mouse
    ]
    del full
    logger.info("Extra far-background pool (full non-ictal - D0 - %s): %d",
                args.exclude_mouse, len(extra_pool))

    # ---- how many extra to add ----------------------------------------- #
    target_nonictal = int(round(args.target_nonictal_per_ictal * n_ictal))
    n_extra = max(0, target_nonictal - len(d0_nonictal))
    if n_extra > len(extra_pool):
        logger.warning("Requested %d extra but pool has only %d; capping.",
                       n_extra, len(extra_pool))
        n_extra = len(extra_pool)
    rng = random.Random(args.seed)
    extra = rng.sample(extra_pool, n_extra) if n_extra else []
    logger.info("Adding %d extra far-background segments (seed %d).", n_extra, args.seed)

    broad_train = d0_ictal + d0_nonictal + extra
    rng.shuffle(broad_train)
    n_nonictal_final = len(d0_nonictal) + len(extra)
    ratio = n_nonictal_final / max(1, n_ictal)
    logger.info("D_broad train: %d ictal, %d non-ictal (ratio 1:%.2f), %d total",
                n_ictal, n_nonictal_final, ratio, len(broad_train))

    # ---- copy val/test unchanged from the filtered+enriched manifest --- #
    logger.info("Copying val/test from %s", args.valtest_manifest.name)
    val = load_records(args.valtest_manifest, "val")
    test = load_records(args.valtest_manifest, "test")

    out = {
        "metadata": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "description": "D_broad: D0 + extra far-from-seizure background",
            "target_nonictal_per_ictal": args.target_nonictal_per_ictal,
            "n_ictal": n_ictal,
            "n_nonictal": n_nonictal_final,
            "n_extra_added": len(extra),
            "final_ratio_ictal_to_nonictal": round(ratio, 4),
            "excluded_mouse": args.exclude_mouse,
            "seed": args.seed,
            "sources": {
                "full": str(args.full_manifest),
                "d0": str(args.d0_manifest),
                "valtest": str(args.valtest_manifest),
            },
            "tbd": "TBD-4 (broadening level) -- confirm R before final runs.",
        },
        "train": broad_train,
        "val": val,
        "test": test,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    logger.info("Wrote D_broad manifest: %s", args.out)
    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
