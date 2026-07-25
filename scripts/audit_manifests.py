"""audit_manifests.py -- Phase 0 data audit (resolves [TBD-8] and the D_full check).

Reads the reused parent manifests and reports, per manifest and partition:
  * segment counts (total / seizure / non-seizure) and ictal %;
  * unique mouse (subject) counts;
  * whether records carry chronology fields (chrono_idx, t_start_sec);
  * a subject-level LEAKAGE CHECK (no mouse in more than one partition).

It also answers the D_full question: how many TRAIN segments the full manifests
actually enumerate (so we know whether the SSL loader can read the manifest
directly, or must scan the non_seizure/ folders).

Runs offline / CPU-only.  No torch needed.  Writes a JSON report + a log to
OUTPUT_SSL_DATA_DISTRIBUTION/ANALYSIS/audit/.

Usage:
    python scripts/audit_manifests.py
    python scripts/audit_manifests.py --scan-folders   # also count .npy on disk (slow)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable (scripts/ -> repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as cfg              # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402

PARTITION_KEYS = ("train", "val", "test")


def _mouse_id(rec: dict) -> str:
    """Subject id: prefer the manifest field, else parse the filename (baseline
    convention: substring before the first underscore)."""
    mid = rec.get("mouse_id")
    if mid:
        return str(mid)
    fp = rec.get("filepath") or rec.get("filename") or ""
    stem = Path(fp).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def audit_partition(records: list) -> dict:
    n_total = len(records)
    n_seiz = sum(1 for r in records if int(r.get("label", 0)) == 1)
    n_non = n_total - n_seiz
    mice = sorted({_mouse_id(r) for r in records})
    sample = records[:50]
    enriched = bool(sample) and all(
        ("chrono_idx" in r and "t_start_sec" in r) for r in sample)
    return {
        "n_total": n_total,
        "n_seizure": n_seiz,
        "n_non_seizure": n_non,
        "ictal_pct": round(100.0 * n_seiz / n_total, 4) if n_total else 0.0,
        "n_mice": len(mice),
        "mice": mice,
        "enriched_chronology": enriched,
    }


def leakage_check(part_mice: dict, logger) -> dict:
    """Return pairwise mouse overlaps between partitions; log any leakage."""
    result, leak = {}, False
    keys = [k for k in PARTITION_KEYS if k in part_mice]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = sorted(set(part_mice[a]) & set(part_mice[b]))
            result[f"{a}__{b}"] = overlap
            if overlap:
                leak = True
                logger.error("LEAKAGE: mice in BOTH %s and %s: %s", a, b, overlap)
    if not leak:
        logger.info("Leakage check PASSED (no mouse shared across partitions).")
    result["leakage_detected"] = leak
    return result


def audit_manifest(path: Path, logger) -> dict:
    logger.info("-" * 70)
    logger.info("Manifest: %s", path)
    if not path.exists():
        logger.warning("  MISSING -- skipping.")
        return {"path": str(path), "present": False}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = {"path": str(path), "present": True, "partitions": {}}
    if isinstance(data.get("metadata"), dict):
        report["metadata"] = data["metadata"]

    part_mice = {}
    for key in PARTITION_KEYS:
        recs = data.get(key)
        if not isinstance(recs, list) or not recs:
            continue
        pa = audit_partition(recs)
        part_mice[key] = pa["mice"]
        # Do not dump the full mouse list into the log; keep the report tidy.
        logged = {k: v for k, v in pa.items() if k != "mice"}
        logger.info("  [%s] %s", key.upper(), logged)
        report["partitions"][key] = pa

    report["leakage"] = leakage_check(part_mice, logger)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, default=cfg.PARENT_MANIFEST_DIR,
                    help="Directory holding the reused parent manifests.")
    ap.add_argument("--out-dir", type=Path, default=cfg.analysis_dir() / "audit",
                    help="Where to write the report + log.")
    ap.add_argument("--scan-folders", action="store_true",
                    help="Also count .npy files physically on disk (slow; needs "
                         "read access to the TRAIN/VAL/TEST folders).")
    args = ap.parse_args()

    cfg.ensure_dirs(args.out_dir)
    logger = setup_logging("audit_manifests", args.out_dir / "audit_manifests.log")

    logger.info("=" * 70)
    logger.info("audit_manifests.py  |  %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("=" * 70)
    logger.info(cfg.summary())

    manifests = {
        "D_full_raw":        args.manifest_dir / cfg.MANIFEST_FULL,
        "D_full_enriched":   args.manifest_dir / cfg.MANIFEST_FULL_ENRICHED,
        "D0":                args.manifest_dir / cfg.MANIFEST_D0,
        "valtest_enriched":  args.manifest_dir / cfg.MANIFEST_VALTEST,
    }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest_dir": str(args.manifest_dir),
        "manifests": {},
    }
    for label, path in manifests.items():
        report["manifests"][label] = audit_manifest(path, logger)

    # ---- D_full enumeration answer (the [TBD-8] question) ---------------- #
    logger.info("=" * 70)
    logger.info("D_full ENUMERATION CHECK")
    for label in ("D_full_raw", "D_full_enriched"):
        m = report["manifests"].get(label, {})
        train = m.get("partitions", {}).get("train") if m.get("present") else None
        if train:
            logger.info("  %s train: %d segments (%d seizure / %d non-seizure), %d mice",
                        label, train["n_total"], train["n_seizure"],
                        train["n_non_seizure"], train["n_mice"])
        else:
            logger.info("  %s: no train partition found.", label)
    logger.info("  -> If these counts approach ~28.7M, the manifest enumerates "
                "the full corpus and the SSL loader can read it directly; if far "
                "smaller, D_full must be enumerated by scanning non_seizure/.")

    # ---- Optional physical folder scan --------------------------------- #
    if args.scan_folders:
        logger.info("=" * 70)
        logger.info("PHYSICAL FOLDER SCAN (this can be slow)")
        scan = {}
        train_roots = [cfg.SCRATCH / f"TRAIN_DATA{'' if i == 1 else f'_{i}'}"
                       for i in range(1, 6)]
        for root in train_roots + [cfg.SCRATCH / "VAL_DATA", cfg.SCRATCH / "TEST_DATA"]:
            entry = {}
            for cls in ("seizure", "non_seizure"):
                d = root / cls
                entry[cls] = sum(1 for _ in d.glob("*.npy")) if d.exists() else None
            scan[str(root)] = entry
            logger.info("  %s : %s", root.name, entry)
        report["folder_scan"] = scan

    out_json = args.out_dir / "audit_manifests_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("=" * 70)
    logger.info("Report written: %s", out_json)

    any_leak = any(
        m.get("leakage", {}).get("leakage_detected")
        for m in report["manifests"].values() if m.get("present"))
    logger.info("DONE. Leakage detected: %s", any_leak)
    return 1 if any_leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
