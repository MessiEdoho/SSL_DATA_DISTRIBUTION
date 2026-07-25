"""Logging setup -- mirrors DL_WITH_SSL_GA setup_logging.

Every script logs to BOTH a file (DEBUG, append) and stdout (INFO), with the
baseline format.  One log file per script, appended across resumes; the Slurm
%j files are the per-run copies.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str, log_file: Path) -> logging.Logger:
    """Return a logger writing DEBUG->file (append) and INFO->stdout.

    name      : logger name (e.g. "audit_manifests").
    log_file  : path to the .log file; its parent dir is created.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on resume / re-entry

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
