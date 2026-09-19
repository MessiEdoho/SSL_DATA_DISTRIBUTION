"""pretrain_ssl.py -- self-supervised masked-autoencoder pretraining.

Trains the M3 encoder by masked-span reconstruction for a FIXED number of steps
(held constant across R1/R2/R3 so R1->R2 isolates data, not compute).  Keyed by
the SSL corpus (D0 or D_full); the D_full encoder is trained ONCE and shared by
R2 and R3.

Checkpoint / resume / logging follow the baseline conventions (STUDY_REPORT S13),
adapted to be STEP-based and to track BEST = lowest validation reconstruction
loss.  Re-submitting the same Slurm job auto-resumes from latest.pt.

Usage:
    python scripts/pretrain_ssl.py --ssl-data D0    --m3-params <best_params.json>
    python scripts/pretrain_ssl.py --ssl-data D_full --m3-params <best_params.json>
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as cfg                     # noqa: E402
from src.common.logging_utils import setup_logging        # noqa: E402
from src.common.seed import set_seed                      # noqa: E402
from src.data.datasets import SSLWindowDataset, load_filepaths  # noqa: E402
from src.models.masked_autoencoder import MaskedAutoencoder     # noqa: E402

PREFIX = "ssl_mae"


def build_scheduler(optimiser, warmup: int, total: int):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def save_ckpt(path, step, model, optimiser, scheduler, val_recon,
              best_val_recon, best_step, hparams, logger):
    try:
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_recon": val_recon,
            "best_val_recon": best_val_recon,
            "best_step": best_step,
            "hyperparameters": hparams,
            "timestamp": datetime.datetime.now().isoformat(),
        }, path)
        logger.debug("Checkpoint saved: %s", path)
    except Exception as exc:  # non-fatal, per baseline
        logger.error("Checkpoint save failed (%s): %s", path, exc)


@torch.no_grad()
def eval_val_recon(model, val_loader, device, seed):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)  # fixed masking for a stable metric
    total, n = 0.0, 0
    for x in val_loader:
        x = x.to(device, non_blocking=True)
        recon, mask, target = model(x, generator=gen)
        total += float(model.masked_mse(recon, target, mask)) * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssl-data", required=True, choices=["D0", "D_full"],
                    help="SSL pretraining corpus (D0 for R1; D_full for R2/R3).")
    ap.add_argument("--m3-params", type=Path, default=None,
                    help="Path to best_multiscale_params.json (else auto-resolve).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: shared SSL dir for this corpus).")
    # Approved SSL knobs (defaults = STUDY_REPORT S5.3).
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--span-len", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=5_000, help="val windows for the recon monitor")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1_000)
    ap.add_argument("--num-workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or cfg.ssl_shared_dir(args.ssl_data)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    cfg.ensure_dirs(out_dir, ckpt_dir, log_dir)
    logger = setup_logging("pretrain_ssl", log_dir / "pretrain_ssl.log")

    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("pretrain_ssl.py | ssl_data=%s | %s", args.ssl_data,
                datetime.datetime.now().isoformat(timespec="seconds"))
    logger.info("device=%s | steps=%d warmup=%d batch=%d lr=%.1e wd=%.3f "
                "mask=%.2f span=%d", device, args.steps, args.warmup,
                args.batch_size, args.lr, args.weight_decay, args.mask_ratio,
                args.span_len)
    logger.info(cfg.summary())

    # ---- model ---------------------------------------------------------- #
    hparams = cfg.load_m3_hparams(args.m3_params)
    logger.info("M3 hyperparameters (matched to R0): %s", hparams)
    model = MaskedAutoencoder(hparams, args.mask_ratio, args.span_len).to(device)

    # ---- data ----------------------------------------------------------- #
    train_manifest = cfg.manifest_path(args.ssl_data)
    logger.info("Loading SSL train filepaths from %s ...", train_manifest)
    train_fps = load_filepaths(train_manifest, "train", labels=None)  # full distribution
    logger.info("SSL train windows: %d", len(train_fps))
    train_ds = SSLWindowDataset(train_fps)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)

    val_manifest = cfg.manifest_path("valtest")
    val_fps = load_filepaths(val_manifest, "val", labels=None,
                             max_n=args.val_n, seed=cfg.SEED)
    logger.info("Val recon-monitor windows: %d", len(val_fps))
    val_loader = DataLoader(SSLWindowDataset(val_fps), batch_size=args.batch_size,
                            shuffle=False, num_workers=min(4, args.num_workers))

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimiser, args.warmup, args.steps)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- resume --------------------------------------------------------- #
    latest = ckpt_dir / f"{PREFIX}_latest.pt"
    best_ckpt = ckpt_dir / f"{PREFIX}_best.pt"
    start_step, best_val_recon, best_step = 0, float("inf"), -1
    if latest.exists() or best_ckpt.exists():
        resume = latest if latest.exists() else best_ckpt
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimiser.load_state_dict(ck["optimiser_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_step = ck["step"] + 1
        best_val_recon = ck.get("best_val_recon", float("inf"))
        best_step = ck.get("best_step", -1)
        logger.info("RESUMED from %s at step %d (best_val_recon=%.6f @ %d)",
                    resume.name, ck["step"], best_val_recon, best_step)
    else:
        logger.info("No checkpoint found. Starting fresh.")

    # ---- train loop (fixed steps) -------------------------------------- #
    model.train()
    batches = infinite_batches(train_loader)
    t0 = datetime.datetime.now()
    running = 0.0
    for step in range(start_step, args.steps):
        x = next(batches).to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            recon, mask, target = model(x)
            loss = model.masked_mse(recon, target, mask)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        running += float(loss)

        if (step + 1) % args.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            logger.info("step %6d/%d | loss=%.6f | lr=%.2e | %.0fs",
                        step + 1, args.steps, running / args.log_every, lr_now,
                        (datetime.datetime.now() - t0).total_seconds())
            running = 0.0

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            val_recon = eval_val_recon(model, val_loader, device, cfg.SEED)
            improved = val_recon < best_val_recon
            if improved:
                best_val_recon, best_step = val_recon, step + 1
                save_ckpt(best_ckpt, step, model, optimiser, scheduler, val_recon,
                          best_val_recon, best_step, hparams, logger)
                torch.save(model.encoder_state_dict(), out_dir / f"{PREFIX}_encoder_best.pt")
            save_ckpt(latest, step, model, optimiser, scheduler, val_recon,
                      best_val_recon, best_step, hparams, logger)
            logger.info("  [val] recon=%.6f | best=%.6f @ %d%s",
                        val_recon, best_val_recon, best_step,
                        "  <- new best" if improved else "")

    # ---- final encoder weights (drop-in for fine-tuning) --------------- #
    torch.save(model.encoder_state_dict(), out_dir / f"{PREFIX}_encoder_final.pt")
    logger.info("TRAINING COMPLETE. best_val_recon=%.6f @ step %d",
                best_val_recon, best_step)
    logger.info("Encoder weights: %s (best), %s (final)",
                out_dir / f"{PREFIX}_encoder_best.pt", out_dir / f"{PREFIX}_encoder_final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
