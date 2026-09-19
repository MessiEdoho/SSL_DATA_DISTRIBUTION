"""Masked EEG autoencoder ([TBD-1,2,3], approved 2026-07-25).

  * Encoder: the vendored M3 MultiScaleTCN, tapped at the PRE-POOL feature map
    (B, num_filters, T) -- built with the same hyperparameters as R0 so the
    encoder is identical.  Classification still uses the pooled vector; SSL uses
    the time-resolved features.
  * Masking: contiguous SPAN masking (ratio 0.5, span 50 samples), masked spans
    zeroed in the encoder input.
  * Decoder: 2x Conv1d (k=3, GELU): F -> F -> 1; discarded after pretraining.
  * Loss: MSE on MASKED positions only (time domain).

The trained encoder's state_dict is a drop-in for a fine-tuning MultiScaleTCN
(the branch/fusion keys match; the unused classifier is loaded non-strict).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vendor.tcn_utils import MultiScaleTCN


# --------------------------------------------------------------------------- #
# Encoder construction + pre-pool tap
# --------------------------------------------------------------------------- #
def build_encoder(hparams: dict) -> MultiScaleTCN:
    """Instantiate the M3 encoder with the baseline-selected hyperparameters."""
    return MultiScaleTCN(
        num_filters=hparams["num_filters"],
        kernel_size=hparams["kernel_size"],
        dropout=hparams["dropout"],
        fusion=hparams.get("fusion", "concat"),
        return_embedding=False,  # irrelevant; we call branches directly for SSL
    )


def encode_features(encoder: MultiScaleTCN, x: torch.Tensor) -> torch.Tensor:
    """Return the PRE-POOL fused feature map (B, num_filters, T), replicating
    MultiScaleTCN.forward up to (but not including) global average pooling.
    Reuses the vendored branch/fusion modules without editing them."""
    out1 = encoder.branch1(x)
    out2 = encoder.branch2(x)
    out3 = encoder.branch3(x)
    if encoder.fusion == "concat":
        fused = encoder.fusion_conv(torch.cat([out1, out2, out3], dim=1))
    else:
        fused = (out1 + out2 + out3) / 3.0
    return fused  # (B, num_filters, T)


# --------------------------------------------------------------------------- #
# Span masking
# --------------------------------------------------------------------------- #
def generate_span_mask(batch: int, length: int, mask_ratio: float, span_len: int,
                       device, generator: torch.Generator = None) -> torch.Tensor:
    """Boolean mask (B, length), True where MASKED.  Contiguous spans of
    `span_len`; total masked ~= mask_ratio * length (spans may overlap, which
    slightly reduces the realised ratio -- acceptable and standard)."""
    n_spans = max(1, round(mask_ratio * length / span_len))
    max_start = max(1, length - span_len)
    starts = torch.randint(0, max_start, (batch, n_spans), device=device,
                           generator=generator)                     # (B, n_spans)
    offsets = torch.arange(span_len, device=device)                 # (span_len,)
    idx = (starts.unsqueeze(-1) + offsets).clamp_(0, length - 1)    # (B, n_spans, span_len)
    idx = idx.reshape(batch, -1)                                    # (B, n_spans*span_len)
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    return mask


# --------------------------------------------------------------------------- #
# Decoder + full model
# --------------------------------------------------------------------------- #
class ConvDecoder(nn.Module):
    """Lightweight reconstruction head: (B, F, T) -> (B, 1, T)."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(num_filters, 1, kernel_size=3, padding=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)  # (B, 1, T)


class MaskedAutoencoder(nn.Module):
    """M3 encoder + conv decoder trained by masked-span reconstruction."""

    def __init__(self, hparams: dict, mask_ratio: float = 0.5, span_len: int = 50):
        super().__init__()
        self.encoder = build_encoder(hparams)
        self.decoder = ConvDecoder(hparams["num_filters"])
        self.mask_ratio = float(mask_ratio)
        self.span_len = int(span_len)

    def forward(self, x: torch.Tensor, generator: torch.Generator = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 1, T). Returns (reconstruction, mask, target=x)."""
        B, _, T = x.shape
        mask = generate_span_mask(B, T, self.mask_ratio, self.span_len,
                                  x.device, generator)              # (B, T)
        x_masked = x.masked_fill(mask.unsqueeze(1), 0.0)            # zero masked spans
        feats = encode_features(self.encoder, x_masked)            # (B, F, T)
        recon = self.decoder(feats)                                # (B, 1, T)
        return recon, mask, x

    @staticmethod
    def masked_mse(recon: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """MSE over MASKED positions only (MAE convention)."""
        r = recon[:, 0, :]
        t = target[:, 0, :]
        sel = mask  # (B, T) bool
        if sel.any():
            return F.mse_loss(r[sel], t[sel])
        return F.mse_loss(r, t)  # degenerate fallback

    def encoder_state_dict(self):
        """State dict of the M3 encoder alone (drop-in for fine-tuning)."""
        return self.encoder.state_dict()
