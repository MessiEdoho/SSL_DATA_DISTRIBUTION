# Vendored baseline modules

These files are **verbatim pinned copies** from the parent project
`DL_WITH_SSL_GA`, vendored so this repo is self-contained and the encoder /
evaluation code cannot drift from the R0 baseline.

| File | Source | Purpose here |
|------|--------|--------------|
| `tcn_utils.py` | `DL_WITH_SSL_GA/tcn_utils.py` | `MultiScaleTCN` encoder (M3), `EEGSegmentDataset`, `set_seed`, train/eval helpers |
| `eval_utils.py` | `DL_WITH_SSL_GA/eval_utils.py` | segment + event metrics, post-processing, `write_event_level_bundle`, chronology |

## Pinned source

- **Commit:** `f4280a7894676e2e8e781bd821689ced9dc04f9e`
- **Short:** `f4280a7` — "Updated report_study. Added new scripts replotting"
- **Committed:** 2026-07-07 17:47:15 +0100
- **Vendored on:** 2026-07-25

## Rules

- Do **not** edit these files. If a baseline fix is needed, re-vendor from the
  parent at a new commit and update this file.
- We import only stable entry points (`MultiScaleTCN`, `EEGSegmentDataset`,
  `set_seed`, the metric / post-processing functions). Any hard-coded cluster
  paths inside these modules are irrelevant because we import classes/functions,
  not run their `__main__`.
