# ACE Tennis Vision — Training Pipeline

CTR-GCN + AimCLR + MAML training pipeline for the ACE tennis iOS app.

## Overview

```
Phase 0  Repository setup (this file)
Phase 1  Data collection & annotation (manual)
Phase 2  30fps skeleton extraction
Phase 3  Dataset construction (45-frame windows)
Phase 4  AimCLR self-supervised pretraining
Phase 5  CTR-GCN supervised fine-tuning
Phase 6  Export: PyTorch → ONNX → CoreML INT8
```

## Directory Layout

```
data/
  raw_videos/        Raw .mov files (gitignored, managed locally)
  skeletons_30fps/   Extracted keypoint JSON (30fps)
  annotations.json   Stroke labels (5 classes)
  windows/           Processed (45,17,3) .npz (gitignored)
  label_studio/      LS config + exports
  _debug/            Coordinate verification frames

tools/
  extract_skeletons.swift   Apple Vision skeleton extractor (30fps)
  infer_timestamps.py       Auto-infer stroke timestamps
  setup_label_studio.py     Label Studio import/export
  build_windows.py          Build 45-frame .npz windows
  build_unlabeled_windows.py  AimCLR unlabeled sequences
  verify_coords.py          Coordinate system sanity check

notebooks/
  01_dataset_inspection.ipynb
  02_aimclr_pretrain.ipynb
  03_ctrgcn_finetune.ipynb

ctrgcn/   CTR-GCN model code (PyTorch)
aimclr/   AimCLR pretraining wrapper
export/   Export scripts (PyTorch → ONNX → CoreML)
models/   Checkpoints (gitignored)
```

## Model Target

- Input:  `(1, 3, 45, 17, 1)` — batch × channels × frames × joints × person
- Output: `(1, 5)` — forehand / backhand / serve / volley / invalid
- Target: Top-1 ≥ 90%, per-class F1 ≥ 0.85, INT8 ~1.5 MB

## Coordinate Convention

Skeletons stored in raw Vision coordinates (bottom-left origin, [0,1] normalized).
Model input applies x↔y swap + Y-flip to match training data orientation.
**No** hip-center normalization for model input (display overlay only).

## Player Table

| ID  | Name | Videos | Date |
|-----|------|--------|------|
| p01 | Michael | 69 | 2026-03 |

*(Update as more players are added in Phase 1)*
