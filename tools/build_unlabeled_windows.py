#!/usr/bin/env python3
from __future__ import annotations
"""
build_unlabeled_windows.py — Phase 4: Unlabeled sliding-window dataset for AimCLR

Reads all 30fps skeleton JSON files and extracts overlapping 45-frame windows
using a sliding window (stride=15, ~0.5s) WITHOUT requiring stroke annotations.

Applies the same x↔y swap as build_windows.py (matches iOS CVPipelineCoordinator).
Filters windows where mean confidence < min_conf (Vision detection failures).

Usage:
    python tools/build_unlabeled_windows.py \
        --skeletons data/skeletons_30fps/ \
        --output data/windows/ \
        [--window 45] [--stride 15] [--min-conf 0.1] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("Missing dependency: pip install numpy")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────
WINDOW_SIZE   = 45
STRIDE        = 15
JOINT_COUNT   = 17
CHANNEL_COUNT = 3
MIN_CONF      = 0.1

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ── Helpers (mirror of build_windows.py) ──────────────────────────────────

def load_skeleton(skel_path: Path) -> dict | None:
    try:
        with open(skel_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Cannot load {skel_path.name}: {e}")
        return None


def frames_to_array(frames: list) -> np.ndarray:
    """
    Convert list of frame dicts → numpy array (T, V, C).
    Applies x↔y swap identical to build_windows.py.
    """
    arr = np.zeros((len(frames), JOINT_COUNT, CHANNEL_COUNT), dtype=np.float32)
    for t, frame in enumerate(frames):
        kp_map = {kp["joint"]: kp for kp in frame["keypoints"]}
        for v, name in enumerate(JOINT_NAMES):
            if name in kp_map:
                kp = kp_map[name]
                arr[t, v, 0] = float(kp["y"])           # model x = json y
                arr[t, v, 1] = float(kp["x"])           # model y = json x
                arr[t, v, 2] = float(kp["confidence"])
    return arr


def extract_sliding_windows(
    all_frames: list,
    window_size: int,
    stride: int,
    min_conf: float,
) -> list[np.ndarray]:
    """
    Slide over all_frames with the given stride, return windows that pass
    the mean-confidence filter.
    """
    windows = []
    n = len(all_frames)
    for start in range(0, n - window_size + 1, stride):
        window_frames = all_frames[start : start + window_size]
        arr = frames_to_array(window_frames)
        # Filter: require average confidence across all joints/frames >= min_conf
        if arr[:, :, 2].mean() >= min_conf:
            windows.append(arr)
    return windows


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build unlabeled sliding-window dataset for AimCLR pretraining"
    )
    parser.add_argument("--skeletons", required=True, help="Path to skeletons_30fps/ dir")
    parser.add_argument("--output",    required=True, help="Output dir for unlabeled.npz")
    parser.add_argument("--window",   type=int, default=WINDOW_SIZE, help=f"Window size (default {WINDOW_SIZE})")
    parser.add_argument("--stride",   type=int, default=STRIDE,      help=f"Stride (default {STRIDE})")
    parser.add_argument("--min-conf", type=float, default=MIN_CONF,  help=f"Min mean confidence (default {MIN_CONF})")
    parser.add_argument("--dry-run",  action="store_true",           help="Print stats without saving")
    args = parser.parse_args()

    skel_dir = Path(args.skeletons)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    skel_files = sorted(skel_dir.glob("*.json"))
    if not skel_files:
        print(f"✗ No JSON files found in {skel_dir}")
        sys.exit(1)

    print(f"Skeleton dir: {skel_dir}  ({len(skel_files)} files)")
    print(f"Window={args.window}  Stride={args.stride}  MinConf={args.min_conf}")
    print()

    all_windows = []
    per_file_stats = defaultdict(int)
    skipped_files = 0

    for skel_path in skel_files:
        skel_data = load_skeleton(skel_path)
        if skel_data is None:
            skipped_files += 1
            continue

        frames = skel_data["frames"]
        if len(frames) < args.window:
            # Video too short for even one window — skip silently
            per_file_stats["too_short"] += 1
            continue

        windows = extract_sliding_windows(frames, args.window, args.stride, args.min_conf)
        all_windows.extend(windows)
        per_file_stats["ok"] += 1

    print("─── Unlabeled Dataset Summary ────────────────────────────────")
    print(f"  Files processed:  {per_file_stats['ok']}")
    print(f"  Files too short:  {per_file_stats['too_short']}")
    print(f"  Files skipped:    {skipped_files}")
    print(f"  Total windows:    {len(all_windows)}")
    print()

    if len(all_windows) < 3000:
        print("⚠  WARNING: fewer than 3,000 windows.")
        print("   AimCLR benefit will be limited. Consider skipping to Phase 5.")
        print()

    if args.dry_run:
        print("(dry-run — not saving)")
        return

    if not all_windows:
        print("✗ No windows extracted. Check --skeletons path and --min-conf threshold.")
        sys.exit(1)

    X = np.stack(all_windows).astype(np.float32)  # (M, T, V, C)
    out_path = out_dir / "unlabeled.npz"
    np.savez_compressed(out_path, X=X, joint_names=JOINT_NAMES)
    print(f"  Saved: {out_path}  shape={X.shape}")
    print()
    print("✓ Done. Next step: run notebooks/02_aimclr_pretrain.ipynb on Colab.")
    print()
    print("Input convention reminder:")
    print("  Array shape (M, T=45, V=17, C=3) where C = (x_model, y_model, conf)")
    print("  Reshape to (N, 3, 45, 17, 1) before feeding to CTR-GCN.")


if __name__ == "__main__":
    main()
