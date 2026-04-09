#!/usr/bin/env python3
"""
build_windows.py — Phase 3: Dataset construction (45-frame windows)

Reads 30fps skeleton JSON files + annotations.json and produces:
  data/windows/train.npz   — shape (N_train, 45, 17, 3)
  data/windows/val.npz     — shape (N_val,   45, 17, 3)

Coordinate convention applied here:
  Input:  stored JSON coords — top-left origin (Y already flipped by extract_skeletons.swift)
  Output: CTR-GCN model input format — x↔y swap applied (matches iOS CVPipelineCoordinator)
  NO hip-center normalization (that is display-only via KeypointNormalizer in iOS)

Usage:
    python tools/build_windows.py \
        --skeletons data/skeletons_30fps/ \
        --annotations data/annotations.json \
        --output data/windows/ \
        [--pre 30] [--post 15] [--fps 30]

    # Preview stats without saving:
    python tools/build_windows.py --skeletons ... --annotations ... --output ... --dry-run
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
WINDOW_SIZE = 45          # total frames per sample
PRE_FRAMES  = 30          # frames before stroke event
POST_FRAMES = 15          # frames after stroke event
assert PRE_FRAMES + POST_FRAMES == WINDOW_SIZE

JOINT_COUNT   = 17        # COCO 17 keypoints
CHANNEL_COUNT = 3         # x, y, confidence

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

CLASS_NAMES = ["forehand", "backhand", "serve", "volley", "invalid"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Helpers ────────────────────────────────────────────────────────────────

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
    Applies x↔y swap to match iOS CVPipelineCoordinator model input.

    JSON stores:  x = vision_x (left-right), y = 1 - vision_y (top-left origin)
    Model input:  x_model = y_json (≈ vision_y flipped), y_model = x_json
    This x↔y swap is what CVPipelineCoordinator.swift does for raw model input.
    """
    arr = np.zeros((len(frames), JOINT_COUNT, CHANNEL_COUNT), dtype=np.float32)

    for t, frame in enumerate(frames):
        kp_map = {}
        for kp in frame["keypoints"]:
            kp_map[kp["joint"]] = kp

        for v, name in enumerate(JOINT_NAMES):
            if name in kp_map:
                kp = kp_map[name]
                x_json = float(kp["x"])
                y_json = float(kp["y"])
                conf   = float(kp["confidence"])
                # x↔y swap to match iOS model input
                arr[t, v, 0] = y_json   # model x = json y
                arr[t, v, 1] = x_json   # model y = json x
                arr[t, v, 2] = conf
    return arr


def find_nearest_frame_idx(all_frames: list, target_ts: float) -> int:
    """
    Return the index of the frame whose timestamp is closest to target_ts.
    Frames are NOT assumed to be uniformly spaced (Vision only stores detected frames).
    """
    best_idx = 0
    best_diff = abs(all_frames[0]["timestamp"] - target_ts)
    for i, f in enumerate(all_frames[1:], 1):
        diff = abs(f["timestamp"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def extract_window(all_frames: list, stroke_frame_idx: int, pre: int, post: int) -> np.ndarray:
    """
    Extract a (pre+post, V, C) window centred on stroke_frame_idx.
    Edge-pads with the boundary frame if window extends beyond video.

    Note: frames may be non-uniformly spaced in time (Vision skips frames
    where no person is detected). We take a contiguous slice by index.
    """
    n = len(all_frames)
    start = stroke_frame_idx - pre
    end   = stroke_frame_idx + post  # exclusive

    indices = [max(0, min(n - 1, i)) for i in range(start, end)]
    return frames_to_array([all_frames[i] for i in indices])


def infer_player_id(video_stem: str) -> str:
    """
    Infer player ID from video filename prefix.
    Convention: 'video1-...' → 'p01', 'video2-...' → 'p02', etc.
    Extend this as you add more players.
    """
    # Try to extract a numeric prefix
    for prefix in video_stem.split("-"):
        digits = "".join(c for c in prefix if c.isdigit())
        if digits:
            return f"p{int(digits):02d}"
    return "p00"  # fallback


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build 45-frame training windows")
    parser.add_argument("--skeletons",   required=True, help="Path to skeletons_30fps/ dir")
    parser.add_argument("--annotations", required=True, help="Path to annotations.json")
    parser.add_argument("--output",      required=True, help="Output dir for .npz files")
    parser.add_argument("--pre",  type=int, default=PRE_FRAMES,  help=f"Frames before event (default {PRE_FRAMES})")
    parser.add_argument("--post", type=int, default=POST_FRAMES, help=f"Frames after event (default {POST_FRAMES})")
    parser.add_argument("--fps",  type=int, default=30,  help="Skeleton FPS (default 30)")
    parser.add_argument("--val-players", nargs="+", default=None,
                        help="Player IDs to use as val set (e.g. p02 p05). "
                             "If omitted, uses 20%% random split by player.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    skel_dir    = Path(args.skeletons)
    annot_path  = Path(args.annotations)
    out_dir     = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_size = args.pre + args.post

    # Load annotations
    with open(annot_path) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations from {annot_path.name}")

    # Group by video
    by_video = defaultdict(list)
    for ann in annotations:
        by_video[ann["video"]].append(ann)

    # Collect all player IDs
    player_ids = sorted(set(infer_player_id(v) for v in by_video.keys()))
    print(f"Inferred player IDs: {player_ids}")

    # Determine val players
    val_players = set(args.val_players) if args.val_players else None
    if val_players is None:
        # Auto: last 20% of player IDs
        n_val = max(1, len(player_ids) // 5)
        val_players = set(player_ids[-n_val:])
    print(f"Val players: {sorted(val_players)}")
    print(f"Train players: {sorted(set(player_ids) - val_players)}")
    print()

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    skipped = 0

    for video_stem, anns in sorted(by_video.items()):
        skel_path = skel_dir / f"{video_stem}.json"
        if not skel_path.exists():
            print(f"  ⚠ Skeleton not found: {skel_path.name} — skipping {len(anns)} annotation(s)")
            skipped += len(anns)
            continue

        skel_data = load_skeleton(skel_path)
        if skel_data is None:
            skipped += len(anns)
            continue

        all_frames = skel_data["frames"]
        player_id = infer_player_id(video_stem)
        is_val = player_id in val_players

        for ann in anns:
            stroke_type = ann["stroke_type"]
            if stroke_type not in CLASS_INDEX:
                print(f"  ⚠ Unknown stroke type '{stroke_type}' — skipping")
                skipped += 1
                continue

            # Find the stored frame whose timestamp is nearest to the annotation.
            # We use timestamp-based lookup because Vision only stores frames where
            # a person was detected, so the frame list is non-uniformly spaced.
            ts = ann["timestamp"]
            frame_idx = find_nearest_frame_idx(all_frames, ts)

            window = extract_window(all_frames, frame_idx, args.pre, args.post)
            label  = CLASS_INDEX[stroke_type]

            if is_val:
                X_val.append(window)
                y_val.append(label)
            else:
                X_train.append(window)
                y_train.append(label)

    # Stats
    print("─── Dataset Summary ──────────────────────────────────────────────")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Skipped:       {skipped}")
    print()

    def class_dist(labels):
        dist = defaultdict(int)
        for l in labels: dist[CLASS_NAMES[l]] += 1
        return dict(dist)

    if X_train:
        print(f"  Train distribution: {class_dist(y_train)}")
    if X_val:
        print(f"  Val   distribution: {class_dist(y_val)}")
    print()

    if args.dry_run:
        print("(dry-run — not saving)")
        return

    if not X_train:
        print("✗ No training samples. Check skeleton paths and annotations.")
        sys.exit(1)

    # Save
    X_train_arr = np.stack(X_train).astype(np.float32)  # (N, T, V, C)
    y_train_arr = np.array(y_train, dtype=np.int64)
    train_path  = out_dir / "train.npz"
    np.savez_compressed(train_path, X=X_train_arr, y=y_train_arr,
                        class_names=CLASS_NAMES, joint_names=JOINT_NAMES)
    print(f"  Saved: {train_path}  shape={X_train_arr.shape}")

    if X_val:
        X_val_arr = np.stack(X_val).astype(np.float32)
        y_val_arr = np.array(y_val, dtype=np.int64)
        val_path  = out_dir / "val.npz"
        np.savez_compressed(val_path, X=X_val_arr, y=y_val_arr,
                            class_names=CLASS_NAMES, joint_names=JOINT_NAMES)
        print(f"  Saved: {val_path}  shape={X_val_arr.shape}")

    print()
    print("✓ Done. Next step: run notebooks/03_ctrgcn_finetune.ipynb on Colab.")
    print()
    print("Input convention reminder:")
    print("  Array shape (N, T=45, V=17, C=3) where C = (x_model, y_model, conf)")
    print("  x_model = json_y,  y_model = json_x  (x↔y swap applied)")
    print("  Reshape to (N, 3, 45, 17, 1) before feeding to CTR-GCN.")


if __name__ == "__main__":
    main()
