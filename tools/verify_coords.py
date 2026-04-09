#!/usr/bin/env python3
"""
verify_coords.py — Coordinate system sanity check (Phase 2.5)

Renders skeleton stick figures overlaid on video frames to verify
that extracted keypoints align with the actual human body.

Usage:
    python tools/verify_coords.py \
        --video  data/raw_videos/video1-0513_clip000.mov \
        --skel   data/skeletons_30fps/video1-0513_clip000.json \
        --output data/_debug/ \
        --frames 5 10 30 60 90

Output:
    data/_debug/<video_stem>_frame<N>.png  — one image per frame
    Human review: skeleton joints should sit ON the body, not mirrored/flipped.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependencies. Run:  pip install opencv-python numpy")
    sys.exit(1)


# COCO 17-joint skeleton edges (joint index pairs)
SKELETON_EDGES = [
    (0, 1), (0, 2),          # nose → eyes
    (1, 3), (2, 4),          # eyes → ears
    (5, 6),                  # shoulders
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # shoulders → hips
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_skeleton(skel_path: Path) -> dict:
    with open(skel_path) as f:
        return json.load(f)


def get_frame_by_index(video_path: Path, target_frame_index: int):
    """Extract a specific processed-frame-index from video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_raw


def get_frame_at_timestamp(video_path: Path, timestamp: float):
    """Seek to timestamp (seconds) and return BGR frame."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def draw_skeleton(frame: np.ndarray, keypoints: list, conf_thresh: float = 0.1) -> np.ndarray:
    """Draw skeleton joints and edges on a copy of frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # Build joint position dict
    joint_pos = {}
    for kp in keypoints:
        name = kp["joint"]
        x = kp["x"]
        y = kp["y"]  # already top-left origin (Y was flipped in extract_skeletons.swift)
        conf = kp["confidence"]
        if name in JOINT_NAMES and conf >= conf_thresh:
            px = int(x * w)
            py = int(y * h)
            joint_pos[name] = (px, py)

    # Draw edges
    for i, j in SKELETON_EDGES:
        n1, n2 = JOINT_NAMES[i], JOINT_NAMES[j]
        if n1 in joint_pos and n2 in joint_pos:
            cv2.line(img, joint_pos[n1], joint_pos[n2], (0, 255, 0), 2)

    # Draw joints
    for name, (px, py) in joint_pos.items():
        cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
        cv2.circle(img, (px, py), 5, (255, 255, 255), 1)

    # Label
    cv2.putText(img, "Verify: joints should align with body",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img


def main():
    parser = argparse.ArgumentParser(description="Coordinate system verification tool")
    parser.add_argument("--video",  required=True, help="Path to source video file")
    parser.add_argument("--skel",   required=True, help="Path to skeleton JSON file")
    parser.add_argument("--output", default="data/_debug/", help="Output directory for debug PNGs")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 5, 15, 30, 60],
                        help="Skeleton frame indices to visualize")
    args = parser.parse_args()

    video_path = Path(args.video)
    skel_path  = Path(args.skel)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        sys.exit(1)
    if not skel_path.exists():
        print(f"✗ Skeleton JSON not found: {skel_path}")
        sys.exit(1)

    skel_data = load_skeleton(skel_path)
    frames_data = {f["frame_index"]: f for f in skel_data["frames"]}
    stem = video_path.stem

    print(f"Video:     {video_path.name}")
    print(f"Skeleton:  {skel_path.name}  ({len(skel_data['frames'])} frames @ {skel_data['fps']} fps)")
    print(f"Output:    {out_dir}/")
    print()

    saved = 0
    for fi in args.frames:
        if fi not in frames_data:
            print(f"  [frame {fi:4d}] ✗ Not in skeleton data (max index: {max(frames_data.keys())})")
            continue

        frame_info = frames_data[fi]
        ts = frame_info["timestamp"]
        bgr = get_frame_at_timestamp(video_path, ts)
        if bgr is None:
            print(f"  [frame {fi:4d}] ✗ Could not seek to timestamp {ts:.3f}s")
            continue

        annotated = draw_skeleton(bgr, frame_info["keypoints"])
        out_path = out_dir / f"{stem}_frame{fi:04d}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"  [frame {fi:4d}] ✓ Saved: {out_path.name}  (t={ts:.2f}s)")
        saved += 1

    print(f"\nSaved {saved} debug frame(s) to {out_dir}/")
    print()
    print("─── Coordinate Check ───────────────────────────────────────────")
    print("  PASS: Green skeleton lines sit ON the body in each image")
    print("  FAIL: Skeleton is flipped/mirrored/offset → check extract_skeletons.swift")
    print("────────────────────────────────────────────────────────────────")
    print()
    print("Note: Stored Y is top-left origin (Y-flipped from Vision raw).")
    print("  Model input additionally requires x↔y swap (done in build_windows.py).")


if __name__ == "__main__":
    main()
