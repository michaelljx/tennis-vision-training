#!/usr/bin/env python3
"""
Label Studio setup and conversion tool for Tennis BiLSTM annotation.

Usage:
  Step 1 - Generate import tasks:
    python training/setup_label_studio.py --generate

  Step 2 - Convert Label Studio export to annotations.json:
    python training/setup_label_studio.py --convert path/to/export.json

Supports both Label Studio export formats:
  - Full JSON (recommended): exported with "JSON" format option
  - JSON-MIN (legacy): flat format, only works if annotations contain currentTime
"""

import json
import sys
import argparse
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent / "videos"
SKELETONS_DIR = Path(__file__).parent / "skeletons"
OUTPUT_DIR = Path(__file__).parent

STROKE_TYPES = {"forehand", "backhand", "serve", "volley", "invalid"}

LABEL_MAP = {
    "forehand": "forehand",
    "backhand": "backhand",
    "serve": "serve",
    "volley": "volley",
    "invalid": "invalid",
}


def generate_tasks():
    """Generate Label Studio import JSON for all local videos."""
    video_extensions = {".mp4", ".mov", ".m4v"}
    video_files = sorted([
        f for f in VIDEOS_DIR.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"No video files found in {VIDEOS_DIR}")
        sys.exit(1)

    tasks = [
        {"video": f"file://{f.absolute()}"}
        for f in video_files
    ]

    output_path = OUTPUT_DIR / "label_studio_tasks.json"
    with open(output_path, "w") as fh:
        json.dump(tasks, fh, indent=2)

    print(f"Generated {len(tasks)} tasks -> {output_path}")
    print("\nNext steps:")
    print("  1. Start Label Studio: label-studio start")
    print("  2. Create project with the labeling interface XML")
    print("  3. Settings > Cloud Storage > Local Files")
    print(f"     Set document root to: {VIDEOS_DIR.absolute()}")
    print("  4. Import label_studio_tasks.json into the project")
    print("  5. Annotate: pause at ball contact, draw box around player, select stroke type")
    print("     Target: 30 per class (150 total)")
    print("  6. Export > JSON format (NOT JSON-MIN) -> save as label_studio_full_export.json")
    print(f"  7. Run: python training/setup_label_studio.py --convert <export_file>")


def load_skeleton(video_name: str) -> list | None:
    """Load skeleton frames for a video. Returns list of frame dicts or None."""
    skel_path = SKELETONS_DIR / f"{video_name}.json"
    if not skel_path.exists():
        return None
    with open(skel_path) as fh:
        data = json.load(fh)
    return data.get("frames", [])


def find_nearest_frame(skeleton_frames: list, target_timestamp: float) -> int:
    """
    Find the index of the skeleton frame closest to target_timestamp.
    Uses actual frame timestamps rather than assuming a fixed FPS,
    which is important because:
      - extract_skeletons.swift targets 5fps but samples at ~6fps for 29.97fps video
      - Detection failures cause uneven frame spacing
    """
    if not skeleton_frames:
        return 0
    best_idx = 0
    best_diff = abs(skeleton_frames[0]["timestamp"] - target_timestamp)
    for i, frame in enumerate(skeleton_frames[1:], 1):
        diff = abs(frame["timestamp"] - target_timestamp)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def extract_video_stem(video_uri: str) -> str:
    """
    Extract the video stem (no extension, no path) from various URI formats:
      - file:///absolute/path/video1-0513_clip003.mov
      - /data/local-files/?d=videos/video1-0513_clip003.mov
      - video1-0513_clip003.mov
    """
    # Strip file:// prefix
    uri = video_uri.replace("file://", "")
    # Strip Label Studio local-files query path prefix
    if "?d=" in uri:
        uri = uri.split("?d=")[-1]
    return Path(uri).stem


def convert_export(export_path: str):
    """
    Convert Label Studio JSON export to annotations.json format.

    Supports Label Studio full JSON export (recommended) where each task has:
      {
        "id": 1,
        "data": {"video": "file:///path/video.mp4"},
        "annotations": [{
          "result": [
            {
              "type": "videorectangle",
              "value": {
                "sequence": [{"frame": 45, "x": ..., "y": ..., ...}],
                ...
              }
            },
            {
              "type": "labels",
              "value": {"labels": ["forehand"]}
            }
          ]
        }]
      }

    Also handles JSON-MIN format if it contains currentTime in choices results.
    """
    export_file = Path(export_path)
    if not export_file.exists():
        print(f"Export file not found: {export_path}")
        sys.exit(1)

    with open(export_file) as fh:
        ls_data = json.load(fh)

    # Detect format
    if isinstance(ls_data, list) and ls_data and "data" in ls_data[0]:
        print("Detected: Label Studio full JSON format")
        annotations = _convert_full_json(ls_data)
    elif isinstance(ls_data, list) and ls_data and "video" in ls_data[0]:
        print("Detected: Label Studio JSON-MIN format")
        annotations = _convert_json_min(ls_data)
    else:
        print("ERROR: Unrecognized export format")
        print("Please export using JSON format (not JSON-MIN) from Label Studio")
        sys.exit(1)

    output_path = OUTPUT_DIR / "annotations.json"
    with open(output_path, "w") as fh:
        json.dump(annotations, fh, indent=2)

    print(f"\nConverted {len(annotations)} annotations -> {output_path}")
    _print_stats(annotations)

    print("\nNext step:")
    print(f"  Upload {output_path} to Google Drive at:")
    print(f"  MyDrive/Tennis-BiLSTM/data/annotations.json")


def _convert_full_json(ls_data: list) -> list:
    """Convert Label Studio full JSON format."""
    annotations = []
    skipped = 0

    for task in ls_data:
        video_uri = task.get("data", {}).get("video", "")
        video_name = extract_video_stem(video_uri)

        skeleton_frames = load_skeleton(video_name)

        task_annotations = task.get("annotations", [])
        if not task_annotations:
            skipped += 1
            continue

        for ann in task_annotations:
            result_list = ann.get("result", [])

            # Collect all results: may have multiple videorectangle + labels pairs
            # Group by region id where possible, otherwise pair them up
            rect_results = []
            label_results = []

            for result in result_list:
                rtype = result.get("type", "")
                if rtype == "videorectangle":
                    rect_results.append(result)
                elif rtype in ("labels", "choices"):
                    label_results.append(result)

            # Strategy: pair each label with the nearest rect by id, or sequentially
            # Build a map from id -> rect for quick lookup
            rect_by_id = {r.get("id"): r for r in rect_results if r.get("id")}

            for label_result in label_results:
                stroke_type = _extract_stroke_type(label_result)
                if not stroke_type:
                    skipped += 1
                    continue

                # Find paired rect: same id, or parentID, or first available
                parent_id = label_result.get("parentID") or label_result.get("id")
                rect = rect_by_id.get(parent_id)
                if rect is None and rect_results:
                    rect = rect_results[0]

                timestamp = None
                frame_idx = None

                if rect is not None:
                    timestamp, frame_idx = _extract_time_from_rect(rect, skeleton_frames)

                # Fallback: try currentTime in label value
                if timestamp is None:
                    value = label_result.get("value", {})
                    current_time = value.get("currentTime")
                    if current_time is not None:
                        timestamp = round(float(current_time), 2)
                        if skeleton_frames:
                            frame_idx = find_nearest_frame(skeleton_frames, timestamp)
                        else:
                            frame_idx = int(timestamp * 6)  # fallback estimate

                if timestamp is None or frame_idx is None:
                    print(f"  Warning: no timestamp for {video_name} label={stroke_type}, skipping")
                    skipped += 1
                    continue

                annotations.append({
                    "video": video_name,
                    "timestamp": timestamp,
                    "stroke_type": stroke_type,
                    "frame_idx": frame_idx,
                })

    if skipped:
        print(f"Skipped {skipped} annotations (missing timestamp or unrecognized label)")

    return annotations


def _extract_time_from_rect(rect_result: dict, skeleton_frames: list | None) -> tuple:
    """
    Extract (timestamp, frame_idx) from a videorectangle result.
    Returns (None, None) if not found.
    """
    value = rect_result.get("value", {})
    sequence = value.get("sequence", [])

    if not sequence:
        return None, None

    first = sequence[0]

    # Prefer explicit time field
    if "time" in first:
        timestamp = round(float(first["time"]), 2)
        if skeleton_frames:
            frame_idx = find_nearest_frame(skeleton_frames, timestamp)
        else:
            frame_idx = int(timestamp * 6)
        return timestamp, frame_idx

    # Use frame number — need skeleton FPS to convert to timestamp
    if "frame" in first:
        frame_num = int(first["frame"])
        if skeleton_frames and frame_num < len(skeleton_frames):
            timestamp = round(skeleton_frames[frame_num]["timestamp"], 2)
            frame_idx = frame_num
            return timestamp, frame_idx
        # Fallback: estimate at 30fps video / 6fps skeleton
        timestamp = round(frame_num / 30.0, 2)
        frame_idx = frame_num // 5  # rough skeleton frame estimate
        return timestamp, frame_idx

    return None, None


def _extract_stroke_type(result: dict) -> str | None:
    """Extract and normalize stroke type from a labels or choices result."""
    value = result.get("value", {})
    labels = value.get("labels") or value.get("choices") or []
    if not labels:
        return None
    raw = labels[0]
    return LABEL_MAP.get(raw.lower())


def _convert_json_min(ls_data: list) -> list:
    """
    Convert Label Studio JSON-MIN format.
    Only works if annotations contain currentTime (older Label Studio behavior).
    """
    annotations = []
    skipped = 0

    for item in ls_data:
        video_uri = item.get("video", "")
        video_name = extract_video_stem(video_uri)
        stroke_type_raw = item.get("stroke_type")
        current_time = item.get("currentTime") or item.get("timestamp")

        if not stroke_type_raw:
            skipped += 1
            continue

        stroke_type = LABEL_MAP.get(str(stroke_type_raw).lower())
        if not stroke_type:
            print(f"  Warning: unknown label '{stroke_type_raw}' for {video_name}, skipping")
            skipped += 1
            continue

        if current_time is None:
            print(f"  Warning: no timestamp for {video_name} ({stroke_type})")
            print("  JSON-MIN format lacks timestamps. Re-export as full JSON from Label Studio.")
            skipped += 1
            continue

        timestamp = round(float(current_time), 2)
        skeleton_frames = load_skeleton(video_name)
        if skeleton_frames:
            frame_idx = find_nearest_frame(skeleton_frames, timestamp)
        else:
            frame_idx = int(timestamp * 6)

        annotations.append({
            "video": video_name,
            "timestamp": timestamp,
            "stroke_type": stroke_type,
            "frame_idx": frame_idx,
        })

    if skipped:
        print(f"Skipped {skipped} items (missing stroke_type or timestamp)")

    return annotations


def _print_stats(annotations: list):
    stats = {}
    for ann in annotations:
        st = ann["stroke_type"]
        stats[st] = stats.get(st, 0) + 1

    print("\nAnnotation stats:")
    target = 30
    for st in ["forehand", "backhand", "serve", "volley", "invalid"]:
        count = stats.get(st, 0)
        bar = "#" * min(count, 40)
        status = "DONE" if count >= target else f"need {target - count} more"
        print(f"  {st:10s}: {bar:40s} {count:3d}/{target}  {status}")
    print(f"\n  Total: {len(annotations)}")

    total_needed = sum(max(0, target - stats.get(st, 0)) for st in STROKE_TYPES)
    if total_needed == 0:
        print("\n  All classes complete!")
    else:
        print(f"\n  Still need {total_needed} more annotations")


def main():
    parser = argparse.ArgumentParser(description="Label Studio setup and conversion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true",
                       help="Generate Label Studio import tasks JSON")
    group.add_argument("--convert", metavar="EXPORT_FILE",
                       help="Convert Label Studio export to annotations.json")
    args = parser.parse_args()

    if args.generate:
        generate_tasks()
    else:
        convert_export(args.convert)


if __name__ == "__main__":
    main()
