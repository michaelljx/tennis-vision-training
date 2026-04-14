#!/usr/bin/env python3
"""
merge_annotations.py — Merge old video1-4 annotations with new Label Studio export.

New Label Studio file covers video5-7 + all 04- videos (103 total).
Old annotations.json covers video1-7 + 04- (172 total).
We take video1-4 from old (69 clips) + all from new (103 videos), skipping volley.
"""

import json
import urllib.parse
from pathlib import Path


def load_old_video1_4(old_path: Path) -> list[dict]:
    """Extract video1-4 annotations from old file, excluding volley."""
    with open(old_path) as f:
        data = json.load(f)

    result = []
    skipped_volley = 0
    for ann in data:
        video = ann["video"]
        # Only keep video1, video2, video3, video4
        if not any(video.startswith(f"video{i}") for i in [1, 2, 3, 4]):
            continue
        if ann["stroke_type"] == "volley":
            skipped_volley += 1
            continue
        result.append(ann)

    print(f"Old file (video1-4): {len(result)} annotations, {skipped_volley} volley skipped")
    return result


def convert_label_studio(ls_path: Path) -> list[dict]:
    """Convert Label Studio JSON export to annotations.json format, excluding volley."""
    with open(ls_path) as f:
        data = json.load(f)

    result = []
    skipped_volley = 0
    for task in data:
        video_path = task["data"]["video"]
        video_stem = urllib.parse.unquote(video_path.split("/")[-1]).replace(".mp4", "")

        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                value = r.get("value", {})
                labels = value.get("labels", [])
                sequence = value.get("sequence", [])

                if not labels or not sequence:
                    continue

                stroke_type = labels[0]
                if stroke_type == "volley":
                    skipped_volley += 1
                    continue

                frame = sequence[0].get("frame", 0)
                time = sequence[0].get("time", 0.0)

                result.append({
                    "video": video_stem,
                    "timestamp": round(time, 4),
                    "stroke_type": stroke_type,
                    "frame_idx": frame,
                })

    print(f"New file (Label Studio): {len(result)} annotations, {skipped_volley} volley skipped")
    return result


def main():
    old_path = Path("data/annotations.json")
    new_path = Path("data/label_studio/project-5-at-2026-04-14-05-10-d5cba2a1.json")
    out_path = Path("data/annotations.json")
    backup_path = Path("data/annotations_old_5class.json")

    # Backup old file
    if old_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(old_path, backup_path)
        print(f"Backed up old annotations to {backup_path}")

    # Merge
    old_anns = load_old_video1_4(old_path if not backup_path.exists() else backup_path)
    new_anns = convert_label_studio(new_path)
    merged = old_anns + new_anns

    # Sort by video then timestamp
    merged.sort(key=lambda a: (a["video"], a["timestamp"]))

    # Stats
    from collections import Counter
    labels = Counter(a["stroke_type"] for a in merged)
    videos = len(set(a["video"] for a in merged))
    print(f"\nMerged: {len(merged)} annotations, {videos} videos")
    print(f"Labels: {dict(sorted(labels.items()))}")
    assert "volley" not in labels, "Volley should not exist!"

    # Write
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
