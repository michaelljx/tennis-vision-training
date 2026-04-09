#!/usr/bin/env python3
"""
从骨骼速度信号自动推断击球时间戳。

用法：
  python training/infer_timestamps.py

输入：
  training/label_studio_full_export.json  — Label Studio 导出（只需 stroke_type）
  training/skeletons/*.json               — 骨骼数据（含帧时间戳）

输出：
  training/annotations.json              — 可直接上传 Colab 的标注文件

算法：
  1. 对每个 clip，计算每帧手腕速度
  2. 非 invalid 击球 → 取速度最高的 N 个峰值，按时间顺序匹配标注
  3. invalid → 取速度最低的 N 个帧（非击球时刻）
  4. 假设标注者按视频顺序打标（第 1 条 = 时间上最早的那次击球）
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

EXPORT_PATH = Path(__file__).parent / "label_studio_full_export.json"
SKELETONS_DIR = Path(__file__).parent / "skeletons"
OUTPUT_PATH = Path(__file__).parent / "annotations.json"

STROKE_TYPES = {"forehand", "backhand", "serve", "volley", "invalid"}


def load_skeleton(video_name: str):
    """返回 (timestamps, frames_list) 或 (None, None)"""
    path = SKELETONS_DIR / f"{video_name}.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get("frames", [])


def compute_wrist_speed(frames: list) -> np.ndarray:
    """
    计算每帧的手腕速度（像素/秒）。
    取左右手腕中置信度较高的，若置信度 < 0.3 则速度为 0。
    """
    speed = np.zeros(len(frames))
    for i in range(1, len(frames)):
        curr = frames[i]
        prev = frames[i - 1]

        kp_curr = {k["joint"]: k for k in curr["keypoints"]}
        kp_prev = {k["joint"]: k for k in prev["keypoints"]}

        def best_wrist(kp):
            rw = kp.get("right_wrist", {"x": 0, "y": 0, "confidence": 0})
            lw = kp.get("left_wrist", {"x": 0, "y": 0, "confidence": 0})
            return rw if rw["confidence"] >= lw["confidence"] else lw

        c = best_wrist(kp_curr)
        p = best_wrist(kp_prev)

        if c["confidence"] > 0.3 and p["confidence"] > 0.3:
            dt = curr["timestamp"] - prev["timestamp"]
            if dt > 0:
                dx = c["x"] - p["x"]
                dy = c["y"] - p["y"]
                speed[i] = np.sqrt(dx**2 + dy**2) / dt

    return speed


def find_peaks(speed: np.ndarray, min_speed: float = 0.2) -> list[int]:
    """找所有局部速度峰值（帧索引列表）。"""
    peaks = []
    for i in range(1, len(speed) - 1):
        if (speed[i] > speed[i - 1]
                and speed[i] > speed[i + 1]
                and speed[i] > min_speed):
            peaks.append(i)
    return peaks


def assign_timestamps(frames: list, stroke_types: list) -> list[dict]:
    """
    对单个 clip 的标注列表，推断每条的帧索引和时间戳。

    参数：
      frames       — 骨骼帧列表（含 timestamp）
      stroke_types — 按标注顺序排列的 stroke_type 字符串列表

    返回：
      [{"frame_idx": int, "timestamp": float, "stroke_type": str}, ...]
    """
    if not frames or not stroke_types:
        return []

    timestamps = np.array([f["timestamp"] for f in frames])
    speed = compute_wrist_speed(frames)

    n_stroke = sum(1 for t in stroke_types if t != "invalid")
    n_invalid = sum(1 for t in stroke_types if t == "invalid")

    # ── 找峰值 ────────────────────────────────────────────────────────────
    peaks = find_peaks(speed, min_speed=0.2)

    # 如果峰值不足，降低阈值重试
    if len(peaks) < n_stroke:
        peaks = find_peaks(speed, min_speed=0.05)

    # 最后兜底：若仍然不够，用速度最高的 N 帧
    if len(peaks) < n_stroke:
        peaks = list(np.argsort(speed)[::-1][:max(n_stroke, 1)])

    # ── 分配非 invalid 击球时刻 ──────────────────────────────────────────
    # 取速度最高的 n_stroke 个峰值，按时间排序
    peak_speeds = [(p, speed[p]) for p in peaks]
    peak_speeds.sort(key=lambda x: -x[1])          # 按速度降序
    top_peaks = [p for p, _ in peak_speeds[:n_stroke]]
    top_peaks.sort()                                 # 按时间升序（标注顺序）

    # ── 分配 invalid 时刻 ────────────────────────────────────────────────
    if n_invalid > 0:
        peak_set = set(top_peaks)
        # 候选：速度最低的帧，且不在峰值附近（±3 帧）
        excluded = set()
        for p in top_peaks:
            for delta in range(-3, 4):
                excluded.add(p + delta)

        non_peak_indices = [
            i for i in range(len(frames))
            if i not in excluded and i not in peak_set
        ]

        if len(non_peak_indices) >= n_invalid:
            # 均匀分布取 n_invalid 个低速帧
            step = max(1, len(non_peak_indices) // n_invalid)
            invalid_frames = [non_peak_indices[i * step] for i in range(n_invalid)]
        else:
            # 兜底：均匀间隔
            step = max(1, len(frames) // (n_invalid + 1))
            invalid_frames = [min((i + 1) * step, len(frames) - 1) for i in range(n_invalid)]
    else:
        invalid_frames = []

    # ── 组合并按时间排序 ──────────────────────────────────────────────────
    # stroke_types 中按出现顺序，先消耗 top_peaks，再消耗 invalid_frames
    stroke_iter = iter(top_peaks)
    invalid_iter = iter(sorted(invalid_frames))

    results = []
    for stype in stroke_types:
        if stype == "invalid":
            try:
                idx = next(invalid_iter)
            except StopIteration:
                idx = len(frames) // 2  # 兜底
        else:
            try:
                idx = next(stroke_iter)
            except StopIteration:
                idx = len(frames) // 2  # 兜底

        results.append({
            "frame_idx": int(idx),
            "timestamp": round(float(timestamps[idx]), 2),
            "stroke_type": stype,
        })

    return results


def extract_video_stem(uri: str) -> str:
    if "?d=" in uri:
        uri = uri.split("?d=")[-1]
    uri = uri.replace("file://", "")
    return Path(uri).stem


def main():
    if not EXPORT_PATH.exists():
        print(f"ERROR: {EXPORT_PATH} not found")
        print("Run: label-studio export project -> JSON format")
        return

    with open(EXPORT_PATH) as f:
        export = json.load(f)

    annotations = []
    stats = defaultdict(int)
    skipped_clips = []

    for task in export:
        uri = task["data"]["video"]
        video_name = extract_video_stem(uri)

        # 收集该 clip 所有标注的 stroke_type（按 annotation id 顺序）
        stroke_types = []
        for ann in sorted(task.get("annotations", []), key=lambda a: a.get("id", 0)):
            for r in ann.get("result", []):
                choices = r.get("value", {}).get("choices", [])
                if choices:
                    raw = choices[0].lower()
                    if raw in STROKE_TYPES:
                        stroke_types.append(raw)

        if not stroke_types:
            continue

        frames = load_skeleton(video_name)
        if not frames:
            print(f"  WARNING: no skeleton for {video_name}, skipping")
            skipped_clips.append(video_name)
            continue

        assigned = assign_timestamps(frames, stroke_types)

        for item in assigned:
            annotations.append({
                "video": video_name,
                "timestamp": item["timestamp"],
                "stroke_type": item["stroke_type"],
                "frame_idx": item["frame_idx"],
            })
            stats[item["stroke_type"]] += 1

    # 写输出
    with open(OUTPUT_PATH, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n✓ 生成 {len(annotations)} 条标注 → {OUTPUT_PATH}")
    if skipped_clips:
        print(f"  跳过 {len(skipped_clips)} 个 clip（无骨骼文件）: {skipped_clips}")

    # 统计
    print("\n标注统计：")
    target = 30
    for st in ["forehand", "backhand", "serve", "volley", "invalid"]:
        count = stats.get(st, 0)
        bar = "#" * min(count, 40)
        status = "✓ DONE" if count >= target else f"差 {target - count}"
        print(f"  {st:10s}: {bar:40s} {count:3d}/{target}  {status}")
    print(f"\n  合计: {len(annotations)}")

    needed = sum(max(0, target - stats.get(st, 0)) for st in STROKE_TYPES)
    if needed == 0:
        print("\n  所有类别达标！可以上传 Colab 训练。")
    else:
        print(f"\n  还差 {needed} 条标注（需在 Label Studio 补标后重新运行）")


if __name__ == "__main__":
    main()
