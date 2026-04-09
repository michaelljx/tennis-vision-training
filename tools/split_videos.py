#!/usr/bin/env python3
"""
视频切分工具
将大视频文件切分为 30-50MB 的小片段，便于在 Colab 中加载和标注

使用方法:
    python training/split_videos.py

输入: ./training/videos/
输出: ./training/videos_split/
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path


def check_ffmpeg():
    """检查 ffmpeg 是否安装"""
    if not shutil.which('ffmpeg'):
        print("❌ 错误: 未找到 ffmpeg")
        print("\n请安装 ffmpeg:")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg")
        print("  Windows: 从 https://ffmpeg.org/download.html 下载")
        sys.exit(1)

    if not shutil.which('ffprobe'):
        print("❌ 错误: 未找到 ffprobe (ffmpeg 的一部分)")
        sys.exit(1)

    print("✓ ffmpeg 已安装")


def get_video_info(video_path):
    """获取视频时长和大小"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration,size',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        duration = float(info['format']['duration'])
        size_bytes = int(info['format']['size'])
        size_mb = size_bytes / (1024 * 1024)

        return duration, size_mb
    except Exception as e:
        print(f"  ❌ 获取视频信息失败: {e}")
        return None, None


def calculate_segment_duration(total_duration, total_size_mb, target_size_mb=40.0):
    """
    计算每个片段的时长

    Args:
        total_duration: 视频总时长（秒）
        total_size_mb: 视频总大小（MB）
        target_size_mb: 目标片段大小（MB），默认 40MB（在 30-50MB 范围内）

    Returns:
        每个片段的时长（秒）
    """
    # 计算比特率 (MB/s)
    bitrate_mb_per_sec = total_size_mb / total_duration

    # 计算每个片段的时长
    segment_duration = target_size_mb / bitrate_mb_per_sec

    # 确保至少 10 秒（避免片段太短，影响标注）
    segment_duration = max(10.0, segment_duration)

    # 确保不超过 120 秒（避免片段太长）
    segment_duration = min(120.0, segment_duration)

    return segment_duration


def split_video(input_path, output_dir, segment_duration):
    """
    使用 ffmpeg 切分视频

    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每个片段的时长（秒）

    Returns:
        生成的片段文件列表
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件模板: 原文件名_clip001.mp4
    output_pattern = output_dir / f"{input_path.stem}_clip%03d{input_path.suffix}"

    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c', 'copy',  # 复制流，不重新编码（快速且无质量损失）
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-reset_timestamps', '1',  # 重置每个片段的时间戳从 0 开始
        '-y',  # 覆盖已存在的文件
        str(output_pattern)
    ]

    print(f"  执行切分...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # 返回生成的文件列表
        clips = sorted(output_dir.glob(f"{input_path.stem}_clip*{input_path.suffix}"))
        return clips

    except subprocess.CalledProcessError as e:
        print(f"  ❌ 切分失败:")
        print(f"     {e.stderr}")
        return []


def main():
    print("=" * 70)
    print("视频切分工具 - 将大视频切分为 30-50MB 小片段")
    print("=" * 70)
    print()

    # 检查 ffmpeg
    check_ffmpeg()
    print()

    # 输入输出目录
    input_dir = Path('./training/videos')
    output_dir = Path('./training/videos_split')

    if not input_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        print(f"   请确保在项目根目录运行此脚本")
        sys.exit(1)

    # 查找所有视频文件
    video_extensions = ['*.mp4', '*.mov', '*.m4v', '*.MP4', '*.MOV', '*.M4V']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(ext))

    if not video_files:
        print(f"❌ 错误: 在 {input_dir} 中未找到视频文件")
        sys.exit(1)

    print(f"找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        size_mb = vf.stat().st_size / (1024 * 1024)
        print(f"  - {vf.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 70)
    print("开始处理...")
    print("=" * 70)

    total_clips = 0

    # 处理每个视频
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {video_path.name}")

        # 获取视频信息
        duration, size_mb = get_video_info(video_path)

        if duration is None or size_mb is None:
            print(f"  ⏭️  跳过（无法获取视频信息）")
            continue

        print(f"  时长: {duration:.1f}s ({duration/60:.1f}分钟)")
        print(f"  大小: {size_mb:.1f} MB")

        # 如果视频已经很小，跳过
        if size_mb <= 50:
            print(f"  ⏭️  跳过（文件已经 ≤ 50MB，无需切分）")
            continue

        # 计算切分参数
        segment_duration = calculate_segment_duration(duration, size_mb, target_size_mb=40.0)
        num_segments = int(duration / segment_duration) + 1
        avg_segment_size = size_mb / num_segments

        print(f"  切分参数:")
        print(f"    - 每段时长: {segment_duration:.1f}s")
        print(f"    - 预计片段数: {num_segments}")
        print(f"    - 预计每段大小: {avg_segment_size:.1f} MB")

        # 切分视频
        clips = split_video(video_path, output_dir, segment_duration)

        if clips:
            print(f"  ✅ 成功生成 {len(clips)} 个片段:")
            for clip in clips:
                clip_size = clip.stat().st_size / (1024 * 1024)
                print(f"     - {clip.name} ({clip_size:.1f} MB)")
            total_clips += len(clips)
        else:
            print(f"  ❌ 切分失败")

    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"✅ 共生成 {total_clips} 个视频片段")
    print(f"✅ 输出目录: {output_dir.absolute()}")
    print()
    print("下一步:")
    print("1. 对切分后的视频运行骨骼提取")
    print("2. 在 Colab 中使用切分后的小视频进行标注")


if __name__ == '__main__':
    main()
