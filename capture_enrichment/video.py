"""
Video processing via FFmpeg.

Responsibilities:
- Select the best available video from a capture package (SBS → left fallback)
- Downsample to a lower resolution/framerate for Gemini analysis
- Plan overlapping chunk windows
- Extract individual chunk files
"""

import json
import subprocess
from pathlib import Path

import ffmpeg

from .models import CapturePackage


def select_video(pkg: CapturePackage) -> Path:
    """
    Return the SBS video path if it exists, otherwise camera_left.
    (Already resolved during ingest; this is a convenience accessor.)
    """
    return pkg.video_path


def get_video_duration(video_path: Path) -> float:
    """Use ffprobe to return video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    for stream in probe.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise ValueError(f"Could not determine duration for {video_path}")


def downsample_video(
    input_path: Path,
    output_path: Path,
    fps: int = 5,
    height: int = 480,
) -> Path:
    """
    Downsample video to a lower resolution and framerate for Gemini analysis.

    Uses scale=-2:{height} to preserve aspect ratio (width rounded to even).
    Overwrites output if it already exists.
    """
    (
        ffmpeg
        .input(str(input_path))
        .filter("scale", "-2", str(height))
        .filter("fps", fps=fps)
        .output(str(output_path), vcodec="libx264", crf=23, preset="fast", acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def plan_chunks(
    duration_sec: float,
    chunk_sec: int = 30,
    overlap_sec: int = 5,
) -> list[tuple[float, float]]:
    """
    Generate (start_sec, end_sec) pairs for overlapping chunks.

    Step = chunk_sec - overlap_sec. The last chunk always extends to duration_sec
    to avoid dropping trailing frames.

    Example — 90s video, chunk=30, overlap=5:
        [(0, 30), (25, 55), (50, 80), (75, 90)]
    """
    step = chunk_sec - overlap_sec
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + chunk_sec, duration_sec)
        chunks.append((start, end))
        if end >= duration_sec:
            break
        start += step
    return chunks


def extract_chunk(src: Path, dst: Path, start_sec: float, end_sec: float) -> Path:
    """
    Extract a time segment from src into dst using stream copy (no re-encode).

    Uses -ss before -i (input seek) for fast extraction on large files.
    """
    duration = end_sec - start_sec
    (
        ffmpeg
        .input(str(src), ss=start_sec)
        .output(str(dst), t=duration, c="copy")
        .overwrite_output()
        .run(quiet=True)
    )
    return dst
