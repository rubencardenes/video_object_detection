from __future__ import annotations

import datetime
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
from loguru import logger


@dataclass
class VideoInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    codec_name: str
    format_name: str
    bit_rate: int | None
    size_bytes: int
    created: datetime.datetime


def _resolve_ffprobe(ffprobe_path: str | None) -> str | None:
    return ffprobe_path or shutil.which("ffprobe")


def _parse_frame_rate(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate)


def _probe_with_ffprobe(path: Path, ffprobe: str) -> VideoInfo:
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    fmt = data.get("format", {})

    fps = _parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))
    duration = float(fmt.get("duration") or video_stream.get("duration") or 0.0)
    bit_rate = fmt.get("bit_rate")

    return VideoInfo(
        path=path,
        duration_sec=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        codec_name=video_stream.get("codec_name", "unknown"),
        format_name=fmt.get("format_long_name") or fmt.get("format_name", "unknown"),
        bit_rate=int(bit_rate) if bit_rate else None,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        created=_creation_time(path),
    )


def _creation_time(path: Path) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(path.stat().st_ctime)


def _probe_with_cv2(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (frame_count / fps) if fps else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()
    finally:
        cap.release()

    return VideoInfo(
        path=path,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        codec_name=f"{fourcc_str or 'unknown'} (ffprobe unavailable)",
        format_name=path.suffix.lstrip("."),
        bit_rate=None,
        size_bytes=path.stat().st_size,
        created=_creation_time(path),
    )


def probe_video(path: Path, ffprobe_path: str | None = None) -> VideoInfo:
    ffprobe = _resolve_ffprobe(ffprobe_path)
    if ffprobe:
        try:
            return _probe_with_ffprobe(path, ffprobe)
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
            logger.warning(f"ffprobe failed for {path}, falling back to cv2: {exc}")
    return _probe_with_cv2(path)
