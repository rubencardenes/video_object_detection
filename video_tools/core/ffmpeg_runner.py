from __future__ import annotations

import shutil
from pathlib import Path

from video_tools.config.settings import Settings

PROGRESS_ARGS = ["-progress", "pipe:2", "-nostats"]


class FfmpegNotFoundError(RuntimeError):
    pass


def find_ffmpeg(settings: Settings) -> str:
    ffmpeg = settings.ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        raise FfmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it or set ffmpeg_path in settings.yaml."
        )
    return ffmpeg


def build_convert_cmd(src: Path, dst: Path, ffmpeg: str) -> list[str]:
    return [
        ffmpeg, "-y",
        "-i", str(src),
        "-c:v", "libx264",
        # libx264 requires even width/height; screen recordings/webcam clips often aren't.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "aac",
        *PROGRESS_ARGS,
        str(dst),
    ]


def build_resize_cmd(src: Path, dst: Path, width: int, height: int, ffmpeg: str) -> list[str]:
    # width/height are expected to already be even (libx264 requirement).
    return [
        ffmpeg, "-y",
        "-i", str(src),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-c:a", "copy",
        *PROGRESS_ARGS,
        str(dst),
    ]


def build_cut_cmd(src: Path, dst: Path, start_sec: float, end_sec: float, ffmpeg: str) -> list[str]:
    # -ss before -i does fast input seeking; -t (duration, as an output option) is
    # unambiguous regardless of timestamp-reset behavior, unlike -to which can be
    # interpreted relative to the seek point rather than the original timeline.
    # -c copy is a stream copy (no re-encode), so it's fast but only cuts accurately
    # at the nearest keyframe before the requested start point.
    duration = max(0.0, end_sec - start_sec)
    return [
        ffmpeg, "-y",
        "-ss", str(start_sec),
        "-i", str(src),
        "-t", str(duration),
        "-c", "copy",
        *PROGRESS_ARGS,
        str(dst),
    ]


def build_fps_cmd(src: Path, dst: Path, fps: float, ffmpeg: str) -> list[str]:
    return [
        ffmpeg, "-y",
        "-i", str(src),
        "-r", str(fps),
        "-c:v", "libx264",
        # libx264 requires even width/height; the source may not have it (e.g. odd-height captures).
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "copy",
        *PROGRESS_ARGS,
        str(dst),
    ]
