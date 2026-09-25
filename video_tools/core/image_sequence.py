from __future__ import annotations

import configparser
import re
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import cv2
import numpy as np


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_NUMBER_RE = re.compile(r"(\d+)")


def natural_path_key(path: Path) -> tuple[object, ...]:
    """Sort numbered frame names numerically (1, 2, 10), not lexically."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NUMBER_RE.split(path.name)
    )


def resolve_frames_dir(path: Path) -> Path | None:
    """Return the actual frames directory for a plain folder or MOT sequence root."""
    path = Path(path)
    if not path.is_dir():
        return None
    img1 = path / "img1"
    if img1.is_dir() and any(
        child.is_file() and child.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
        for child in img1.iterdir()
    ):
        return img1
    if any(
        child.is_file() and child.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
        for child in path.iterdir()
    ):
        return path
    return None


def is_image_sequence(path: Path) -> bool:
    return resolve_frames_dir(path) is not None


def scan_image_sequences(root: Path, recursive: bool) -> list[Path]:
    """Find frame folders while returning MOT roots instead of their img1 child."""
    if not root.is_dir():
        return []
    candidates = [root]
    if recursive:
        candidates.extend(p for p in root.rglob("*") if p.is_dir())
    else:
        candidates.extend(p for p in root.iterdir() if p.is_dir())

    found: list[Path] = []
    claimed_frame_dirs: set[Path] = set()
    # Prefer MOT roots so that det/det.txt and seqinfo.ini remain discoverable.
    for directory in candidates:
        frames_dir = directory / "img1"
        if frames_dir.is_dir() and resolve_frames_dir(directory) == frames_dir:
            found.append(directory)
            claimed_frame_dirs.add(frames_dir)
    for directory in candidates:
        if directory in claimed_frame_dirs or directory.name == "det":
            continue
        if resolve_frames_dir(directory) == directory:
            found.append(directory)
    return sorted(set(found))


class ImageSequence:
    """A numbered image sequence with a thread-safe, bounded decoded-frame cache."""

    def __init__(self, path: Path, cache_size: int = 24, default_fps: float = 30.0):
        self.path = Path(path)
        frames_dir = resolve_frames_dir(self.path)
        if frames_dir is None:
            raise ValueError(f"No image frames found in {self.path}")
        self.frames_dir = frames_dir
        selected_img1 = self.path == frames_dir and frames_dir.name == "img1"
        self.sequence_root = frames_dir.parent if selected_img1 else self.path
        self.frame_paths = sorted(
            (
                child
                for child in frames_dir.iterdir()
                if child.is_file() and child.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
            ),
            key=natural_path_key,
        )
        if not self.frame_paths:
            raise ValueError(f"No image frames found in {frames_dir}")
        self.fps = self._read_fps(default_fps)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        return len(self.frame_paths)

    @property
    def duration_ms(self) -> int:
        return round(len(self) / self.fps * 1000)

    @property
    def frame_interval_ms(self) -> int:
        return max(1, round(1000 / self.fps))

    @property
    def detection_file(self) -> Path | None:
        candidates = [
            self.sequence_root / "det" / "det.txt",
            self.frames_dir / "det" / "det.txt",
        ]
        return next((path for path in candidates if path.is_file()), None)

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self._lock:
            cached = self._cache.pop(index, None)
            if cached is not None:
                self._cache[index] = cached
                return cached
        frame_bgr = cv2.imread(str(self.frame_paths[index]), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame: {self.frame_paths[index]}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            existing = self._cache.pop(index, None)
            self._cache[index] = existing if existing is not None else frame_rgb
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return self._cache[index]

    def prefetch(self, start: int, count: int = 4) -> None:
        for index in range(start, min(len(self), start + count)):
            self.frame(index)

    def _read_fps(self, fallback: float) -> float:
        config_path = self.sequence_root / "seqinfo.ini"
        if config_path.is_file():
            parser = configparser.ConfigParser()
            parser.read(config_path)
            try:
                fps = parser.getfloat("Sequence", "frameRate")
                if fps > 0:
                    return fps
            except (configparser.Error, ValueError):
                pass
        return max(0.1, float(fallback))
