from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from loguru import logger

# detection.config is dependency-free (no cv2/onnx/Qt) and the detection package's
# __init__ is lazy, so importing DetectionConfig here doesn't drag the detection
# runtime into config loading.
from video_tools.detection.config import DetectionConfig

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "settings.yaml"

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
DEFAULT_THUMBNAIL_SIZE = (160, 90)


@dataclass
class Settings:
    videos_root: Path
    output_dir: Path | None = None
    recursive: bool = True
    video_extensions: tuple[str, ...] = field(default=DEFAULT_VIDEO_EXTENSIONS)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    thumbnail_size: tuple[int, int] = field(default=DEFAULT_THUMBNAIL_SIZE)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["videos_root"] = str(self.videos_root)
        data["output_dir"] = str(self.output_dir) if self.output_dir else None
        data["video_extensions"] = list(self.video_extensions)
        data["thumbnail_size"] = list(self.thumbnail_size)
        data["detection"] = asdict(self.detection)  # all primitives -> yaml-safe
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        detection_data = data.get("detection") or {}
        detection = DetectionConfig(
            **{k: v for k, v in detection_data.items() if k in DetectionConfig.__annotations__}
        )
        return cls(
            videos_root=Path(data.get("videos_root") or "").expanduser(),
            output_dir=Path(data["output_dir"]).expanduser() if data.get("output_dir") else None,
            recursive=bool(data.get("recursive", True)),
            video_extensions=tuple(data.get("video_extensions") or DEFAULT_VIDEO_EXTENSIONS),
            ffmpeg_path=data.get("ffmpeg_path") or None,
            ffprobe_path=data.get("ffprobe_path") or None,
            thumbnail_size=tuple(data.get("thumbnail_size") or DEFAULT_THUMBNAIL_SIZE),
            detection=detection,
        )


def default_settings_path() -> Path:
    return DEFAULT_SETTINGS_PATH


def _write_default(path: Path) -> None:
    settings = Settings(videos_root=Path.home())
    save_settings(settings, path)
    logger.info(f"Created default settings file at {path}")


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    if not path.exists():
        _write_default(path)
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        settings = Settings.from_dict(data)
    except (yaml.YAMLError, TypeError, ValueError, OSError) as exc:
        logger.error(f"Could not parse {path} ({exc}); using defaults for this session")
        settings = Settings(videos_root=Path.home())
    if not settings.videos_root.exists():
        logger.warning(f"Configured videos_root does not exist: {settings.videos_root}")
    return settings


def save_settings(settings: Settings, path: Path = DEFAULT_SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(settings.to_dict(), f, sort_keys=False)
