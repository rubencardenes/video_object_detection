from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QFormLayout, QLabel, QWidget

from video_tools.config.settings import Settings
from video_tools.core.jobs import Worker
from video_tools.core.metadata import VideoInfo, probe_video


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class InfoPanel(QWidget):
    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._path: Path | None = None

        self._fields = {
            "path": QLabel(),
            "duration": QLabel(),
            "resolution": QLabel(),
            "fps": QLabel(),
            "codec": QLabel(),
            "format": QLabel(),
            "bit_rate": QLabel(),
            "size": QLabel(),
            "created": QLabel(),
        }
        for label in self._fields.values():
            label.setWordWrap(True)

        layout = QFormLayout(self)
        layout.addRow("Path:", self._fields["path"])
        layout.addRow("Duration:", self._fields["duration"])
        layout.addRow("Resolution:", self._fields["resolution"])
        layout.addRow("FPS:", self._fields["fps"])
        layout.addRow("Codec:", self._fields["codec"])
        layout.addRow("Format:", self._fields["format"])
        layout.addRow("Bit rate:", self._fields["bit_rate"])
        layout.addRow("Size:", self._fields["size"])
        layout.addRow("Created:", self._fields["created"])

    def load_video(self, path: Path) -> None:
        self._path = path
        for label in self._fields.values():
            label.setText("Loading...")

        worker = Worker(probe_video, path, self._settings.ffprobe_path)
        worker.signals.result.connect(self._on_info_loaded)
        worker.signals.error.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    def _on_info_loaded(self, info: VideoInfo) -> None:
        if info.path != self._path:
            return  # a newer selection has since superseded this result
        self._fields["path"].setText(str(info.path))
        self._fields["duration"].setText(_format_duration(info.duration_sec))
        self._fields["resolution"].setText(f"{info.width} x {info.height}")
        self._fields["fps"].setText(f"{info.fps:.2f}")
        self._fields["codec"].setText(info.codec_name)
        self._fields["format"].setText(info.format_name)
        self._fields["bit_rate"].setText(
            f"{info.bit_rate // 1000} kbps" if info.bit_rate else "unknown"
        )
        self._fields["size"].setText(_format_size(info.size_bytes))
        self._fields["created"].setText(info.created.strftime("%Y-%m-%d %H:%M"))

    def _on_error(self, message: str) -> None:
        for label in self._fields.values():
            label.setText("")
        self._fields["path"].setText(f"Error: {message}")
