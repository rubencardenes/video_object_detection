from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from video_tools.config.settings import Settings
from video_tools.core.ffmpeg_runner import build_convert_cmd
from video_tools.gui.central.ffmpeg_task_panel import FfmpegTaskPanel


class ConvertPanel(FfmpegTaskPanel):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(settings, parent)
        self.form_layout.addRow(
            "Convert to:", QLabel("MP4 (H.264 video / AAC audio)")
        )

    def output_suffix(self) -> str:
        return "converted"

    def build_command(self, src: Path, dst: Path, ffmpeg: str) -> list[str]:
        dst = dst.with_suffix(".mp4")
        return build_convert_cmd(src, dst, ffmpeg)
