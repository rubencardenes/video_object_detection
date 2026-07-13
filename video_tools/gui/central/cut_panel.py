from __future__ import annotations

from pathlib import Path

from PySide6.QtMultimedia import QMediaPlayer

from video_tools.config.settings import Settings
from video_tools.core.ffmpeg_runner import build_cut_cmd
from video_tools.core.metadata import VideoInfo
from video_tools.gui.central.ffmpeg_task_panel import FfmpegTaskPanel
from video_tools.gui.central.trim_scrubber import TrimScrubber


class CutPanel(FfmpegTaskPanel):
    def __init__(self, settings: Settings, player: QMediaPlayer, parent=None):
        super().__init__(settings, parent)
        self._player = player
        self._scrubber = TrimScrubber(player)
        self.form_layout.addRow(self._scrubber)
        player.durationChanged.connect(self._on_duration_changed)

    @property
    def scrubber(self) -> TrimScrubber:
        return self._scrubber

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._scrubber.set_max(duration_ms / 1000.0)

    def output_suffix(self) -> str:
        return "cut"

    def job_duration_sec(self, info: VideoInfo) -> float:
        return max(0.0, self._scrubber.out_point() - self._scrubber.in_point())

    def build_command(self, src: Path, dst: Path, ffmpeg: str) -> list[str]:
        # Stream copy (-c copy) keeps the source's own container/codec, so no
        # extension override is needed here (unlike Convert/Resize/FPS, which
        # re-encode to libx264 and therefore need an mp4/mov-family container).
        return build_cut_cmd(src, dst, self._scrubber.in_point(), self._scrubber.out_point(), ffmpeg)
