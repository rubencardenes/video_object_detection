from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QSpinBox

from video_tools.config.settings import Settings
from video_tools.core.ffmpeg_runner import build_resize_cmd
from video_tools.gui.central.ffmpeg_task_panel import FfmpegTaskPanel

PRESETS: dict[str, tuple[int, int] | None] = {
    "1080p (1920x1080)": (1920, 1080),
    "720p (1280x720)": (1280, 720),
    "480p (854x480)": (854, 480),
    "Custom": None,
}


class ResizePanel(FfmpegTaskPanel):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(settings, parent)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESETS.keys())
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(2, 7680)
        self._width_spin.setSingleStep(2)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(2, 4320)
        self._height_spin.setSingleStep(2)

        self.form_layout.addRow("Preset:", self._preset_combo)
        self.form_layout.addRow("Width:", self._width_spin)
        self.form_layout.addRow("Height:", self._height_spin)

        self._on_preset_changed(self._preset_combo.currentText())

    def _on_preset_changed(self, text: str) -> None:
        preset = PRESETS.get(text)
        is_custom = preset is None
        self._width_spin.setEnabled(is_custom)
        self._height_spin.setEnabled(is_custom)
        if preset:
            self._width_spin.setValue(preset[0])
            self._height_spin.setValue(preset[1])

    def output_suffix(self) -> str:
        return f"{self._height_spin.value()}p"

    def build_command(self, src: Path, dst: Path, ffmpeg: str) -> list[str]:
        # libx264 requires even dimensions.
        width = (self._width_spin.value() // 2) * 2
        height = (self._height_spin.value() // 2) * 2
        # Re-encoding to libx264 needs an mp4/mov-family container, not the source's own
        # (e.g. a .webm source can't hold an h264 stream).
        dst = dst.with_suffix(".mp4")
        return build_resize_cmd(src, dst, width, height, ffmpeg)
