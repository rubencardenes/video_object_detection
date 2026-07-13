from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from video_tools.config.settings import Settings
from video_tools.core.ffmpeg_runner import build_fps_cmd
from video_tools.gui.central.ffmpeg_task_panel import FfmpegTaskPanel

FPS_PRESETS = ["24", "30", "60", "Custom"]


class FpsPanel(FfmpegTaskPanel):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(settings, parent)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(FPS_PRESETS)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setDecimals(2)
        self._fps_spin.setValue(30.0)
        self._fps_spin.setEnabled(False)

        self.form_layout.addRow("Preset:", self._preset_combo)
        self.form_layout.addRow("FPS:", self._fps_spin)

        self._on_preset_changed(self._preset_combo.currentText())

    def _on_preset_changed(self, text: str) -> None:
        if text == "Custom":
            self._fps_spin.setEnabled(True)
        else:
            self._fps_spin.setEnabled(False)
            self._fps_spin.setValue(float(text))

    def output_suffix(self) -> str:
        return f"{self._fps_spin.value():g}fps"

    def build_command(self, src: Path, dst: Path, ffmpeg: str) -> list[str]:
        dst = dst.with_suffix(".mp4")
        return build_fps_cmd(src, dst, self._fps_spin.value(), ffmpeg)
