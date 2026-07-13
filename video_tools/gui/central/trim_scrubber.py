from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QHBoxLayout, QPushButton, QWidget


class TrimScrubber(QWidget):
    """v1: in/out point spinboxes with "set from current playhead position" buttons.

    A custom-painted dual-handle range slider is deferred to a later polish pass.
    """

    in_point_changed = Signal(float)
    out_point_changed = Signal(float)

    def __init__(self, player: QMediaPlayer, parent: QWidget | None = None):
        super().__init__(parent)
        self._player = player

        self._in_spin = QDoubleSpinBox()
        self._in_spin.setDecimals(2)
        self._in_spin.setSuffix(" s")
        self._in_spin.valueChanged.connect(self.in_point_changed)

        self._out_spin = QDoubleSpinBox()
        self._out_spin.setDecimals(2)
        self._out_spin.setSuffix(" s")
        self._out_spin.valueChanged.connect(self.out_point_changed)

        set_in_button = QPushButton("Set from playhead")
        set_in_button.clicked.connect(self._set_in_from_position)
        set_out_button = QPushButton("Set from playhead")
        set_out_button.clicked.connect(self._set_out_from_position)

        in_row = QHBoxLayout()
        in_row.addWidget(self._in_spin)
        in_row.addWidget(set_in_button)

        out_row = QHBoxLayout()
        out_row.addWidget(self._out_spin)
        out_row.addWidget(set_out_button)

        layout = QFormLayout(self)
        layout.addRow("In point:", in_row)
        layout.addRow("Out point:", out_row)

    def set_max(self, duration_sec: float) -> None:
        self._in_spin.setRange(0, duration_sec)
        self._out_spin.setRange(0, duration_sec)
        if self._out_spin.value() == 0:
            self._out_spin.setValue(duration_sec)

    def in_point(self) -> float:
        return self._in_spin.value()

    def out_point(self) -> float:
        return self._out_spin.value()

    def _set_in_from_position(self) -> None:
        self._in_spin.setValue(self._player.position() / 1000.0)

    def _set_out_from_position(self) -> None:
        self._out_spin.setValue(self._player.position() / 1000.0)
