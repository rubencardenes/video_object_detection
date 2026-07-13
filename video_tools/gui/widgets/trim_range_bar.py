from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

_TRACK_COLOR = QColor("#262626")
_RANGE_COLOR = QColor("#e0a030")


class TrimRangeBar(QWidget):
    """Thin bar under the position slider highlighting the current cut in/out range.

    Purely cosmetic and decoupled from QSlider's own painting (subclassing QSlider's
    paintEvent to draw over the native groove is style-fragile); this instead sits
    in its own row, sized to line up under the slider above it.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(5)
        self._duration_ms = 0
        self._start_ms = 0.0
        self._end_ms = 0.0

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, duration_ms)
        self.update()

    def set_range(self, start_ms: float, end_ms: float) -> None:
        self._start_ms = start_ms
        self._end_ms = end_ms
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_TRACK_COLOR)
        painter.drawRoundedRect(self.rect(), 2, 2)

        if self._duration_ms <= 0:
            return

        width = self.width()
        x1 = int(width * (self._start_ms / self._duration_ms))
        x2 = int(width * (self._end_ms / self._duration_ms))
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        if x2 <= x1:
            return

        painter.setBrush(_RANGE_COLOR)
        painter.drawRoundedRect(x1, 0, x2 - x1, self.height(), 2, 2)
