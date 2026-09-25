from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QSlider, QWidget


_GROOVE_COLOR = QColor("#3a3a3a")
_ELAPSED_COLOR = QColor("#2a6ed9")
_HANDLE_COLOR = QColor("#ffffff")
_DISABLED_COLOR = QColor("#666666")


class PlayheadSlider(QSlider):
    """Horizontal timeline with an explicit, directly draggable playhead."""

    seek_requested = Signal(int)
    seek_finished = Signal()

    _HANDLE_RADIUS = 6
    _GROOVE_HEIGHT = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setMinimumHeight(self._HANDLE_RADIUS * 2 + 4)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        radius = self._HANDLE_RADIUS
        left = float(radius)
        right = float(max(radius, self.width() - radius))
        center_y = self.height() / 2.0
        groove_y = center_y - self._GROOVE_HEIGHT / 2.0
        handle_x = self._handle_x(left, right)

        painter.setBrush(_GROOVE_COLOR if self.isEnabled() else _DISABLED_COLOR)
        painter.drawRoundedRect(
            QRectF(left, groove_y, right - left, self._GROOVE_HEIGHT), 2, 2
        )
        if self.isEnabled() and handle_x > left:
            painter.setBrush(_ELAPSED_COLOR)
            painter.drawRoundedRect(
                QRectF(left, groove_y, handle_x - left, self._GROOVE_HEIGHT), 2, 2
            )

        painter.setBrush(_HANDLE_COLOR if self.isEnabled() else _DISABLED_COLOR)
        painter.drawEllipse(QPointF(handle_x, center_y), radius, radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        self.setSliderDown(True)
        self._seek_to_x(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.isSliderDown():
            super().mouseMoveEvent(event)
            return
        self._seek_to_x(event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.isSliderDown():
            super().mouseReleaseEvent(event)
            return
        self._seek_to_x(event.position().x())
        self.setSliderDown(False)
        self.seek_finished.emit()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        previous = self.value()
        super().keyPressEvent(event)
        if self.value() != previous:
            self.seek_requested.emit(self.value())
            self.seek_finished.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        previous = self.value()
        super().wheelEvent(event)
        if self.value() != previous:
            self.seek_requested.emit(self.value())
            self.seek_finished.emit()

    def _seek_to_x(self, x: float) -> None:
        radius = self._HANDLE_RADIUS
        available = max(1, self.width() - 2 * radius)
        ratio = max(0.0, min(1.0, (x - radius) / available))
        if self.layoutDirection() == Qt.LayoutDirection.RightToLeft:
            ratio = 1.0 - ratio
        value = round(self.minimum() + ratio * (self.maximum() - self.minimum()))
        self.setValue(value)
        self.seek_requested.emit(value)

    def _handle_x(self, left: float, right: float) -> float:
        span = self.maximum() - self.minimum()
        ratio = (self.value() - self.minimum()) / span if span else 0.0
        if self.layoutDirection() == Qt.LayoutDirection.RightToLeft:
            ratio = 1.0 - ratio
        return left + ratio * (right - left)
