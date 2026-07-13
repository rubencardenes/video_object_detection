from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPen, QPixmap

# Hand-drawn with QPainter rather than emoji or a bundled font/image icon set:
# on this machine, color-emoji glyph rendering (Apple's CoreText/CG "sbix" bitmap
# path) crashes with SIGBUS inside CopyEmojiImage while painting widget text (see
# the crash reports in ~/Library/Logs/DiagnosticReports/python3.12-*.ips). Plain
# QPainter primitives never touch that code path, so this is both crash-safe and
# gives icons that actually match each action instead of generic OS dialog icons.

_COLOR = "#e6e6e6"
_SIZE = 20


def _new_painter(size: int) -> tuple[QPixmap, QPainter]:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_COLOR)
    pen.setWidthF(size * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    return pixmap, painter


def _finish(pixmap: QPixmap, painter: QPainter) -> QIcon:
    painter.end()
    return QIcon(pixmap)


def play_icon(size: int = _SIZE) -> QIcon:
    pixmap, painter = _new_painter(size)
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    m = size * 0.22
    triangle = QPainterPath()
    triangle.moveTo(m, m * 0.9)
    triangle.lineTo(size - m * 0.9, size / 2)
    triangle.lineTo(m, size - m * 0.9)
    triangle.closeSubpath()
    painter.drawPath(triangle)
    return _finish(pixmap, painter)


def pause_icon(size: int = _SIZE) -> QIcon:
    pixmap, painter = _new_painter(size)
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    bar_w = size * 0.18
    m = size * 0.22
    painter.drawRoundedRect(QRectF(m, m, bar_w, size - 2 * m), bar_w * 0.3, bar_w * 0.3)
    painter.drawRoundedRect(
        QRectF(size - m - bar_w, m, bar_w, size - 2 * m), bar_w * 0.3, bar_w * 0.3
    )
    return _finish(pixmap, painter)


def info_icon(size: int = _SIZE) -> QIcon:
    pixmap, painter = _new_painter(size)
    m = size * 0.12
    painter.drawEllipse(QRectF(m, m, size - 2 * m, size - 2 * m))
    painter.setBrush(painter.pen().color())
    dot_r = size * 0.06
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QPointF(size / 2, size * 0.32), dot_r, dot_r)
    painter.setPen(painter.pen())
    pen = QPen(_COLOR)
    pen.setWidthF(size * 0.1)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(size / 2, size * 0.47), QPointF(size / 2, size * 0.75))
    return _finish(pixmap, painter)


def convert_icon(size: int = _SIZE) -> QIcon:
    """Circular refresh arrow: a ~280-degree ring with one arrowhead."""
    pixmap, painter = _new_painter(size)
    painter.setBrush(Qt.GlobalColor.transparent)
    r = size * 0.32
    center = QPointF(size / 2, size / 2)
    rect = QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r)
    start_deg, span_deg = 50.0, 280.0
    painter.drawArc(rect, int(start_deg * 16), int(span_deg * 16))

    # Qt angles: 0 deg = 3 o'clock, positive = counter-clockwise; screen y grows
    # downward so the y-component of both position and tangent gets negated.
    end_rad = math.radians(start_deg + span_deg)
    ex = center.x() + r * math.cos(end_rad)
    ey = center.y() - r * math.sin(end_rad)
    tangent_rad = end_rad + math.pi / 2
    tx, ty = math.cos(tangent_rad), -math.sin(tangent_rad)
    nx, ny = -ty, tx  # perpendicular to the tangent, for the arrowhead's width

    head = size * 0.2
    tip = QPointF(ex + head * 0.5 * tx, ey + head * 0.5 * ty)
    back_left = QPointF(ex - head * 0.5 * tx + head * 0.4 * nx, ey - head * 0.5 * ty + head * 0.4 * ny)
    back_right = QPointF(ex - head * 0.5 * tx - head * 0.4 * nx, ey - head * 0.5 * ty - head * 0.4 * ny)

    path = QPainterPath()
    path.moveTo(tip)
    path.lineTo(back_left)
    path.lineTo(back_right)
    path.closeSubpath()
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPath(path)
    return _finish(pixmap, painter)


def cut_icon(size: int = _SIZE) -> QIcon:
    """Scissors: two ring handles converging to a blade point."""
    pixmap, painter = _new_painter(size)
    tip = QPointF(size * 0.85, size * 0.5)
    left_handle = QPointF(size * 0.22, size * 0.22)
    right_handle = QPointF(size * 0.22, size * 0.78)
    painter.drawLine(left_handle, tip)
    painter.drawLine(right_handle, tip)
    ring_r = size * 0.12
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.drawEllipse(left_handle, ring_r, ring_r)
    painter.drawEllipse(right_handle, ring_r, ring_r)
    return _finish(pixmap, painter)


def resize_icon(size: int = _SIZE) -> QIcon:
    """Diagonal double-headed arrow (expand/scale)."""
    pixmap, painter = _new_painter(size)
    start = QPointF(size * 0.22, size * 0.78)
    end = QPointF(size * 0.78, size * 0.22)
    painter.drawLine(start, end)
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    head = size * 0.15
    top_head = QPainterPath()
    top_head.moveTo(end.x(), end.y())
    top_head.lineTo(end.x() - head, end.y())
    top_head.lineTo(end.x(), end.y() + head)
    top_head.closeSubpath()
    painter.drawPath(top_head)
    bot_head = QPainterPath()
    bot_head.moveTo(start.x(), start.y())
    bot_head.lineTo(start.x() + head, start.y())
    bot_head.lineTo(start.x(), start.y() - head)
    bot_head.closeSubpath()
    painter.drawPath(bot_head)
    return _finish(pixmap, painter)


def fps_icon(size: int = _SIZE) -> QIcon:
    """Filmstrip: rounded rect with perforations along the edges."""
    pixmap, painter = _new_painter(size)
    m = size * 0.14
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.drawRoundedRect(QRectF(m, m * 0.6, size - 2 * m, size - 1.2 * m), 2, 2)
    hole = size * 0.07
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    for frac in (0.18, 0.42, 0.66, 0.9):
        y = m * 0.6 + frac * (size - 1.2 * m)
        painter.drawRect(QRectF(m * 0.35, y - hole / 2, hole, hole))
        painter.drawRect(QRectF(size - m * 0.35 - hole, y - hole / 2, hole, hole))
    return _finish(pixmap, painter)


def cancel_icon(size: int = _SIZE) -> QIcon:
    pixmap, painter = _new_painter(size)
    m = size * 0.26
    painter.drawLine(QPointF(m, m), QPointF(size - m, size - m))
    painter.drawLine(QPointF(size - m, m), QPointF(m, size - m))
    return _finish(pixmap, painter)


def refresh_icon(size: int = _SIZE) -> QIcon:
    return convert_icon(size)


def detect_icon(size: int = _SIZE) -> QIcon:
    """Four corner brackets, like a camera autofocus/viewfinder reticle."""
    pixmap, painter = _new_painter(size)
    painter.setBrush(Qt.GlobalColor.transparent)
    m = size * 0.16
    arm = size * 0.24
    corners = [(m, m, 1, 1), (size - m, m, -1, 1), (m, size - m, 1, -1), (size - m, size - m, -1, -1)]
    for x, y, dx, dy in corners:
        painter.drawLine(QPointF(x, y), QPointF(x + arm * dx, y))
        painter.drawLine(QPointF(x, y), QPointF(x, y + arm * dy))
    dot_r = size * 0.08
    painter.setBrush(painter.pen().color())
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QPointF(size / 2, size / 2), dot_r, dot_r)
    return _finish(pixmap, painter)


def folder_icon(size: int = _SIZE) -> QIcon:
    pixmap, painter = _new_painter(size)
    painter.setBrush(Qt.GlobalColor.transparent)
    path = QPainterPath()
    left = size * 0.15
    top = size * 0.32
    right = size * 0.85
    bottom = size * 0.78
    tab_w = size * 0.28
    tab_h = size * 0.1
    path.moveTo(left, bottom)
    path.lineTo(left, top + tab_h)
    path.lineTo(left + tab_w * 0.3, top + tab_h)
    path.lineTo(left + tab_w * 0.5, top)
    path.lineTo(left + tab_w, top)
    path.lineTo(left + tab_w + tab_w * 0.3, top + tab_h)
    path.lineTo(right, top + tab_h)
    path.lineTo(right, bottom)
    path.closeSubpath()
    painter.drawPath(path)
    return _finish(pixmap, painter)
