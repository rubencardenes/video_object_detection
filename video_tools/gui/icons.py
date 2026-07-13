from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# Icons are rendered from Lucide (https://lucide.dev, ISC license) SVG path data,
# embedded below so the app needs no icon files or network access. We render the
# SVG via QtSvg rather than drawing emoji/font glyphs: on this machine Apple's
# color-emoji glyph path (CoreText/CG "sbix") crashes with SIGBUS inside
# CopyEmojiImage while painting widget text (see crash reports under
# ~/Library/Logs/DiagnosticReports/python3.12-*.ips). QtSvg never touches that
# code path, so this is both crash-safe and gives a clean, consistent icon set.

_COLOR = "#e6e6e6"
_SIZE = 20

# Lucide icons are authored on a 24x24 grid with a 2px round stroke.
_SVG_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
    'fill="{fill}" stroke="{stroke}" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)

# name -> inner SVG elements, taken verbatim from the Lucide source icons.
_BODIES = {
    "play": '<path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z"/>',
    "pause": '<rect x="14" y="3" width="5" height="18" rx="1"/><rect x="5" y="3" width="5" height="18" rx="1"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "rotate-cw": '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/><path d="M14.8 14.8 20 20"/>',
    "maximize-2": '<path d="M15 3h6v6"/><path d="m21 3-7 7"/><path d="m3 21 7-7"/><path d="M9 21H3v-6"/>',
    "film": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
    "settings": '<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
}


def _render(name: str, size: int, *, filled: bool = False) -> QIcon:
    body = _BODIES[name]
    svg = _SVG_TEMPLATE.format(fill=_COLOR if filled else "none", stroke=_COLOR, body=body)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    # Render at 2x physical resolution, filling the whole pixmap via an explicit
    # target rect, then tag it 2x so it displays crisp on retina. (Setting the
    # device-pixel-ratio *before* rendering would double-apply the scale and clip
    # the icon.)
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


# Purpose-built application icon (Dock / window / About). Unlike the toolbar
# glyphs above this is a full-bleed "app tile": a rounded accent-blue square with
# a white play triangle, so macOS shows something distinctive instead of the
# generic document icon a bare Python script otherwise gets.
_APP_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" '
    'viewBox="0 0 1024 1024">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#3a7ee0"/><stop offset="1" stop-color="#1f5bc0"/>'
    "</linearGradient></defs>"
    '<rect x="64" y="64" width="896" height="896" rx="200" fill="url(#g)"/>'
    '<path d="M415 320a24 24 0 0 1 36-20.8l300 172a24 24 0 0 1 0 41.6l-300 172'
    'A24 24 0 0 1 415 656z" fill="#ffffff"/>'
    "</svg>"
)


def app_icon() -> QIcon:
    """Distinctive application/Dock icon rendered from an embedded SVG."""
    renderer = QSvgRenderer(QByteArray(_APP_ICON_SVG.encode("utf-8")))
    icon = QIcon()
    for px in (16, 32, 64, 128, 256, 512, 1024):
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, px, px))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def play_icon(size: int = _SIZE) -> QIcon:
    return _render("play", size, filled=True)


def pause_icon(size: int = _SIZE) -> QIcon:
    return _render("pause", size, filled=True)


def info_icon(size: int = _SIZE) -> QIcon:
    return _render("info", size)


def convert_icon(size: int = _SIZE) -> QIcon:
    return _render("repeat", size)


def cut_icon(size: int = _SIZE) -> QIcon:
    return _render("scissors", size)


def resize_icon(size: int = _SIZE) -> QIcon:
    return _render("maximize-2", size)


def fps_icon(size: int = _SIZE) -> QIcon:
    return _render("film", size)


def cancel_icon(size: int = _SIZE) -> QIcon:
    return _render("x", size)


def refresh_icon(size: int = _SIZE) -> QIcon:
    return _render("rotate-cw", size)


def detect_icon(size: int = _SIZE) -> QIcon:
    return _render("scan", size)


def settings_icon(size: int = _SIZE) -> QIcon:
    return _render("settings", size)


def folder_icon(size: int = _SIZE) -> QIcon:
    return _render("folder", size)
