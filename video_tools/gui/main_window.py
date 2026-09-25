from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QSplitter, QStackedWidget, QVBoxLayout

from video_tools.config.settings import Settings
from video_tools.gui.central.convert_panel import ConvertPanel
from video_tools.gui.central.cut_panel import CutPanel
from video_tools.gui.central.fps_panel import FpsPanel
from video_tools.gui.central.info_panel import InfoPanel
from video_tools.gui.central.preview_panel import PreviewPanel
from video_tools.gui.central.resize_panel import ResizePanel
from video_tools.gui.sidebar import SidebarWidget


class MainWindow(QMainWindow):
    video_selected = Signal(Path)

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._current_video: Path | None = None

        self.setWindowTitle("Video Tools")
        self.resize(1100, 600)

        self._central_stack = QStackedWidget()
        placeholder = QLabel("Select a video or image sequence from the sidebar to get started.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._central_stack.addWidget(placeholder)

        self._preview_panel = PreviewPanel(settings)
        self._info_panel = InfoPanel(settings)
        self._convert_panel = ConvertPanel(settings)
        self._resize_panel = ResizePanel(settings)
        self._fps_panel = FpsPanel(settings)
        self._cut_panel = CutPanel(settings, self._preview_panel.player)
        self._preview_panel.register_panel("Convert", self._convert_panel)
        self._preview_panel.register_panel("Resize", self._resize_panel)
        self._preview_panel.register_panel("FPS", self._fps_panel)
        self._preview_panel.register_panel("Cut", self._cut_panel)
        self._preview_panel.info_clicked.connect(self._show_info_dialog)
        self._central_stack.addWidget(self._preview_panel)

        self._cut_panel.scrubber.in_point_changed.connect(self._on_trim_range_changed)
        self._cut_panel.scrubber.out_point_changed.connect(self._on_trim_range_changed)
        self._preview_panel.player.durationChanged.connect(self._on_trim_range_changed)

        self._info_dialog = QDialog(self)
        self._info_dialog.setWindowTitle("Video Info")
        self._info_dialog.resize(420, 380)
        info_dialog_layout = QVBoxLayout(self._info_dialog)
        info_dialog_layout.addWidget(self._info_panel)

        self._sidebar = SidebarWidget(settings)
        self._sidebar.video_selected.connect(self._on_video_selected)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._central_stack)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)

    def _on_video_selected(self, path: Path) -> None:
        self._current_video = path
        logger.info(f"Selected video: {path}")
        self._info_panel.load_video(path)
        if path.is_file():
            self._convert_panel.load_video(path)
            self._resize_panel.load_video(path)
            self._fps_panel.load_video(path)
            self._cut_panel.load_video(path)
        self._preview_panel.load_video(path)
        self._central_stack.setCurrentWidget(self._preview_panel)
        self.video_selected.emit(path)

    def _on_trim_range_changed(self, *_args) -> None:
        scrubber = self._cut_panel.scrubber
        self._preview_panel.set_trim_range(scrubber.in_point(), scrubber.out_point())

    def _show_info_dialog(self) -> None:
        self._info_dialog.show()
        self._info_dialog.raise_()
        self._info_dialog.activateWindow()
