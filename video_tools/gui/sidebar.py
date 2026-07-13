from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import QSize, QThreadPool, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from video_tools.config.settings import Settings, save_settings
from video_tools.core.jobs import ThumbnailWorker
from video_tools.gui import icons


def scan_videos(root: Path, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    if not root.exists():
        return []
    pattern = "**/*" if recursive else "*"
    matches = [
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(matches)


class SidebarWidget(QWidget):
    video_selected = Signal(Path)

    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._videos: list[Path] = []
        self._thumbnail_cache: dict[Path, QIcon] = {}
        self._item_by_path: dict[Path, QListWidgetItem] = {}

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter...")
        self._filter_edit.textChanged.connect(self._apply_filter)

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.setIcon(icons.refresh_icon())
        self._refresh_button.clicked.connect(self.reload)

        self._change_folder_button = QPushButton("Change Folder...")
        self._change_folder_button.setIcon(icons.folder_icon())
        self._change_folder_button.clicked.connect(self._change_folder)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._list_widget = QListWidget()
        # Extraction uses settings.thumbnail_size for quality; the list row itself
        # needs a much smaller icon so the filename text stays readable next to it.
        self._list_widget.setIconSize(QSize(64, 36))
        self._list_widget.itemSelectionChanged.connect(self._on_selection_changed)

        top_row = QHBoxLayout()
        top_row.addWidget(self._filter_edit)
        top_row.addWidget(self._refresh_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._list_widget)
        layout.addWidget(self._change_folder_button)

        self.reload()

    def reload(self) -> None:
        root = self._settings.videos_root
        self._videos = scan_videos(root, self._settings.video_extensions, self._settings.recursive)
        if not root.exists():
            self._status_label.setText(f"Folder not found: {root}")
        elif not self._videos:
            self._status_label.setText(f"No videos found in {root}")
        else:
            self._status_label.setText("")
        self._apply_filter(self._filter_edit.text())
        self._start_thumbnail_jobs()

    def _apply_filter(self, text: str) -> None:
        self._list_widget.clear()
        self._item_by_path = {}
        needle = text.lower()
        root = self._settings.videos_root
        for path in self._videos:
            try:
                display = str(path.relative_to(root))
            except ValueError:
                display = str(path)
            if needle and needle not in display.lower():
                continue
            item = QListWidgetItem(display)
            icon = self._thumbnail_cache.get(path)
            if icon is not None:
                item.setIcon(icon)
            item.setToolTip(str(path))
            item.setData(1000, path)
            self._list_widget.addItem(item)
            self._item_by_path[path] = item

    def _start_thumbnail_jobs(self) -> None:
        for path in self._videos:
            if path in self._thumbnail_cache:
                continue
            worker = ThumbnailWorker(path, self._settings.thumbnail_size)
            worker.signals.done.connect(self._on_thumbnail_ready)
            QThreadPool.globalInstance().start(worker)

    def _on_thumbnail_ready(self, path: Path, image: QImage | None) -> None:
        if image is None:
            return
        icon = QIcon(QPixmap.fromImage(image))
        self._thumbnail_cache[path] = icon
        item = self._item_by_path.get(path)
        if item is not None:
            item.setIcon(icon)

    def _on_selection_changed(self) -> None:
        items = self._list_widget.selectedItems()
        if not items:
            return
        path = items[0].data(1000)
        self.video_selected.emit(path)

    def _change_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose videos folder", str(self._settings.videos_root)
        )
        if not directory:
            return
        self._settings.videos_root = Path(directory)
        save_settings(self._settings)
        logger.info(f"Videos root changed to {directory}")
        self.reload()
