import sys

from loguru import logger
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication, QFileDialog

from video_tools.config.settings import default_settings_path, load_settings, save_settings
from video_tools.gui.main_window import MainWindow

DARK_STYLESHEET = """
QWidget { background-color: #000000; color: #e6e6e6; }
QMainWindow, QDialog { background-color: #000000; }
QLineEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar {
    background-color: #141414; color: #e6e6e6; border: 1px solid #333333;
}
QPushButton {
    background-color: #1c1c1c; color: #e6e6e6; border: 1px solid #3a3a3a;
    padding: 4px 10px; border-radius: 3px;
}
QPushButton:hover { background-color: #2a2a2a; }
QPushButton:pressed { background-color: #333333; }
QPushButton:disabled { color: #666666; border-color: #262626; }
QPushButton:checked { background-color: #2a6ed9; border-color: #2a6ed9; color: #ffffff; }
QPushButton:checked:hover { background-color: #3a7ee0; }
QListWidget::item:selected { background-color: #2a4d6e; }
QProgressBar::chunk { background-color: #2a6ed9; }
QSplitter::handle { background-color: #1a1a1a; }
"""


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    # Let in-flight thumbnail/probe QRunnables finish before Qt tears down their
    # QObject signals on shutdown; otherwise a worker mid-emit on quit can hit
    # "Signal source has been deleted".
    app.aboutToQuit.connect(lambda: QThreadPool.globalInstance().waitForDone(3000))

    settings_path = default_settings_path()
    first_run = not settings_path.exists()
    settings = load_settings(settings_path)

    if first_run:
        logger.info("First run detected, prompting for videos folder")
        directory = QFileDialog.getExistingDirectory(None, "Choose your videos folder")
        if directory:
            from pathlib import Path

            settings.videos_root = Path(directory)
            save_settings(settings, settings_path)

    window = MainWindow(settings)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
