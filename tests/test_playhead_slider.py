from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from video_tools.gui.widgets.playhead_slider import PlayheadSlider  # noqa: E402


class PlayheadSliderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.slider = PlayheadSlider()
        self.slider.resize(212, 20)
        self.slider.setRange(0, 1_000)

    def test_click_uses_absolute_timeline_position(self) -> None:
        QTest.mouseClick(
            self.slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(self.slider.width() // 2, self.slider.height() // 2),
        )

        self.assertEqual(self.slider.value(), 500)

    def test_drag_changes_position_and_emits_finished(self) -> None:
        positions: list[int] = []
        finished: list[bool] = []
        self.slider.seek_requested.connect(positions.append)
        self.slider.seek_finished.connect(lambda: finished.append(True))

        QTest.mousePress(
            self.slider, Qt.MouseButton.LeftButton, pos=QPoint(6, 10)
        )
        QTest.mouseRelease(
            self.slider, Qt.MouseButton.LeftButton, pos=QPoint(206, 10)
        )

        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 1_000)
        self.assertEqual(self.slider.value(), 1_000)
        self.assertEqual(finished, [True])

    def test_can_render_playhead(self) -> None:
        self.slider.setValue(500)

        self.assertFalse(self.slider.grab().toImage().isNull())


if __name__ == "__main__":
    unittest.main()
