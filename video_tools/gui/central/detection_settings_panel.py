from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
)

from video_tools.detection.config import DetectionConfig
from video_tools.detection.model import list_detection_models
from video_tools.detection.tracking import available_methods


class DetectionSettingsPanel(QWidget):
    """Form for the detect-every-N-frames / visual-tracking parameters.

    Emits config_changed(DetectionConfig) whenever the user edits a field, so the
    live pipeline can be updated and the values persisted to settings.yaml.
    """

    config_changed = Signal(DetectionConfig)

    def __init__(
        self,
        config: DetectionConfig,
        models_dir: Path,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._config = config
        self._emitting = True  # suppress signals while populating initial values

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(0, 120)
        self._interval_spin.setToolTip(
            "Frames to track between detections; 0 runs detection only, without trackers."
        )

        self._method_combo = QComboBox()
        methods = available_methods() or [config.method]
        self._method_combo.addItems(methods)

        self._model_combo = QComboBox()
        models = list_detection_models(models_dir)
        # Empty string entry = "use the first model found" (config.model_filename None).
        self._model_combo.addItem("(auto)", userData=None)
        for name in models:
            self._model_combo.addItem(name, userData=name)

        self._reacquire_spin = QSpinBox()
        self._reacquire_spin.setRange(0, 100)
        self._reacquire_spin.setSuffix(" %")
        self._reacquire_spin.setToolTip(
            "Re-run detection early when fewer than this % of objects are still tracked."
        )

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.01, 0.99)
        self._confidence_spin.setSingleStep(0.01)
        self._confidence_spin.setDecimals(2)

        form = QFormLayout(self)
        form.addRow("Track frames between detects", self._interval_spin)
        form.addRow("Tracking method", self._method_combo)
        form.addRow("Detection model", self._model_combo)
        form.addRow("Re-detect below", self._reacquire_spin)
        form.addRow("Confidence", self._confidence_spin)

        self._load(config)

        self._interval_spin.valueChanged.connect(self._on_changed)
        self._method_combo.currentTextChanged.connect(self._on_changed)
        self._model_combo.currentIndexChanged.connect(self._on_changed)
        self._reacquire_spin.valueChanged.connect(self._on_changed)
        self._confidence_spin.valueChanged.connect(self._on_changed)
        self._emitting = False

    def _load(self, config: DetectionConfig) -> None:
        self._config = config
        self._interval_spin.setValue(config.interval_frames)
        methods = [
            self._method_combo.itemText(i)
            for i in range(self._method_combo.count())
        ]
        if config.method in methods:
            self._method_combo.setCurrentText(config.method)
        model_index = self._model_combo.findData(config.model_filename)
        self._model_combo.setCurrentIndex(model_index if model_index >= 0 else 0)
        self._reacquire_spin.setValue(config.reacquire_pct)
        self._confidence_spin.setValue(config.confidence)

    def set_config(self, config: DetectionConfig) -> None:
        """Synchronize controls after the model is changed in the main window."""
        self._emitting = True
        self._load(config)
        self._emitting = False

    def _on_changed(self, *_args) -> None:
        if self._emitting:
            return
        self._config = replace(
            self._config,
            interval_frames=self._interval_spin.value(),
            method=self._method_combo.currentText(),
            model_filename=self._model_combo.currentData(),
            reacquire_pct=self._reacquire_spin.value(),
            confidence=round(self._confidence_spin.value(), 2),
        )
        self.config_changed.emit(self._config)
