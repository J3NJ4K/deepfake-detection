"""
Small window to set the strength of a filter, with a live preview on the image:

    Radius:  ━━━━●━━━━━━  [ 2 px ]
    [x] Preview
                    [ Cancel ] [ OK ]

The window does not compute anything itself: it emits preview_requested(value) and
MainWindow shows the filtered image. Moving the slider quickly only triggers one
preview once it stops for a moment, so big images stay smooth.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout,
)

PREVIEW_DELAY_MS = 120


class FilterDialog(QDialog):
    preview_requested = Signal(int)   # value to preview
    preview_cleared = Signal()        # show the image without the filter

    def __init__(self, parent, title: str, label: str, minimum: int, maximum: int,
                 value: int, suffix: str = "", hint: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setWordWrap(True)
            hint_label.setEnabled(False)   # grey, smaller importance
            layout.addWidget(hint_label)

        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        row.addWidget(self.slider, 1)
        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.spin.setKeyboardTracking(False)   # apply a typed value on Enter
        row.addWidget(self.spin)
        layout.addLayout(row)

        self.preview_box = QCheckBox("Preview")
        self.preview_box.setChecked(True)
        layout.addWidget(self.preview_box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wait a little after the last change before computing the preview
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(PREVIEW_DELAY_MS)
        self._timer.timeout.connect(self._emit_preview)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.preview_box.toggled.connect(self._on_preview_toggled)

        # First preview as soon as the window opens
        QTimer.singleShot(0, self._emit_preview)

    def value(self) -> int:
        return self.spin.value()

    # ------------------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._timer.start()

    def _on_spin(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._timer.start()

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._emit_preview()
        else:
            self._timer.stop()
            self.preview_cleared.emit()

    def _emit_preview(self) -> None:
        if self.preview_box.isChecked():
            self.preview_requested.emit(self.value())
