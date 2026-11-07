from PySide6.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QSlider, QLabel,
                               QDialogButtonBox, QPushButton)
from PySide6.QtCore import Qt, Signal, QTimer


class AdjustDialog(QDialog):
    preview_requested = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Brightness / Contrast")
        self.setMinimumWidth(380)

        self.brightness = self._make_slider()
        self.contrast = self._make_slider()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(lambda: self.preview_requested.emit(*self.values()))

        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QFormLayout(self)
        layout.addRow("Brightness:", self.brightness[0])
        layout.addRow("Contrast:", self.contrast[0])
        layout.addRow(buttons)

    def _make_slider(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(0)
        label = QLabel("0")
        label.setMinimumWidth(32)
        slider.valueChanged.connect(lambda v: label.setText(str(v)))
        slider.valueChanged.connect(lambda _: self._timer.start())
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        return row, slider

    def values(self):
        return self.brightness[1].value(), self.contrast[1].value()

    def reset(self):
        self.brightness[1].setValue(0)
        self.contrast[1].setValue(0)