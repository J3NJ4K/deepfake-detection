from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QAbstractSpinBox, QFrame, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ui.utils import load_icon

MIN_SIZE, MAX_SIZE = 1, 200


class _OptionColumn(QFrame):
    value_changed = Signal(int)

    def __init__(self, icon_name, fallback_text, minimum, maximum, suffix, tooltip, parent=None):
        super().__init__(parent)
        self.setObjectName("OptionColumn")
        self.setToolTip(tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(6)

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = load_icon(icon_name)
        if icon:
            icon_label.setPixmap(icon.pixmap(QSize(20, 20)))
        else:
            icon_label.setText(fallback_text)
        layout.addWidget(icon_label)

        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(minimum, maximum)
        self.slider.setMinimumHeight(180)
        self.slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.slider, 1, Qt.AlignmentFlag.AlignHCenter)

        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSuffix(suffix)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(56)
        self.spin.valueChanged.connect(self._on_spin_changed)
        layout.addWidget(self.spin, 0, Qt.AlignmentFlag.AlignHCenter)

    def set_value(self, value: int) -> None:
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _on_slider_moved(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.value_changed.emit(value)

    def _on_spin_changed(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.value_changed.emit(value)


class ToolOptionsBar(QWidget):
    size_changed = Signal(int)
    opacity_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolOptionsBar")
        self.setStyleSheet("#OptionColumn { border: 1px solid palette(mid); border-radius: 6px; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.size_column = _OptionColumn("pen-size", "Size", MIN_SIZE, MAX_SIZE, " px",
                                         "Size of the brush, eraser and shape outlines")
        self.size_column.value_changed.connect(self.size_changed.emit)
        layout.addWidget(self.size_column)

        self.opacity_column = _OptionColumn("opacity", "Opacity", 1, 100, " %",
                                            "Opacity of the brush")
        self.opacity_column.value_changed.connect(self.opacity_changed.emit)
        layout.addWidget(self.opacity_column)

        self.hide()

    def set_size(self, size: int) -> None:
        self.size_column.set_value(size)

    def set_opacity(self, percent: int) -> None:
        self.opacity_column.set_value(percent)

    def show_for_tool(self, uses_size: bool, uses_opacity: bool) -> None:
        self.size_column.setVisible(uses_size)
        self.opacity_column.setVisible(uses_opacity)
        self.setVisible(uses_size or uses_opacity)
