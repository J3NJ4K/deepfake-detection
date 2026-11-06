# Text tool
import numpy as np
from PySide6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QSpinBox, QCheckBox,
                               QDialogButtonBox)
from PySide6.QtGui import QImage, QPainter, QFont, QFontMetrics, QColor
from PySide6.QtCore import Qt, QPointF


class TextDialog(QDialog):
    last_size = 40
    last_bold = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add text")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("Type your text")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 500)
        self.size_spin.setSuffix(" px")
        self.size_spin.setValue(TextDialog.last_size)
        self.bold_check = QCheckBox("Bold")
        self.bold_check.setChecked(TextDialog.last_bold)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QFormLayout(self)
        layout.addRow("Text:", self.text_edit)
        layout.addRow("Size:", self.size_spin)
        layout.addRow("", self.bold_check)
        layout.addRow(buttons)

    def values(self):
        TextDialog.last_size = self.size_spin.value()
        TextDialog.last_bold = self.bold_check.isChecked()
        return self.text_edit.text(), TextDialog.last_size, TextDialog.last_bold


def render_text_layer(text, x, y, size, color, width, height, bold=False):
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)

    font = QFont()
    font.setPixelSize(size)
    font.setBold(bold)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(*color))
    painter.drawText(QPointF(x, y + QFontMetrics(font).ascent()), text)
    painter.end()

    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
    alpha = np.ascontiguousarray(pixels[:, :width, 3])
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[:] = color
    return rgb, alpha