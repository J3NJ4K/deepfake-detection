import numpy as np
from PySide6.QtWidgets import (QDialog, QFormLayout, QPlainTextEdit, QSpinBox, QCheckBox,
                               QDialogButtonBox, QHBoxLayout, QRadioButton, QButtonGroup,
                               QFontComboBox)
from PySide6.QtGui import QImage, QPainter, QFont, QFontMetrics, QColor, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QRectF, QRect, Signal

ALIGNMENTS = ("left", "center", "right")


class TextDialog(QDialog):
    changed = Signal()
    last = {"family": "", "size": 40, "bold": False, "italic": False, "underline": False, "align": "left"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add text")
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Type your text")
        self.text_edit.setFixedHeight(100)
        self.font_combo = QFontComboBox()
        if TextDialog.last["family"]:
            self.font_combo.setCurrentFont(QFont(TextDialog.last["family"]))
        self.font_combo.currentFontChanged.connect(lambda *_: self.changed.emit())
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 500)
        self.size_spin.setSuffix(" px")
        self.size_spin.setValue(TextDialog.last["size"])

        # Different styles
        self.bold_check = QCheckBox("Bold")
        self.italic_check = QCheckBox("Italic")
        self.underline_check = QCheckBox("Underline")
        style_row = QHBoxLayout()
        for check, key in ((self.bold_check, "bold"), (self.italic_check, "italic"),
                           (self.underline_check, "underline")):
            check.setChecked(TextDialog.last[key])
            check.toggled.connect(self.changed)
            style_row.addWidget(check)

        # Alignment of the lines
        self.align_group = QButtonGroup(self)
        align_row = QHBoxLayout()
        for align in ALIGNMENTS:
            button = QRadioButton(align.capitalize())
            button.setProperty("align", align)
            button.setChecked(align == TextDialog.last["align"])
            self.align_group.addButton(button)
            align_row.addWidget(button)
        self.align_group.buttonToggled.connect(lambda *_: self.changed.emit())

        self.text_edit.textChanged.connect(self.changed)
        self.size_spin.valueChanged.connect(self.changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        #Enter makes a new line
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.accept)

        layout = QFormLayout(self)
        layout.addRow("Text:", self.text_edit)
        layout.addRow("Font:", self.font_combo)
        layout.addRow("Size:", self.size_spin)
        layout.addRow("Style:", style_row)
        layout.addRow("Align:", align_row)
        layout.addRow(buttons)

    def current(self):
        style = {
            "family": self.font_combo.currentFont().family(),
            "bold": self.bold_check.isChecked(),
            "italic": self.italic_check.isChecked(),
            "underline": self.underline_check.isChecked(),
            "align": self.align_group.checkedButton().property("align"),
        }
        return self.text_edit.toPlainText(), self.size_spin.value(), style

    def values(self):
        text, size, style = self.current()
        TextDialog.last = {"size": size, **style}
        return text, size, style


def render_text_layer(text, x, y, size, color, width, height,
                      family="", bold=False, italic=False, underline=False, align="left"):
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)

    font = QFont(family) if family else QFont()
    font.setPixelSize(size)
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    flags = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
             "right": Qt.AlignmentFlag.AlignRight}[align]

    block = QFontMetrics(font).boundingRect(QRect(0, 0, 100000, 100000), int(flags), text)
    left = {"left": x, "center": x - block.width() / 2, "right": x - block.width()}[align]
    box = QRectF(left, y, block.width() + 1, block.height() + 1)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(*color))
    painter.drawText(box, flags | Qt.AlignmentFlag.AlignTop, text)
    painter.end()

    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
    alpha = np.ascontiguousarray(pixels[:, :width, 3])
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[:] = color
    return rgb, alpha