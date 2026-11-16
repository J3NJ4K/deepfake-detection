import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QAbstractTextDocumentLayout, QColor, QFont, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap, QShortcut, QTextOption)
from PySide6.QtWidgets import (QButtonGroup, QColorDialog, QFontComboBox, QFrame, QGraphicsRectItem,
                               QGraphicsTextItem, QHBoxLayout, QSpinBox, QToolButton)

from ui.utils import load_icon

MIN_SIZE, MAX_SIZE = 6, 500
ALIGNMENTS = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
              "right": Qt.AlignmentFlag.AlignRight}
BAR_MARGIN = 10


class TextEditor(QObject):
    finished = Signal(bool)

    #Last settings, re-used for the next text (font, size, styles, alignment)
    last = {"family": "", "size": 40, "bold": False, "italic": False, "underline": False,
            "strikeout": False, "align": "left"}

    def __init__(self, canvas, pos: QPointF, color, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.color = tuple(color)
        self.style = dict(TextEditor.last)

        #the text itself, typed directly on the image
        self.item = QGraphicsTextItem()
        self.item.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.item.document().setDocumentMargin(0)
        self.item.setPos(pos)
        self.item.setZValue(5)
        self.frame = QGraphicsRectItem()
        pen = QPen(QColor(0, 120, 215), 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.frame.setPen(pen)
        self.frame.setZValue(5)
        canvas.scene().addItem(self.item)
        canvas.scene().addItem(self.frame)
        self.item.document().contentsChanged.connect(self._update_layout)
        canvas.text_item = self.item

        #floating bar
        self.bar = self._make_bar()
        canvas.installEventFilter(self)
        self._apply_style()
        self._place_bar()
        self.bar.show()
        self.bar.raise_()
        self.focus_text()

    # Floating bar
    def _make_bar(self) -> QFrame:
        bar = QFrame(self.canvas)
        bar.setObjectName("TextToolbar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet("#TextToolbar { background: palette(window); border: 1px solid palette(mid);"
                          " border-radius: 8px; }")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(3)

        # Font family
        self.font_box = QFontComboBox()
        self.font_box.setMaximumWidth(190)
        if self.style["family"]:
            self.font_box.setCurrentFont(QFont(self.style["family"]))
        else:
            self.style["family"] = self.font_box.currentFont().family()
        self.font_box.currentFontChanged.connect(self._on_font_chosen)
        row.addWidget(self.font_box)

        # Adjusting the size using the - and + buttons, or type a number directly.
        minus = self._button("−", "Smaller text", auto_repeat=True)
        minus.clicked.connect(lambda: self._step_size(-1))
        self.size_box = QSpinBox()
        self.size_box.setRange(MIN_SIZE, MAX_SIZE)
        self.size_box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.size_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_box.setFixedWidth(46)
        self.size_box.setToolTip("Text size (pixels)")
        self.size_box.setValue(self.style["size"])
        self.size_box.valueChanged.connect(self._on_size_changed)
        self.size_box.editingFinished.connect(self.focus_text)
        plus = self._button("+", "Bigger text", auto_repeat=True)
        plus.clicked.connect(lambda: self._step_size(1))
        for widget in (minus, self.size_box, plus):
            row.addWidget(widget)

        # Show an "A" with a line underneath indicating the current color
        self.color_button = self._button("", "Text color")
        self.color_button.setIconSize(QSize(20, 20))
        self.color_button.clicked.connect(self._choose_color)
        row.addWidget(self.color_button)
        row.addWidget(self._separator())

        # Bold / italic / underline / strikethrough
        self.style_buttons = {}
        self._shortcuts = []
        for key, icon, text, tip, shortcut in (
                ("bold", "bold", "B", "Bold (Ctrl+B)", QKeySequence.StandardKey.Bold),
                ("italic", "italic", "I", "Italic (Ctrl+I)", QKeySequence.StandardKey.Italic),
                ("underline", "underline", "U", "Underline (Ctrl+U)", QKeySequence.StandardKey.Underline),
                ("strikeout", "strikethrough", "S", "Strikethrough", None)):
            button = self._button(text, tip, icon, checkable=True)
            button.setChecked(self.style[key])
            button.toggled.connect(lambda checked, k=key: self._set_style(k, checked))
            self.style_buttons[key] = button
            row.addWidget(button)
            if shortcut is not None:
                keys = QShortcut(QKeySequence(shortcut), self.canvas)
                keys.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                keys.activated.connect(button.toggle)
                self._shortcuts.append(keys)
        row.addWidget(self._separator())

        # Alignment of the lines
        self.align_group = QButtonGroup(bar)
        for align, icon, text in (("left", "align-left", "Left"), ("center", "align-center", "Center"),
                                  ("right", "align-right", "Right")):
            button = self._button(text, f"Align {align}", icon, checkable=True)
            button.setChecked(align == self.style["align"])
            button.toggled.connect(lambda checked, a=align: checked and self._set_style("align", a))
            self.align_group.addButton(button)
            row.addWidget(button)
        row.addWidget(self._separator())

        # Apply / cancel
        apply_button = self._button("✓", "Apply the text (Esc, or click outside it)")
        apply_button.clicked.connect(lambda: self.finished.emit(True))
        cancel_button = self._button("✕", "Cancel the text")
        cancel_button.clicked.connect(lambda: self.finished.emit(False))
        row.addWidget(apply_button)
        row.addWidget(cancel_button)
        bar.adjustSize()
        return bar

    def _button(self, text, tooltip, icon_name=None, checkable=False, auto_repeat=False) -> QToolButton:
        button = QToolButton()
        button.setAutoRaise(True)
        button.setCheckable(checkable)
        button.setAutoRepeat(auto_repeat)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip(tooltip)
        icon = load_icon(icon_name) if icon_name else None
        if icon:
            button.setIcon(icon)
            button.setIconSize(QSize(18, 18))
        else:
            button.setText(text)
            if text in ("B", "I", "U", "S"):
                font = button.font()
                font.setBold(text == "B")
                font.setItalic(text == "I")
                font.setUnderline(text == "U")
                font.setStrikeOut(text == "S")
                button.setFont(font)
        return button

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setEnabled(False)
        return line

    def _place_bar(self) -> None:
        self.bar.adjustSize()
        x = max(0, (self.canvas.width() - self.bar.width()) // 2)
        self.bar.move(x, BAR_MARGIN)

    def eventFilter(self, watched, event):
        if watched is self.canvas and event.type() == QEvent.Type.Resize:
            self._place_bar()
        return False

    # Settings
    def _on_font_chosen(self, font: QFont) -> None:
        self._set_style("family", font.family())
        self.focus_text()

    def _on_size_changed(self, value: int) -> None:
        self._set_style("size", value)

    def _step_size(self, direction: int) -> None:
        size = self.size_box.value()
        step = max(1, round(size * 0.1))
        self.size_box.setValue(size + direction * step)

    def _set_style(self, key, value) -> None:
        self.style[key] = value
        self._apply_style()

    def set_color(self, rgb) -> None:
        self.color = tuple(rgb)
        self._apply_style()

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self.color), self.canvas.window(), "Text color")
        if color.isValid():
            self.set_color((color.red(), color.green(), color.blue()))
        self.focus_text()

    def _color_icon(self) -> QPixmap:
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.canvas.font())
        font.setBold(True)
        font.setPixelSize(14)
        painter.setFont(font)
        painter.setPen(self.canvas.palette().color(QPalette.ColorRole.WindowText))
        painter.drawText(QRectF(0, 0, 20, 15), Qt.AlignmentFlag.AlignCenter, "A")
        painter.fillRect(QRectF(2, 16, 16, 4), QColor(*self.color))
        painter.setPen(self.canvas.palette().color(QPalette.ColorRole.Mid))
        painter.drawRect(QRectF(2, 16, 16, 4))
        painter.end()
        return pixmap

    def _font(self) -> QFont:
        font = QFont(self.style["family"])
        font.setPixelSize(self.style["size"])
        font.setBold(self.style["bold"])
        font.setItalic(self.style["italic"])
        font.setUnderline(self.style["underline"])
        font.setStrikeOut(self.style["strikeout"])
        return font

    def _apply_style(self) -> None:
        self.item.setFont(self._font())
        self.item.setDefaultTextColor(QColor(*self.color))
        option = QTextOption(ALIGNMENTS[self.style["align"]])
        option.setWrapMode(QTextOption.WrapMode.NoWrap)
        self.item.document().setDefaultTextOption(option)
        self.color_button.setIcon(self._color_icon())
        self._update_layout()

    def _update_layout(self) -> None:
        document = self.item.document()
        document.setTextWidth(-1)
        document.setTextWidth(document.idealWidth() + 1)
        rect = self.item.sceneBoundingRect()
        pad = 3 / max(self.canvas.transform().m11(), 0.01)
        width = max(rect.width(), self.style["size"] * 0.5)
        self.frame.setRect(QRectF(rect.left() - pad, rect.top() - pad, width + 2 * pad, rect.height() + 2 * pad))

    def focus_text(self) -> None:
        self.canvas.setFocus()
        self.item.setFocus()


    # End
    def text(self) -> str:
        return self.item.toPlainText()

    def render(self, width: int, height: int):
        if not self.text().strip():
            return None
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.translate(self.item.pos())
        context = QAbstractTextDocumentLayout.PaintContext()
        context.palette.setColor(QPalette.ColorRole.Text, QColor(*self.color))
        self.item.document().documentLayout().draw(painter, context)
        painter.end()

        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
        alpha = np.ascontiguousarray(pixels[:, :width, 3])
        if not alpha.any():
            return None
        rgb = np.empty((height, width, 3), np.uint8)
        rgb[:] = self.color
        return rgb, alpha

    def close(self) -> None:
        TextEditor.last = dict(self.style)
        self.canvas.removeEventFilter(self)
        for keys in self._shortcuts:
            keys.setEnabled(False)
            keys.deleteLater()
        self.canvas.text_item = None
        scene = self.canvas.scene()
        scene.removeItem(self.item)
        scene.removeItem(self.frame)
        self.bar.hide()
        self.bar.deleteLater()
