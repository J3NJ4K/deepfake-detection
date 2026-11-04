"""
"Colors" block of the ribbon, like Paint:

    (●)  (○)   ● ● ● ● ● ● ● ● ● ●    (colour wheel)
   Color Fill  ○ ● ● ● ● ● ● ● ● ●

- The two big circles are the two current colours:
    Color = brush, lines and shape outlines
    Fill  = fill tool (bucket)
  Click one to choose which colour the palette changes (it gets highlighted).
- Click a small circle to give that colour to the selected big circle.
- The colour wheel opens the full colour picker for the selected big circle.

The widget only shows colours and emits color_chosen(slot, rgb); MainWindow keeps them.
"""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPalette
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QGridLayout, QHBoxLayout, QLabel, QToolButton,
    QVBoxLayout, QWidget,
)

from ui.utils import load_icon

SLOT_COLOR = 0   # brush / outlines
SLOT_FILL = 1    # fill tool

# Classic Paint palette: first row dark colours, second row light colours
BASIC_COLORS = [
    [(0, 0, 0), (127, 127, 127), (136, 0, 21), (237, 28, 36), (255, 127, 39),
     (255, 242, 0), (34, 177, 76), (0, 162, 232), (63, 72, 204), (163, 73, 164)],
    [(255, 255, 255), (195, 195, 195), (185, 122, 87), (255, 174, 201), (255, 201, 14),
     (239, 228, 176), (181, 230, 29), (153, 217, 234), (112, 146, 190), (200, 191, 231)],
]


def circle_icon(rgb, diameter: int, border: QColor) -> QIcon:
    """A filled circle with a thin border (so white stays visible on a light theme)."""
    size = diameter + 2
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(border, 1))
    painter.setBrush(QColor(*rgb))
    painter.drawEllipse(1, 1, diameter - 1, diameter - 1)
    painter.end()
    return QIcon(pixmap)


class ColorPalette(QWidget):
    color_chosen = Signal(int, tuple)   # slot (SLOT_COLOR / SLOT_FILL), (r, g, b)

    def __init__(self, color=(0, 0, 0), fill=(255, 255, 255), parent=None):
        super().__init__(parent)
        self._colors = [color, fill]
        self._border = self.palette().color(QPalette.ColorRole.Mid)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- the two current colours (big circles), one of them selected
        self._slot_group = QButtonGroup(self)
        self._slot_group.setExclusive(True)
        self._slot_buttons = []
        for slot, name in ((SLOT_COLOR, "Color"), (SLOT_FILL, "Fill")):
            column = QVBoxLayout()
            column.setSpacing(1)
            button = QToolButton()
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setIconSize(QSize(30, 30))
            button.setToolTip("Color: brush, lines and shape outlines" if slot == SLOT_COLOR
                              else "Fill: color used by the fill tool")
            self._slot_group.addButton(button, slot)
            self._slot_buttons.append(button)
            column.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setEnabled(False)   # small grey text, like the group titles
            column.addWidget(label)
            layout.addLayout(column)
        self._slot_buttons[SLOT_COLOR].setChecked(True)

        # --- basic colours (small circles)
        grid = QGridLayout()
        grid.setSpacing(2)
        for row, colors in enumerate(BASIC_COLORS):
            for col, rgb in enumerate(colors):
                button = QToolButton()
                button.setAutoRaise(True)
                button.setIconSize(QSize(18, 18))
                button.setIcon(circle_icon(rgb, 18, self._border))
                button.setToolTip(f"RGB{rgb}")
                button.clicked.connect(lambda _=False, c=rgb: self._choose(c))
                grid.addWidget(button, row, col)
        layout.addLayout(grid)

        # --- colour wheel = full colour picker (its own colours are kept)
        self._wheel = QToolButton()
        self._wheel.setAutoRaise(True)
        self._wheel.setIconSize(QSize(30, 30))
        icon = load_icon("colour_wheel", keep_colors=True)
        if icon:
            self._wheel.setIcon(icon)
        else:
            self._wheel.setText("More...")
        self._wheel.setToolTip("More colors...")
        self._wheel.clicked.connect(self._open_wheel)
        layout.addWidget(self._wheel, 0, Qt.AlignmentFlag.AlignVCenter)

        self._refresh_slots()

    # ------------------------------------------------------------------
    def selected_slot(self) -> int:
        return self._slot_group.checkedId()

    def set_color(self, slot: int, rgb) -> None:
        """Show a colour in a big circle without emitting color_chosen."""
        self._colors[slot] = tuple(rgb)
        self._refresh_slots()

    def _refresh_slots(self) -> None:
        for slot, button in enumerate(self._slot_buttons):
            button.setIcon(circle_icon(self._colors[slot], 30, self._border))

    def _choose(self, rgb) -> None:
        slot = self.selected_slot()
        self.set_color(slot, rgb)
        self.color_chosen.emit(slot, tuple(rgb))

    def _open_wheel(self) -> None:
        slot = self.selected_slot()
        title = "Color" if slot == SLOT_COLOR else "Fill color"
        color = QColorDialog.getColor(QColor(*self._colors[slot]), self, title)
        if color.isValid():   # False if the user pressed Cancel
            self._choose((color.red(), color.green(), color.blue()))
