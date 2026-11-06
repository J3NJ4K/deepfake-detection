"""
"New image" window:

    Preset:       [ Full HD — 1920 × 1080        ▾ ]
    Resolution:   [ 300 DPI ▾ ]          (only for paper formats)
    Width:        [ 1920 px ]
    Height:       [ 1080 px ]
    Orientation:  (•) Landscape  ( ) Portrait
    Background:   [ White ] [ Black ] [ Other... ]   ■
                                     [ Cancel ] [ OK ]

Typing a width or height switches the preset to "Custom".
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
)

MAX_SIZE = 10000  # pixels

# (name, width, height) in pixels
SCREEN_PRESETS = [
    ("HD", 1280, 720),
    ("Full HD", 1920, 1080),
    ("4K UHD", 3840, 2160),
    ("Square (social media)", 1080, 1080),
    ("Portrait (social media)", 1080, 1350),
    ("Story / phone", 1080, 1920),
    ("Classic 4:3", 1024, 768),
]

# (name, width, height) in millimetres, portrait
PAPER_PRESETS = [
    ("A3", 297, 420),
    ("A4", 210, 297),
    ("A5", 148, 210),
    ("A6", 105, 148),
    ("US Letter", 215.9, 279.4),
    ("US Legal", 215.9, 355.6),
]

DPI_CHOICES = [72, 150, 300]


def mm_to_px(mm: float, dpi: int) -> int:
    """Millimetres to pixels at a given resolution (1 inch = 25.4 mm)."""
    return max(1, round(mm / 25.4 * dpi))


def oriented(width: int, height: int, landscape: bool) -> tuple[int, int]:
    """Swap width and height if needed so that the result has the asked orientation."""
    if (width > height) != landscape and width != height:
        return height, width
    return width, height


class NewImageDialog(QDialog):
    def __init__(self, parent=None, width: int = 1920, height: int = 1080):
        super().__init__(parent)
        self.setWindowTitle("New image")
        self._color = (255, 255, 255)
        self._updating = False   # True while the code itself changes the fields

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        # --- preset formats
        self.preset = QComboBox()
        self.preset.addItem("Custom", None)
        self.preset.insertSeparator(self.preset.count())
        for name, w, h in SCREEN_PRESETS:
            self.preset.addItem(f"{name} — {w} × {h} px", ("px", w, h))
        self.preset.insertSeparator(self.preset.count())
        for name, w, h in PAPER_PRESETS:
            self.preset.addItem(f"{name} — {w:g} × {h:g} mm", ("mm", w, h))
        self.preset.currentIndexChanged.connect(self._on_preset_chosen)
        form.addRow("Preset:", self.preset)

        # --- resolution, used to convert paper sizes to pixels
        self.dpi = QComboBox()
        for dpi in DPI_CHOICES:
            label = f"{dpi} DPI" + {72: " (screen)", 150: " (draft print)", 300: " (print quality)"}[dpi]
            self.dpi.addItem(label, dpi)
        self.dpi.setCurrentIndex(DPI_CHOICES.index(300))
        self.dpi.currentIndexChanged.connect(self._apply_preset)
        form.addRow("Resolution:", self.dpi)

        # --- size in pixels (always editable)
        self.width_box = self._size_box(width)
        self.height_box = self._size_box(height)
        form.addRow("Width:", self.width_box)
        form.addRow("Height:", self.height_box)

        # --- orientation
        orientation = QHBoxLayout()
        self.landscape = QRadioButton("Landscape")
        self.portrait = QRadioButton("Portrait")
        group = QButtonGroup(self)
        group.addButton(self.landscape)
        group.addButton(self.portrait)
        (self.landscape if width >= height else self.portrait).setChecked(True)
        self.landscape.toggled.connect(self._apply_orientation)
        orientation.addWidget(self.landscape)
        orientation.addWidget(self.portrait)
        orientation.addStretch()
        form.addRow("Orientation:", orientation)

        # --- background colour
        background = QHBoxLayout()
        for text, rgb in (("White", (255, 255, 255)), ("Black", (0, 0, 0))):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, c=rgb: self._set_color(c))
            background.addWidget(button)
        other = QPushButton("Other...")
        other.clicked.connect(self._choose_color)
        background.addWidget(other)
        self.swatch = QLabel()
        self.swatch.setFixedSize(30, 24)
        self.swatch.setFrameShape(QFrame.Shape.Box)   # border, so white stays visible
        background.addWidget(self.swatch)
        background.addStretch()
        form.addRow("Background:", background)

        # --- OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._set_color(self._color)
        self._select_matching_preset(width, height)
        self._update_dpi_state()

    # ------------------------------------------------------------------
    def values(self) -> tuple[int, int, tuple[int, int, int]]:
        """(width, height, background colour) chosen by the user."""
        return self.width_box.value(), self.height_box.value(), self._color

    # ------------------------------------------------------------------
    def _size_box(self, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, MAX_SIZE)
        box.setSuffix(" px")
        box.setValue(value)
        box.valueChanged.connect(self._on_size_typed)
        return box

    def _set_size(self, width: int, height: int) -> None:
        self._updating = True
        self.width_box.setValue(min(width, MAX_SIZE))
        self.height_box.setValue(min(height, MAX_SIZE))
        self._updating = False

    def _apply_preset(self, *_):
        data = self.preset.currentData()
        self._update_dpi_state()
        if data is None:      # "Custom": keep the current numbers
            return
        unit, w, h = data
        if unit == "mm":
            dpi = self.dpi.currentData()
            w, h = mm_to_px(w, dpi), mm_to_px(h, dpi)
        self._set_size(*oriented(w, h, self.landscape.isChecked()))

    def _on_preset_chosen(self, *_):
        # A newly chosen format starts in its natural orientation
        # (portrait for paper, landscape for screens); the buttons can swap it after.
        data = self.preset.currentData()
        if data is not None:
            self._set_orientation(data[1] > data[2])
        self._apply_preset()

    def _set_orientation(self, landscape: bool) -> None:
        for button in (self.landscape, self.portrait):
            button.blockSignals(True)
        (self.landscape if landscape else self.portrait).setChecked(True)
        for button in (self.landscape, self.portrait):
            button.blockSignals(False)

    def _apply_orientation(self, *_):
        w, h = oriented(self.width_box.value(), self.height_box.value(), self.landscape.isChecked())
        self._set_size(w, h)

    def _on_size_typed(self, *_):
        if self._updating:
            return
        # The user typed a size by hand -> it is a custom size now
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(0)
        self.preset.blockSignals(False)
        self._update_dpi_state()

    def _select_matching_preset(self, width: int, height: int) -> None:
        for index in range(self.preset.count()):
            data = self.preset.itemData(index)
            if data and data[0] == "px" and {data[1], data[2]} == {width, height}:
                self.preset.blockSignals(True)
                self.preset.setCurrentIndex(index)
                self.preset.blockSignals(False)
                return

    def _update_dpi_state(self) -> None:
        data = self.preset.currentData()
        self.dpi.setEnabled(bool(data) and data[0] == "mm")

    def _set_color(self, rgb) -> None:
        self._color = tuple(rgb)
        pixmap = QPixmap(28, 22)
        pixmap.fill(QColor(*self._color))
        self.swatch.setPixmap(pixmap)

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self._color), self, "Background color")
        if color.isValid():
            self._set_color((color.red(), color.green(), color.blue()))
