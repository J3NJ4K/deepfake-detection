"""
Layers panel (shown on the right when the "Layers" button is pressed):

    [ + ] [ - ]
    [x] [thumb] Layer 2      <- top layer first, like Paint / Photoshop
    [x] [thumb] Layer 1
    [x] [thumb] Background
    Opacity  ━━━━●━━  100%

The panel only displays the layers and emits signals; MainWindow does the work.
Indexes in the signals are indexes in the LayerStack (0 = bottom layer).
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QSlider, QToolButton,
    QVBoxLayout, QWidget, QAbstractItemView,
)

from ui.utils import load_icon

THUMB_W, THUMB_H = 48, 36
THUMB_SIZE = QSize(THUMB_W, THUMB_H)


def layer_thumbnail(rgb: np.ndarray, alpha: np.ndarray) -> QIcon:
    """Small preview of a layer, drawn over a checkerboard so transparency is visible."""
    # Downscale with NumPy first (keep about 2x the thumbnail size), so big images stay fast
    step = max(1, min(alpha.shape[0] // (2 * THUMB_H),
                      alpha.shape[1] // (2 * THUMB_W)))
    rgba = np.ascontiguousarray(np.dstack([rgb[::step, ::step], alpha[::step, ::step]]))
    h, w = rgba.shape[:2]
    image = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    scaled = image.scaled(THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    thumb = QPixmap(THUMB_SIZE)
    thumb.fill(Qt.GlobalColor.transparent)
    painter = QPainter(thumb)
    x = (THUMB_SIZE.width() - scaled.width()) // 2
    y = (THUMB_SIZE.height() - scaled.height()) // 2
    cell = 6
    for cy in range(0, scaled.height(), cell):          # checkerboard
        for cx in range(0, scaled.width(), cell):
            light = ((cx // cell) + (cy // cell)) % 2 == 0
            painter.fillRect(x + cx, y + cy, cell, cell, QColor(200, 200, 200) if light else QColor(150, 150, 150))
    painter.drawImage(x, y, scaled)
    painter.end()
    return QIcon(thumb)


class LayersPanel(QWidget):
    add_requested = Signal()
    delete_requested = Signal()
    active_changed = Signal(int)            # layer index
    visibility_changed = Signal(int, bool)  # layer index, visible
    opacity_changed = Signal(int, int)      # layer index, opacity in %
    renamed = Signal(int, str)              # layer index, new name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0
        self._thumb_cache = []   # (rgb, alpha, icon): only redraw thumbnails of layers that changed
        self._setup_ui()
        self.setEnabled(False)  # until an image is opened

    def _setup_ui(self):
        # Thin line on the left to separate the panel from the image (theme colours)
        self.setObjectName("LayersPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#LayersPanel { border-left: 1px solid palette(mid); }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)

        title = QLabel("Layers")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # --- Add / delete buttons
        buttons = QHBoxLayout()
        self.btn_add = self._make_button("layer-add", "+", "Add a new layer")
        self.btn_add.clicked.connect(self.add_requested.emit)
        self.btn_delete = self._make_button("layer-delete", "−", "Delete the selected layer")
        self.btn_delete.clicked.connect(self.delete_requested.emit)
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_delete)
        buttons.addStretch()
        layout.addLayout(buttons)

        # --- Layer list (checkbox = visible, double-click = rename)
        self.list = QListWidget()
        self.list.setIconSize(THUMB_SIZE)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list, 1)

        # --- Opacity of the selected layer
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self._on_opacity_moved)
        opacity_row.addWidget(self.slider, 1)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(36)
        opacity_row.addWidget(self.opacity_label)
        layout.addLayout(opacity_row)

    def _make_button(self, icon_name, fallback_text, tooltip) -> QToolButton:
        button = QToolButton()
        icon = load_icon(icon_name)
        if icon:
            button.setIcon(icon)
        else:
            button.setText(fallback_text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    # ------------------------------------------------------------------
    # Row <-> layer index (the list shows the top layer first)
    # ------------------------------------------------------------------
    def _row_to_index(self, row: int) -> int:
        return self._count - 1 - row

    def _index_to_row(self, index: int) -> int:
        return self._count - 1 - index

    # ------------------------------------------------------------------
    # Refresh from the LayerStack (called by MainWindow after every change)
    # ------------------------------------------------------------------
    def set_layers(self, stack) -> None:
        self.list.blockSignals(True)
        self.slider.blockSignals(True)
        self.list.clear()

        if stack is None:
            self._count = 0
            self.setEnabled(False)
        else:
            self._count = len(stack)
            self.setEnabled(True)
            thumbs = {}
            new_cache = []
            for layer in stack.layers:
                icon = next((ic for rgb, alpha, ic in self._thumb_cache
                             if rgb is layer.rgb and alpha is layer.alpha), None)
                if icon is None:
                    icon = layer_thumbnail(layer.rgb, layer.alpha)
                new_cache.append((layer.rgb, layer.alpha, icon))
                thumbs[id(layer)] = icon
            self._thumb_cache = new_cache
            for layer in reversed(stack.layers):
                item = QListWidgetItem(thumbs[id(layer)], layer.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
                item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
                item.setToolTip("Tick to show / untick to hide. Double-click to rename.")
                self.list.addItem(item)
            self.list.setCurrentRow(self._index_to_row(stack.active))
            self._show_opacity(round(stack.active_layer.opacity * 100))
            self.btn_delete.setEnabled(len(stack) > 1)

        self.list.blockSignals(False)
        self.slider.blockSignals(False)

    def _show_opacity(self, percent: int) -> None:
        self.slider.setValue(percent)
        self.opacity_label.setText(f"{percent}%")

    # ------------------------------------------------------------------
    # User actions -> signals
    # ------------------------------------------------------------------
    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.active_changed.emit(self._row_to_index(row))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        index = self._row_to_index(self.list.row(item))
        self.visibility_changed.emit(index, item.checkState() == Qt.CheckState.Checked)
        self.renamed.emit(index, item.text())

    def _on_opacity_moved(self, value: int) -> None:
        self.opacity_label.setText(f"{value}%")
        row = self.list.currentRow()
        if row >= 0:
            self.opacity_changed.emit(self._row_to_index(row), value)
