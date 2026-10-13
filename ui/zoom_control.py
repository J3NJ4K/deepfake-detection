"""
Barre de zoom style Paint :  [Ajuster]  [ 86% ▾ ]  [−] ━━━●━━━━━ [+]

Aucune couleur n'est écrite en dur : les widgets prennent automatiquement
les couleurs du thème Windows (clair ou sombre).
"""

import math
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QToolButton, QWidget

from ui.utils import load_icon

MIN_ZOOM = 5      # %
MAX_ZOOM = 1000   # %
SLIDER_STEPS = 1000

# Valeurs proposées dans la liste déroulante (du plus grand au plus petit, comme Paint)
PRESETS = [1000, 800, 600, 400, 300, 200, 100, 75, 50, 25, 10]


def percent_to_slider(percent: int) -> int:
    """
    La barre glissante est logarithmique : passer de 10% à 20% prend autant de place
    que de 100% à 200%. Sinon tout ce qui est sous 100% serait écrasé à gauche.
    """
    percent = max(MIN_ZOOM, min(MAX_ZOOM, percent))
    ratio = math.log(percent / MIN_ZOOM) / math.log(MAX_ZOOM / MIN_ZOOM)
    return round(ratio * SLIDER_STEPS)


def slider_to_percent(value: int) -> int:
    ratio = value / SLIDER_STEPS
    return round(MIN_ZOOM * (MAX_ZOOM / MIN_ZOOM) ** ratio)


class ZoomControl(QWidget):
    # Signaux pour communiquer avec le Canvas
    zoom_in_requested = Signal()
    zoom_out_requested = Signal()
    fit_requested = Signal()
    zoom_percent_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        # Bouton Ajuster à la fenêtre
        self.btn_fit = self._make_button("fit-to-screen", "⛶", "Fit to Window")
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        layout.addWidget(self.btn_fit)
        layout.addSpacing(6)

        # Champ du pourcentage : on peut taper une valeur OU choisir dans la liste
        self.combo_percent = QComboBox()
        self.combo_percent.setEditable(True)
        self.combo_percent.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # ne pas ajouter ce qu'on tape à la liste
        self.combo_percent.setMaxVisibleItems(len(PRESETS))
        self.combo_percent.addItems([f"{p}%" for p in PRESETS])
        self.combo_percent.setFixedWidth(90)
        self.combo_percent.setToolTip("Zoom level (type a value and press Enter)")
        self.combo_percent.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.combo_percent.setEditText("100%")
        # Choix dans la liste
        self.combo_percent.textActivated.connect(self._on_text_entered)
        # Valeur tapée + Entrée
        self.combo_percent.lineEdit().returnPressed.connect(
            lambda: self._on_text_entered(self.combo_percent.currentText())
        )
        layout.addWidget(self.combo_percent)
        layout.addSpacing(6)

        # Bouton Moins
        self.btn_minus = self._make_button("zoom-out", "−", "Zoom Out")
        self.btn_minus.setAutoRepeat(True)  # rester appuyé = zoom continu
        self.btn_minus.clicked.connect(self.zoom_out_requested.emit)
        layout.addWidget(self.btn_minus)

        # Barre glissante
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setFixedWidth(160)
        self.slider.setToolTip("Zoom")
        self.slider.setValue(percent_to_slider(100))
        self.slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.slider)

        # Bouton Plus
        self.btn_plus = self._make_button("zoom-in", "+", "Zoom In")
        self.btn_plus.setAutoRepeat(True)
        self.btn_plus.clicked.connect(self.zoom_in_requested.emit)
        layout.addWidget(self.btn_plus)

    def _make_button(self, icon_name: str, fallback_text: str, tooltip: str) -> QToolButton:
        """Bouton plat avec l'icône assets/<icon_name>.svg, ou un symbole en attendant."""
        button = QToolButton()
        icon = load_icon(icon_name)
        if icon:
            button.setIcon(icon)
        else:
            button.setText(fallback_text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    def set_percent(self, value: int):
        """Reçoit le pourcentage du canvas et met à jour le champ et la barre."""
        self.combo_percent.setEditText(f"{value}%")
        # blockSignals évite une boucle : barre -> canvas -> barre -> canvas...
        self.slider.blockSignals(True)
        self.slider.setValue(percent_to_slider(value))
        self.slider.blockSignals(False)

    def _on_slider_moved(self, value: int):
        percent = slider_to_percent(value)
        self.combo_percent.setEditText(f"{percent}%")
        self.zoom_percent_changed.emit(percent)

    def _on_text_entered(self, text: str):
        """Gère la saisie manuelle ou le choix dans la liste."""
        numbers = re.findall(r"\d+", text)
        val = int(numbers[0]) if numbers else 100
        val = max(MIN_ZOOM, min(MAX_ZOOM, val))
        self.zoom_percent_changed.emit(val)
