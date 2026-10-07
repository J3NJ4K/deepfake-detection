import re
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLineEdit, QFrame
from PySide6.QtCore import Qt, Signal

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
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        # Style sombre arrondi inspiré de la capture
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-radius: 8px;
            }
            QPushButton {
                background: transparent;
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
            QLineEdit {
                background: transparent;
                color: #ffffff;
                font-size: 13px;
                font-weight: 600;
                border: none;
                min-width: 48px;
                max-width: 60px;
            }
            QFrame#Separator {
                background-color: #444444;
                width: 1px;
                max-width: 1px;
            }
        """)

        # Bouton Plein écran / Adapter
        self.btn_fit = QPushButton("⛶")
        self.btn_fit.setToolTip("Fit to Window")
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        layout.addWidget(self.btn_fit)

        # Séparateur vertical
        separator = QFrame()
        separator.setObjectName("Separator")
        separator.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(separator)

        # Bouton Moins (-)
        self.btn_minus = QPushButton("—")
        self.btn_minus.setToolTip("Zoom Out (-10%)")
        self.btn_minus.clicked.connect(self.zoom_out_requested.emit)
        layout.addWidget(self.btn_minus)

        # Champ de texte du pourcentage
        self.entry_percent = QLineEdit("100%")
        self.entry_percent.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entry_percent.returnPressed.connect(self._on_text_entered)
        layout.addWidget(self.entry_percent)

        # Bouton Plus (+)
        self.btn_plus = QPushButton("+")
        self.btn_plus.setToolTip("Zoom In (+10%)")
        self.btn_plus.clicked.connect(self.zoom_in_requested.emit)
        layout.addWidget(self.btn_plus)

    def set_percent(self, value: int):
        """Reçoit le pourcentage du canvas et met à jour le texte."""
        self.entry_percent.setText(f"{value}%")

    def _on_text_entered(self):
        """Gère la saisie manuelle de l'utilisateur."""
        text = self.entry_percent.text()
        # Récupère uniquement les chiffres entrés
        numbers = re.findall(r"\d+", text)
        if numbers:
            val = int(numbers[0])
            val = max(5, min(1000, val))
            self.zoom_percent_changed.emit(val)
        else:
            self.entry_percent.setText("100%")
            self.zoom_percent_changed.emit(100)