"""
Ruban style "Paint" : une seule bande, découpée en groupes avec un titre en dessous.

Utilisation :
    ribbon = Ribbon(main_window)
    main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, ribbon)

    group = ribbon.add_group("Image")
    group.add_large_button(une_action)              # gros bouton, icône au-dessus du texte
    group.add_small_buttons([a1, a2, a3, a4])       # petits boutons en grille (2 lignes)

Les boutons réutilisent directement les QAction : si une action est désactivée
ou cochée, le bouton suit automatiquement.
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QToolBar, QToolButton,
    QVBoxLayout, QWidget,
)

LARGE_ICON_SIZE = QSize(32, 32)
SMALL_ICON_SIZE = QSize(20, 20)


class RibbonGroup(QWidget):
    """Un groupe du ruban : des boutons en haut, un titre en bas."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(2)

        # Zone des boutons (on y ajoute des gros boutons et/ou des grilles)
        self._buttons_layout = QHBoxLayout()
        self._buttons_layout.setSpacing(4)
        outer.addLayout(self._buttons_layout, 1)

        # Titre du groupe, centré en bas, un peu plus petit que le texte normal
        self.title_label = QLabel(title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.title_label.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.title_label.setFont(font)
        self.title_label.setEnabled(False)  # couleur grisée, comme dans Paint
        outer.addWidget(self.title_label)

    def add_large_button(self, action: QAction) -> QToolButton:
        """Gros bouton : icône au-dessus du texte (ou juste le texte s'il n'y a pas d'icône)."""
        button = self._make_button(action)
        button.setIconSize(LARGE_ICON_SIZE)
        if action.icon().isNull():
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        else:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setMinimumWidth(56)
        button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._buttons_layout.addWidget(button)
        return button

    def add_small_buttons(self, actions: list[QAction], rows: int = 2) -> None:
        """
        Petits boutons rangés en grille, colonne par colonne.
        Avec une icône : icône seule (le nom s'affiche au survol).
        Sans icône : le texte de l'action, en attendant d'avoir une icône.
        """
        grid = QGridLayout()
        grid.setSpacing(2)
        for index, action in enumerate(actions):
            button = self._make_button(action)
            button.setIconSize(SMALL_ICON_SIZE)
            if action.icon().isNull():
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            else:
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            grid.addWidget(button, index % rows, index // rows)
        self._buttons_layout.addLayout(grid)

    def _make_button(self, action: QAction) -> QToolButton:
        button = QToolButton(self)
        button.setDefaultAction(action)
        button.setAutoRaise(True)  # bouton "plat" qui s'éclaire au survol
        return button


class Ribbon(QToolBar):
    """
    La bande complète. C'est une QToolBar : si la fenêtre est trop étroite,
    les groupes qui ne rentrent plus passent derrière une flèche » à droite.
    """

    def __init__(self, parent=None):
        super().__init__("Ribbon", parent)
        self.setObjectName("Ribbon")
        self.setMovable(False)
        self.setFloatable(False)
        # Empêche le clic droit qui permettrait de masquer la barre par erreur
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._groups: list[RibbonGroup] = []

    def add_group(self, title: str) -> RibbonGroup:
        if self._groups:
            self.addSeparator()  # trait vertical entre deux groupes
        group = RibbonGroup(title, self)
        self.addWidget(group)
        self._groups.append(group)
        return group
