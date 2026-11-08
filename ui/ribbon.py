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
from PySide6.QtGui import QAction, QPalette
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QSizePolicy, QToolBar, QToolButton,
    QVBoxLayout, QWidget,
)

LARGE_ICON_SIZE = QSize(32, 32)
SMALL_ICON_SIZE = QSize(20, 20)


class RibbonGroup(QWidget):
    """Un groupe du ruban : des boutons en haut, un titre en bas."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        # Keep the natural width of the group: extra room in the ribbon is left empty
        # on the right instead of stretching the groups (and their buttons)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

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

    def add_large_button(self, action: QAction, highlight: bool = False) -> QToolButton:
        """
        Gros bouton : icône au-dessus du texte (ou juste le texte s'il n'y a pas d'icône).
        highlight=True : texte en gras, pour la fonction principale de l'appli.
        """
        button = self._make_button(action)
        if highlight:
            font = button.font()
            font.setBold(True)
            button.setFont(font)
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
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            grid.addWidget(button, index % rows, index // rows)
        self._buttons_layout.addLayout(grid)

    def add_labeled_buttons(self, actions: list[QAction], rows: int = 3) -> None:
        """
        Petits boutons avec l'icône ET le nom à côté, en colonne (comme dans Word).
        Prend moins de place qu'un gros bouton, tout en restant lisible.
        """
        grid = QGridLayout()
        grid.setSpacing(0)
        for index, action in enumerate(actions):
            button = self._make_button(action)
            button.setIconSize(SMALL_ICON_SIZE)
            if action.icon().isNull():
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            else:
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            grid.addWidget(button, index % rows, index // rows)
        self._buttons_layout.addLayout(grid)

    def add_menu_button(self, text: str, actions: list[QAction], icon=None,
                        tooltip: str = "") -> QToolButton:
        """Gros bouton qui ouvre un menu déroulant (ex : tous les filtres dans un seul bouton)."""
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip or text)
        button.setAutoRaise(True)
        button.setIconSize(LARGE_ICON_SIZE)
        if icon:
            button.setIcon(icon)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        else:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        menu = QMenu(button)
        for action in actions:
            menu.addAction(action)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)  # one click opens the menu
        button.setMinimumWidth(56)
        button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._buttons_layout.addWidget(button)
        return button

    def add_widget(self, widget: QWidget) -> None:
        """Put any widget in the group (e.g. the colour palette)."""
        self._buttons_layout.addWidget(widget)

    def _make_button(self, action: QAction) -> QToolButton:
        button = QToolButton(self)
        button.setDefaultAction(action)
        button.setAutoRaise(True)  # bouton "plat" qui s'éclaire au survol
        return button


def _separator_color(widget: QWidget) -> str:
    """Light line on a dark theme, dark line on a light theme (semi-transparent)."""
    window = widget.palette().color(QPalette.ColorRole.Window)
    return "rgba(255, 255, 255, 70)" if window.lightness() < 128 else "rgba(0, 0, 0, 60)"


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

    def _make_separator(self) -> QWidget:
        """Thin vertical line between two groups, visible in light and dark themes."""
        holder = QWidget(self)
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(4, 8, 4, 8)   # small gap around the line, shorter than the ribbon
        line = QFrame(holder)
        line.setFixedWidth(1)
        line.setStyleSheet(f"background-color: {_separator_color(self)}; border: none;")
        layout.addWidget(line)
        return holder

    def add_group(self, title: str) -> RibbonGroup:
        if self._groups:
            self.addWidget(self._make_separator())  # trait vertical entre deux groupes
        group = RibbonGroup(title, self)
        self.addWidget(group)
        self._groups.append(group)
        return group
