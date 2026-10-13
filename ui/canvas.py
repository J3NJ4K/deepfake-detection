from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QTransform, QMouseEvent
from PySide6.QtCore import Qt, Signal, QPointF

# Noms des outils (utilisés par la barre d'outils et par le canvas)
TOOL_HAND = "hand"      # déplacer / faire défiler l'image (outil par défaut)
TOOL_SELECT = "select"
TOOL_CROP = "crop"
TOOL_BRUSH = "brush"


class ImageCanvas(QGraphicsView):
    # Signal émis quand le niveau de zoom change (en pourcentage entier)
    zoom_changed = Signal(int)

    # Signaux souris pour les outils (tous sauf "hand").
    # Ils envoient : le nom de l'outil actif, puis la position en PIXELS DE L'IMAGE
    # (x = colonne, y = ligne), quel que soit le zoom.
    # Exemple pour un futur outil crop :
    #     canvas.tool_pressed.connect(lambda tool, pos: ...)
    tool_pressed = Signal(str, QPointF)
    tool_dragged = Signal(str, QPointF)
    tool_released = Signal(str, QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._current_zoom: float = 1.0
        self._min_zoom: float = 0.05   # 5%
        self._max_zoom: float = 10.0   # 1000%

        # Amélioration du rendu et activation des barres de défilement
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._tool = TOOL_HAND
        self.set_tool(TOOL_HAND)

    # ------------------------------------------------------------------
    # Outils
    # ------------------------------------------------------------------
    @property
    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        """Change l'outil actif : "hand" fait défiler l'image, les autres envoient les signaux tool_*."""
        self._tool = tool
        if tool == TOOL_HAND:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _image_pos(self, event: QMouseEvent) -> QPointF:
        """Position de la souris convertie en coordonnées de l'image."""
        return self.mapToScene(event.position().toPoint())

    def _tool_active(self) -> bool:
        return self._tool != TOOL_HAND and not self._pixmap_item.pixmap().isNull()

    def mousePressEvent(self, event: QMouseEvent):
        if self._tool_active() and event.button() == Qt.MouseButton.LeftButton:
            self.tool_pressed.emit(self._tool, self._image_pos(event))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._tool_active() and event.buttons() & Qt.MouseButton.LeftButton:
            self.tool_dragged.emit(self._tool, self._image_pos(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._tool_active() and event.button() == Qt.MouseButton.LeftButton:
            self.tool_released.emit(self._tool, self._image_pos(event))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Affichage et zoom
    # ------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        """Met à jour l'image affichée."""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent):
        """Désactive le zoom par molette et laisse le défilement standard."""
        super().wheelEvent(event)

    def apply_zoom(self, zoom_value: float):
        """Applique une échelle absolue."""
        zoom_value = max(self._min_zoom, min(self._max_zoom, zoom_value))
        self._current_zoom = zoom_value
        self.setTransform(QTransform().scale(zoom_value, zoom_value))
        self.zoom_changed.emit(int(round(self._current_zoom * 100)))

    def zoom_in(self, step: float = 0.10):
        """Augmente le zoom de 10%."""
        if not self._pixmap_item.pixmap().isNull():
            self.apply_zoom(self._current_zoom + step)

    def zoom_out(self, step: float = 0.10):
        """Diminue le zoom de 10%."""
        if not self._pixmap_item.pixmap().isNull():
            self.apply_zoom(self._current_zoom - step)

    def set_zoom_percentage(self, percentage: int):
        """Définit le zoom à partir d'un entier (ex. 64 pour 64%)."""
        if not self._pixmap_item.pixmap().isNull():
            self.apply_zoom(percentage / 100.0)

    def fit_to_window(self):
        """Ajuste l'image pour qu'elle tienne parfaitement dans la fenêtre."""
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        # Calcule l'échelle résultante appliquée par fitInView
        self._current_zoom = self.transform().m11()
        self.zoom_changed.emit(int(round(self._current_zoom * 100)))
