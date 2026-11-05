from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent, QTransform, QMouseEvent, QPen, QColor, QCursor
from PySide6.QtCore import Qt, Signal, QPointF, QRectF

# Noms des outils (utilisés par la barre d'outils et par le canvas)
TOOL_HAND = "hand"      # déplacer / faire défiler l'image (outil par défaut)
TOOL_SELECT = "select"
TOOL_CROP = "crop"
TOOL_BRUSH = "brush"
TOOL_ERASER = "eraser"
TOOL_PICKER = "picker"
TOOL_RECT = "rectangle"
TOOL_ELLIPSE = "ellipse"
TOOL_LINE = "line"
TOOL_FILL = "fill"
TOOL_TRIANGLE = "triangle"
TOOL_RIGHT_TRIANGLE = "right_triangle"
TOOL_DIAMOND = "diamond"
TOOL_STAR = "star"
TOOL_ARROW = "arrow"

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

        # Dashed rectangle when cropping
        self._selection_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._selection_item.setPen(pen)
        self._selection_item.setBrush(QColor(0, 120, 215, 60))
        self._selection_item.setZValue(1)
        self._selection_item.hide()
        self._scene.addItem(self._selection_item)

        self._brush_size = 5
        self.zoom_changed.connect(lambda _: self._update_cursor())

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
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._update_cursor()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = size
        self._update_cursor()

    def _update_cursor(self) -> None:
        if self._tool == TOOL_HAND:
            self.viewport().unsetCursor()
        elif self._tool == TOOL_ERASER:
            self.viewport().setCursor(self._eraser_cursor())
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _eraser_cursor(self) -> QCursor:
        diameter = int(round(self._brush_size * self._current_zoom))
        diameter = max(8, min(diameter, 250))
        pixmap = QPixmap(diameter + 2, diameter + 2)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(QColor(255, 255, 255, 160))
        painter.drawEllipse(1, 1, diameter, diameter)
        painter.end()

        center = pixmap.width() // 2
        return QCursor(pixmap, center, center)

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

    # Selection Rectangle
    def show_selection(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        rect = rect.intersected(self._pixmap_item.boundingRect())  # keep it inside the image
        self._selection_item.setRect(rect)
        self._selection_item.show()

    def show_preview(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)

    def clear_selection(self) -> None:
        self._selection_item.hide()

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
