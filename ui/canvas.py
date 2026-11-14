from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsEllipseItem,
                               QGraphicsItem)
from PySide6.QtGui import (QPixmap, QPainter, QWheelEvent, QTransform, QMouseEvent, QPen, QColor, QCursor,
                           QPainterPath, QPolygonF, QPalette)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF

from ui.placement import HANDLES, ROTATE, MOVE

# Free transform of an imported image (sizes in SCREEN pixels, whatever the zoom)
PLACEMENT_HANDLE = 8          # side of the square handles
PLACEMENT_TOLERANCE = 7       # how close the mouse must be to grab a handle
PLACEMENT_ROTATE_DISTANCE = 28
PLACEMENT_COLOR = QColor(0, 120, 215)

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
TOOL_LASSO = "lasso"
TOOL_ZOOM = "zoom"
TOOL_TEXT = "text"

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

    # Free transform of an imported image: emitted when the user clicks outside it or
    # double-clicks it (= "I am done, put it on the layer")
    placement_done = Signal()

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

        self._lasso_item = QGraphicsPathItem()
        self._lasso_item.setPen(pen)
        self._lasso_item.setBrush(QColor(0, 120, 215, 60))
        self._lasso_item.setZValue(1)
        self._lasso_item.hide()
        self._scene.addItem(self._lasso_item)

        # Yellow frame used to show a face from the results panel
        # (separate from the selection rectangle, which belongs to the Select tool)
        self._highlight_item = QGraphicsRectItem()
        highlight_pen = QPen(QColor(255, 200, 0), 3)
        highlight_pen.setCosmetic(True)       # same thickness whatever the zoom
        self._highlight_item.setPen(highlight_pen)
        self._highlight_item.setZValue(2)
        self._highlight_item.hide()
        self._scene.addItem(self._highlight_item)

        self._brush_size = 5
        self._grabbing = False
        self.grab_check = None
        self._hover_grab = False
        self.viewport().setMouseTracking(True)
        self.zoom_changed.connect(lambda _: self._update_cursor())

        # Grey area around the image, so a white image stays visible in light mode (like Paint)
        window = self.palette().color(QPalette.ColorRole.Window)
        if window.lightness() > 128:      # light theme
            self.setBackgroundBrush(QColor(204, 204, 204))
        else:                             # dark theme
            self.setBackgroundBrush(QColor(32, 32, 32))

        self._placement = None        # ui.placement.Placement being transformed, or None
        self.text_item = None         # text being typed (set by ui.text_tool.TextEditor)
        self._text_mouse = False      # True while a click that started in the text is held
        self._placement_items = {}
        self.zoom_changed.connect(lambda _: self._update_placement())

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
        self._hover_grab = False
        if getattr(self, "_placement", None) is not None:
            return              # the free transform chooses its own cursors
        if self._tool == TOOL_HAND:
            self.viewport().unsetCursor()
        elif self._tool == TOOL_ERASER:
            self.viewport().setCursor(self._eraser_cursor())
        elif self._tool == TOOL_ZOOM:
            self.viewport().setCursor(self._zoom_cursor())
        elif self._tool == TOOL_TEXT:
            self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _zoom_cursor(self) -> QCursor:
        pixmap = QPixmap(26, 26)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for color, width in ((QColor(255, 255, 255), 4), (QColor(30, 30, 30), 2)):
            painter.setPen(QPen(color, width))
            painter.drawEllipse(2, 2, 15, 15)
            painter.drawLine(15, 15, 23, 23)
            painter.drawLine(6, 10, 13, 10)
            painter.drawLine(10, 6, 10, 13)
        painter.end()
        return QCursor(pixmap, 10, 10)

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

    def _over_selection(self, event: QMouseEvent) -> bool:
        pos = self._image_pos(event)
        if self._lasso_item.isVisible():
            if self._lasso_item.path().contains(pos):
                return True
        elif self._selection_item.isVisible() and self._selection_item.rect().contains(pos):
            return True
        return self.grab_check is not None and self.grab_check(pos)

    def _set_hover_grab(self, grab: bool) -> None:
        if grab != self._hover_grab:
            self._hover_grab = grab
            if grab:
                self.viewport() .setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self._update_cursor()

    # ------------------------------------------------------------------
    # Free transform of an imported image (move / resize / rotate)
    # ------------------------------------------------------------------
    def start_placement(self, placement, pixmap: QPixmap, opacity: float = 1.0) -> None:
        """Show the image of `placement` above the picture, with a frame and handles."""
        self.end_placement()
        self._placement = placement

        piece = QGraphicsPixmapItem(pixmap)
        piece.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        piece.setOpacity(opacity)
        piece.setZValue(0.5)          # above the picture, below the selection / handles

        pen = QPen(PLACEMENT_COLOR, 1.5)
        pen.setCosmetic(True)         # same thickness whatever the zoom
        frame = QGraphicsPolygonItem()
        frame.setPen(pen)
        frame.setZValue(3)
        line = QGraphicsLineItem()
        line.setPen(pen)
        line.setZValue(3)

        items = {"piece": piece, "frame": frame, "line": line}
        half = PLACEMENT_HANDLE / 2
        for name in list(HANDLES) + [ROTATE]:
            if name == ROTATE:
                handle = QGraphicsEllipseItem(-half - 1, -half - 1, PLACEMENT_HANDLE + 2, PLACEMENT_HANDLE + 2)
            else:
                handle = QGraphicsRectItem(-half, -half, PLACEMENT_HANDLE, PLACEMENT_HANDLE)
            handle.setPen(pen)
            handle.setBrush(QColor(255, 255, 255))
            # Drawn in screen pixels: the handles keep the same size when zooming
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            handle.setZValue(4)
            items[name] = handle
        for item in items.values():
            self._scene.addItem(item)
        self._placement_items = items
        self._update_placement()
        self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)

    def end_placement(self) -> None:
        """Remove the frame and handles (MainWindow has already used or dropped the image)."""
        for item in self._placement_items.values():
            self._scene.removeItem(item)
        self._placement_items = {}
        self._placement = None
        self._update_cursor()

    def is_placing(self) -> bool:
        return self._placement is not None

    def _rotate_distance(self) -> float:
        return PLACEMENT_ROTATE_DISTANCE / self._current_zoom

    def _update_placement(self) -> None:
        placement = self._placement
        if placement is None:
            return
        m = placement.transform_matrix()
        items = self._placement_items
        items["piece"].setTransform(QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]))
        items["frame"].setPolygon(QPolygonF([QPointF(x, y) for x, y in placement.corners()]))
        points = placement.handle_points(self._rotate_distance())
        for name, (x, y) in points.items():
            items[name].setPos(x, y)
        items["line"].setLine(*points["n"], *points[ROTATE])

    def _placement_hit(self, event: QMouseEvent):
        pos = self._image_pos(event)
        tolerance = PLACEMENT_TOLERANCE / self._current_zoom
        return self._placement.hit_test(pos.x(), pos.y(), tolerance, self._rotate_distance())

    def _placement_cursor(self, kind) -> None:
        if kind == MOVE:
            shape = Qt.CursorShape.SizeAllCursor
        elif kind == ROTATE:
            shape = Qt.CursorShape.PointingHandCursor
        elif kind in HANDLES:
            # Pick the double arrow closest to the direction of the (maybe rotated) handle
            step = round((self._placement.handle_direction(kind) % 180) / 45) % 4
            shape = [Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeFDiagCursor,
                     Qt.CursorShape.SizeVerCursor, Qt.CursorShape.SizeBDiagCursor][step]
        else:
            shape = Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(shape)

    def _placement_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        kind = self._placement_hit(event)
        if kind is None:
            self.placement_done.emit()       # click outside = done
        else:
            pos = self._image_pos(event)
            self._placement.start_drag(kind, pos.x(), pos.y())
        return True

    def _placement_move(self, event: QMouseEvent) -> bool:
        placement = self._placement
        if placement.is_dragging():
            pos = self._image_pos(event)
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            placement.drag_to(pos.x(), pos.y(), shift)
            self._update_placement()
            return True
        if event.buttons() == Qt.MouseButton.NoButton:
            self._placement_cursor(self._placement_hit(event))
            return True
        return False

    def _placement_release(self, event: QMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and self._placement.is_dragging():
            self._placement.end_drag()
            self._placement_cursor(self._placement_hit(event))
            return True
        return False

    # ------------------------------------------------------------------
    # Text being typed: clicks inside it go to the text (cursor, selection)
    # ------------------------------------------------------------------
    def _on_text(self, event: QMouseEvent) -> bool:
        if self.text_item is None:
            return False
        pad = 4 / max(self._current_zoom, 0.01)
        rect = self.text_item.sceneBoundingRect().adjusted(-pad, -pad, pad, pad)
        return rect.contains(self._image_pos(event))

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._on_text(event):
            super().mouseDoubleClickEvent(event)   # double-click = select a word
            return
        if self._placement is not None and event.button() == Qt.MouseButton.LeftButton:
            if self._placement_hit(event) == MOVE:
                self._placement.end_drag()
                self.placement_done.emit()   # double-click on the image = done
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self._placement is not None and self._placement_press(event):
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._on_text(event):
            self._text_mouse = True
            super().mousePressEvent(event)         # the text item places its cursor
            return
        # Zoom tool: left click = zoom in, right click = zoom out

        if self._tool == TOOL_ZOOM and not self._pixmap_item.pixmap().isNull():
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                spot = self._image_pos(event)
                factor = 1.25 if event.button() == Qt.MouseButton.LeftButton else 1 / 1.25
                self.apply_zoom(self._current_zoom * factor)
                self.centerOn(spot)
                event.accept()
                return
        if (event.button() == Qt.MouseButton.LeftButton and self._tool in (TOOL_HAND, TOOL_SELECT, TOOL_LASSO)
                and self._over_selection(event)):
            self._grabbing = True
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            self.tool_pressed.emit(TOOL_SELECT, self._image_pos(event))
            event.accept()
            return
        if self._tool_active() and event.button() == Qt.MouseButton.LeftButton:
            self.tool_pressed.emit(self._tool, self._image_pos(event))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._placement is not None and self._placement_move(event):
            event.accept()
            return
        if self._text_mouse:
            super().mouseMoveEvent(event)          # selecting text with the mouse
            return
        if self._grabbing:
            self.tool_dragged.emit(TOOL_SELECT, self._image_pos(event))
            event.accept()
            return
        if event.buttons() == Qt.MouseButton.NoButton:
            self._set_hover_grab(self._tool in (TOOL_HAND, TOOL_SELECT, TOOL_LASSO) and self._over_selection(event))
        if self._tool_active() and event.buttons() & Qt.MouseButton.LeftButton:
            self.tool_dragged.emit(self._tool, self._image_pos(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._placement is not None and self._placement_release(event):
            event.accept()
            return
        if self._text_mouse and event.button() == Qt.MouseButton.LeftButton:
            self._text_mouse = False
            super().mouseReleaseEvent(event)
            return
        if self._grabbing and event.button() == Qt.MouseButton.LeftButton:
            self._grabbing = False
            self.tool_released.emit(TOOL_SELECT, self._image_pos(event))
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
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
        self._lasso_item.hide()

    def show_lasso(self, points) -> None:
        path = QPainterPath()
        path.addPolygon(QPolygonF(points))
        path.closeSubpath()
        self._lasso_item.setPath(path)
        self._lasso_item.show()
        self._selection_item.hide()

    def show_preview(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)

    def show_highlight(self, start: QPointF, end: QPointF) -> None:
        self._highlight_item.setRect(QRectF(start, end).normalized())
        self._highlight_item.show()

    def clear_highlight(self) -> None:
        self._highlight_item.hide()

    def clear_selection(self) -> None:
        self._selection_item.hide()
        self._lasso_item.hide()

    # ------------------------------------------------------------------
    # Affichage et zoom
    # ------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        """Met à jour l'image affichée."""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent):
        """Ctrl + molette : zoom centré sur la souris. Sans Ctrl : défilement normal."""
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and not self._pixmap_item.pixmap().isNull()):
            steps = event.angleDelta().y() / 120   # one notch of the wheel = 120
            if steps:
                cursor = event.position().toPoint()
                before = self.mapToScene(cursor)               # image point under the mouse
                self.apply_zoom(self._current_zoom * (1.25 ** steps))
                after = self.mapToScene(cursor)
                # Scroll so that the same image point stays under the mouse
                shift = (before - after) * self._current_zoom
                h, v = self.horizontalScrollBar(), self.verticalScrollBar()
                h.setValue(h.value() + round(shift.x()))
                v.setValue(v.value() + round(shift.y()))
            event.accept()
            return
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
