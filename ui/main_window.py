import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QStatusBar, QMessageBox, QToolBar,
    QFileDialog, QWidget, QSizePolicy, QInputDialog,
    QColorDialog, QHBoxLayout
)
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap, QColor, QIcon, QPainter, QPalette
from PySide6.QtCore import Qt, QSize, QPointF
from ui.utils import load_icon, numpy_to_qimage
from ui.canvas import (ImageCanvas, TOOL_HAND, TOOL_SELECT, TOOL_CROP, TOOL_BRUSH, TOOL_ERASER, TOOL_PICKER,
                       TOOL_RECT, TOOL_ELLIPSE, TOOL_LINE, TOOL_FILL,
                       TOOL_TRIANGLE, TOOL_RIGHT_TRIANGLE, TOOL_DIAMOND, TOOL_STAR, TOOL_ARROW)
from ui.zoom_control import ZoomControl
from ui.history import ImageHistory
from ui.ribbon import Ribbon
from ui.layers import LayerStack, overlay_from_drawing
from ui.layers_panel import LayersPanel
from ui.tool_options import ToolOptionsBar
from ui.color_palette import ColorPalette, SLOT_COLOR, SLOT_FILL
from core.transform import (rotate_left, rotate_right, rotate_180, flip_horizontal, flip_vertical,
                            flip_diagonal, resize_scale, crop)
from core.image_io import read_img, save_img, new_img, img_properties
from core.selection import selection_box, empty_region, paste_img
from core.drawing import draw_brush_stroke, get_pixel_color, draw_shape, flood_fill
from detection.face_detector import detect_faces
from detection.pipeline import analyze_deepfake
from core.filters import (
    binary_threshold,
    gaussian_blur,
    histogram_threshold,
    sobel_filter,
)

SHAPE_TOOLS = (TOOL_RECT, TOOL_ELLIPSE, TOOL_LINE,
               TOOL_TRIANGLE, TOOL_RIGHT_TRIANGLE, TOOL_DIAMOND, TOOL_STAR, TOOL_ARROW)

# Operations that change the size or orientation of the image: they are applied
# to ALL layers so that the layers stay aligned. Everything else only changes the
# active layer.
# Tools that use the size setting, and the one that uses the opacity setting
SIZE_TOOLS = (TOOL_BRUSH, TOOL_ERASER) + SHAPE_TOOLS
OPACITY_TOOLS = (TOOL_BRUSH,)

GEOMETRIC_OPS = (rotate_left, rotate_right, rotate_180, flip_horizontal, flip_vertical,
                 flip_diagonal, resize_scale, crop)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Machine Vision Photo Editor & Deepfake Detector")
        self.resize(1280, 760)

        # The document is a stack of layers (see ui/layers.py).
        # self.current_image (property below) = pixels of the ACTIVE layer, NumPy RGB uint8.
        self.layers: LayerStack | None = None
        self._displayed_size = None
        self.current_file_path: str | None = None

        # Historique pour Undo / Redo (20 étapes max)
        self.history = ImageHistory(max_steps=20)

        # Composant central (Canvas). It is placed in the window, next to the
        # layers panel, by _create_layers_panel().
        self.canvas = ImageCanvas(self)

        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open an image to start (Ctrl+O)")

        # Initialisation UI
        self._create_actions()
        self._create_menus()
        self._create_layers_panel()
        self._create_ribbon()
        self._create_bottom_toolbar()
        self._update_undo_redo_state()

    # ------------------------------------------------------------------
    # Current image = active layer
    # ------------------------------------------------------------------
    @property
    def current_image(self) -> np.ndarray | None:
        """Pixels of the active layer (what the tools and filters work on)."""
        return None if self.layers is None else self.layers.active_layer.rgb

    @current_image.setter
    def current_image(self, image: np.ndarray | None) -> None:
        """Setting a whole new image (Open, New) starts a new document with one layer."""
        self.layers = None if image is None else LayerStack.from_image(image)
        self._displayed_size = None
        self.deselect()
        self._refresh_layers_panel()

    def _composite(self) -> np.ndarray | None:
        """The image as the user sees it: all visible layers merged."""
        return None if self.layers is None else self.layers.composite()

    def _make_action(self, text: str, slot, icon_name: str | None = None,
                     tooltip: str | None = None, shortcut=None) -> QAction:
        """
        Crée une QAction. Si l'icône assets/<icon_name>.svg existe, elle est utilisée,
        sinon le bouton affichera simplement le texte.
        """
        icon = load_icon(icon_name) if icon_name else None
        action = QAction(icon, text, self) if icon else QAction(text, self)
        if shortcut is not None:
            if isinstance(shortcut, list):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        if tooltip:
            action.setToolTip(tooltip)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _create_actions(self):
        # --- File ---
        self.new_act = self._make_action("&New...", self.new_image,
                                         shortcut=QKeySequence.StandardKey.New)
        self.open_act = self._make_action("&Open...", self.open_image,
                                          shortcut=QKeySequence.StandardKey.Open)
        self.save_act = self._make_action("&Save", self.save_image, "disk", "Save (Ctrl+S)",
                                          QKeySequence.StandardKey.Save)
        self.save_as_act = self._make_action("Save &As...", self.save_image_as, "floppy-disk-pen",
                                             "Save As (Ctrl+Shift+S)", QKeySequence.StandardKey.SaveAs)
        self.properties_act = self._make_action("&Properties...", self.show_properties)
        self.quit_act = self._make_action("&Quit", self.close,
                                          shortcut=QKeySequence.StandardKey.Quit)

        # --- History (Undo / Redo) ---
        self.undo_act = self._make_action("&Undo", self.undo, "undo", "Undo (Ctrl+Z)",
                                          QKeySequence.StandardKey.Undo)
        # Ctrl+Y (Windows) et Ctrl+Shift+Z (macOS / Linux / Photoshop)
        self.redo_act = self._make_action("&Redo", self.redo, "redo", "Redo (Ctrl+Y)",
                                          [QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])

        # --- Tools (un seul actif à la fois) ---
        # Pour activer un outil quand il sera prêt : enlever setEnabled(False)
        # et brancher canvas.tool_pressed / tool_dragged / tool_released.
        self.tool_group = QActionGroup(self)

        # One tool at a time, pressing the active tool again deactivates it
        self.tool_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)

        self.hand_tool_act = self._add_tool("Hand", TOOL_HAND, "hand", "Hand: drag to scroll the image when zoomed in")
        self.select_tool_act = self._add_tool("Select", TOOL_SELECT, "select",
                                              "Select: drag to select an area, drag inside it to move it")
        self.crop_tool_act = self._add_tool("Crop", TOOL_CROP, "crop", "Crop: Drag a rectangle over the image")
        self.brush_tool_act = self._add_tool("Brush", TOOL_BRUSH, "brush", "Brush: drag to draw")
        self.eraser_tool_act = self._add_tool("Eraser", TOOL_ERASER, "eraser", "Eraser: drag to erase (change its size with Size button)")
        self.picker_tool_act = self._add_tool("Pick color", TOOL_PICKER, "picker",
                                              "Color picker: click the image to use that color for the brush")

        self.rect_tool_act = self._add_tool("Rectangle", TOOL_RECT, "rectangle", "Rectangle: drag to draw")
        self.ellipse_tool_act = self._add_tool("Ellipse", TOOL_ELLIPSE, "ellipse", "Ellipse: drag to draw")
        self.line_tool_act = self._add_tool("Line", TOOL_LINE, "line", "Line: drag to draw")
        self.triangle_tool_act = self._add_tool("Triangle", TOOL_TRIANGLE, "triangle", "Triangle: drag to draw")
        self.right_triangle_tool_act = self._add_tool("Right triangle", TOOL_RIGHT_TRIANGLE, "right_triangle",
                                                      "Right triangle: drag to draw")
        self.diamond_tool_act = self._add_tool("Diamond", TOOL_DIAMOND, "diamond", "Diamond: drag to draw")
        self.star_tool_act = self._add_tool("Star", TOOL_STAR, "star", "Star: drag to draw")
        self.arrow_tool_act = self._add_tool("Arrow", TOOL_ARROW, "arrow", "Arrow: drag to draw")
        self.fill_tool_act = self._add_tool("Fill", TOOL_FILL, "bucket",
                                            "Fill: click inside a shape to fill it with the fill color")

        self.hand_tool_act.setChecked(True)

        self.tool_group.triggered.connect(self._on_tool_clicked)

        # Crop Tool
        self._crop_start = None
        self._stroke_points = []
        self._stroke_mask = None    # coverage of the stroke being drawn (see _begin_stroke)
        self._shape_start = None
        self._shape_view = None     # screen image when a shape started (for its preview)
        self.brush_color = (0, 0, 0)  # black in the beginning, like Paint
        self.brush_size = 5
        self.brush_opacity = 1.0    # 0..1, set with the tool options bar
        self.color_act = self._make_action("Color", self.choose_brush_color, None,
                                           "Color for the brush, lines and shape outlines")
        self.size_act = self._make_action(f"Size: {self.brush_size}", self.choose_brush_size, None,
                                          "Size of the brush, eraser, lines and shape outlines")
        self.fill_color = (255, 255, 255)
        self.fill_color_act = self._make_action("Fill color", self.choose_fill_color, None, "Fill color for shapes")

        # Clipboard
        self._selection = None
        self._select_start = None
        self._move_start = None
        self._lifted = None
        self._floating = None
        self._clipboard = None
        self.cut_act = self._make_action("Cu&t", self.cut_selection, "cut", "Cut selected area (Ctrl+X)",
                                         QKeySequence.StandardKey.Cut)
        self.copy_act = self._make_action("&Copy", self.copy_selection, "copy", "Copy selected area (Ctrl+C)",
                                          QKeySequence.StandardKey.Copy)
        self.paste_act = self._make_action("&Paste", self.paste_clipboard, "paste",
                                           "Paste image onto a new layer (Ctrl+V)", QKeySequence.StandardKey.Paste)
        self.delete_sel_act = self._make_action("&Delete selection", self.delete_selection, None,
                                                "Delete the selected area (Del)", QKeySequence.StandardKey.Delete)
        self.select_all_act = self._make_action("Select &All", self.select_all, None,
                                                "Select the entire image (Ctrl+A)", QKeySequence.StandardKey.SelectAll)
        self.deselect_act = self._make_action("D&eselect", self.deselect, None, "Deselect (Esc)", "Esc")
        self._update_color_icon()
        self._update_size_icon()
        self._update_fill_icon()
        self.canvas.grab_check = self._piece_at
        self.canvas.tool_pressed.connect(self._on_tool_pressed)
        self.canvas.tool_dragged.connect(self._on_tool_dragged)
        self.canvas.tool_released.connect(self._on_tool_released)

        # --- Image ---
        self.rot_ccw_act = self._make_action("Rotate Left", lambda: self._apply_edit(rotate_left),
                                             "rotate_left", "Rotate 90° Counter-Clockwise (Ctrl+Shift+R)",
                                             "Ctrl+Shift+R")
        self.rot_cw_act = self._make_action("Rotate Right", lambda: self._apply_edit(rotate_right),
                                            "rotate_right", "Rotate 90° Clockwise (Ctrl+R)", "Ctrl+R")
        self.flip_h_act = self._make_action("Flip &Horizontal", lambda: self._apply_edit(flip_horizontal),
                                            "reflect-horizontal", "Flip Horizontal (mirror)")
        self.flip_v_act = self._make_action("Flip &Vertical", lambda: self._apply_edit(flip_vertical),
                                            "reflect-vertical", "Flip Vertical (upside down)")
        self.rot_180_act = self._make_action("180°", lambda: self._apply_edit(rotate_180),
                                             "rotate_180", "Rotate 180°")
        self.flip_d_act = self._make_action("Transpose", lambda: self._apply_edit(flip_diagonal),
                                            "reflect-diagonal", "Flip Diagonal (swap rows and columns)")
        self.resize_act = self._make_action("&Resize...", self.resize_image, "resize", "Resize the image")

        # --- Filters ---
        self.blur_act = self._make_action("Blur", self.apply_gaussian_blur, "blur", "Gaussian Blur")
        self.sobel_act = self._make_action("Sobel", self.apply_sobel_filter, "sobel",
                                           "Sobel Filter (Edge Detection)")
        self.threshold_act = self._make_action("Threshold", self.apply_binary_threshold, "threshold",
                                               "Binary Threshold (128)")
        self.otsu_act = self._make_action("Otsu", self.apply_histogram_threshold, "otsu",
                                          "Automatic Threshold (Otsu)")

        # --- Detection ---
        self.face_detect_act = self._make_action("Faces", self.run_face_detection, "face",
                                                 "Detect Faces")
        self.deepfake_act = self._make_action("Deepfake", self.run_deepfake_analysis, "deepfake",
                                              "Run Deepfake Analysis")

    def _on_tool_clicked(self, action):
        """Switch to the clicked tool, or back to the Hand if the active tool was clicked again."""
        if action.isChecked():
            tool = action.data()
        else:
            self.hand_tool_act.setChecked(True)
            tool = TOOL_HAND
        if tool not in (TOOL_SELECT, TOOL_HAND):
            self.deselect()
        self.canvas.set_tool(tool)
        # Show the size / opacity bar only for the tools that use them
        self.tool_options.show_for_tool(tool in SIZE_TOOLS, tool in OPACITY_TOOLS)

    def _add_tool(self, text: str, tool_name: str, icon_name: str, tooltip: str) -> QAction:
        """Crée une action d'outil "cochable" et l'ajoute au groupe exclusif."""
        action = self._make_action(text, None, icon_name, tooltip)
        action.setCheckable(True)
        action.setData(tool_name)
        self.tool_group.addAction(action)
        return action

    def _create_menus(self):
        # Barre de menus réduite : le reste des fonctions est dans le ruban
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.new_act)
        file_menu.addAction(self.open_act)
        file_menu.addSeparator()
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.save_as_act)
        file_menu.addSeparator()
        file_menu.addAction(self.properties_act)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_act)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_act)
        edit_menu.addAction(self.redo_act)
        edit_menu.addSeparator()
        edit_menu.addAction(self.cut_act)
        edit_menu.addAction(self.copy_act)
        edit_menu.addAction(self.paste_act)
        edit_menu.addAction(self.delete_sel_act)
        edit_menu.addSeparator()
        edit_menu.addAction(self.select_all_act)
        edit_menu.addAction(self.deselect_act)

        # Undo / Redo also as icons right after the menus, like Paint (saves room in the ribbon)
        menubar.addAction(self.undo_act)
        menubar.addAction(self.redo_act)

    def apply_gaussian_blur(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self._set_image(gaussian_blur(self.current_image, kernel_size=5))
            self.statusBar().showMessage("Gaussian Blur appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur flou: {e}", 3000)

    def apply_sobel_filter(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self._set_image(sobel_filter(self.current_image))
            self.statusBar().showMessage("Filtre Sobel appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur Sobel: {e}", 3000)

    def apply_binary_threshold(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self._set_image(binary_threshold(self.current_image, threshold=128))
            self.statusBar().showMessage("Seuillage binaire appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur seuillage: {e}", 3000)

    def apply_histogram_threshold(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self._set_image(histogram_threshold(self.current_image))
            self.statusBar().showMessage("Seuillage Otsu appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur Otsu: {e}", 3000)

    def run_face_detection(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        try:
            # Detect on what the user sees (all visible layers merged)
            boxes = detect_faces(self._composite())
            if not boxes:
                QMessageBox.information(self, "Face Detection", "No faces detected in the image.")
                return

            def draw_boxes(image):
                image = image.copy()
                for (x1, y1, x2, y2) in boxes:
                    # Rectangle bleu (RGB: 0, 150, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 150, 255), 2)
                return image

            # The boxes go on their own layer: hide or delete it to get the clean photo back
            self._add_overlay_layer("Faces", draw_boxes)
            self.status_bar.showMessage(f"Detected {len(boxes)} face(s).")
            QMessageBox.information(self, "Face Detection", f"Successfully detected {len(boxes)} face(s).")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run face detection: {e}")

    def run_deepfake_analysis(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        self.status_bar.showMessage("Running deepfake analysis...")
        try:
            # Analyse what the user sees (all visible layers merged)
            results = analyze_deepfake(self._composite())
            if not results:
                QMessageBox.information(
                    self, "Deepfake Detection", "No faces detected to analyze."
                )
                self.status_bar.showMessage("Analysis complete: No faces detected.")
                return

            summary_messages = []
            for i, res in enumerate(results, start=1):
                prob = res["manipulation_probability"]
                verdict = "Manipulated (Fake)" if prob >= 0.5 else "Authentic (Real)"
                summary_messages.append(f"Face #{i}: {verdict} — {prob * 100:.1f}% manipulation probability")

            def draw_results(image):
                image = image.copy()
                for res in results:
                    x1, y1, x2, y2 = res["bbox"]
                    prob = res["manipulation_probability"]
                    is_fake = prob >= 0.5

                    # Vert (réel) si < 50%, Rouge (deepfake) si >= 50%
                    color = (255, 50, 50) if is_fake else (50, 205, 50)
                    label = f"Fake: {prob * 100:.1f}%" if is_fake else f"Real: {(1 - prob) * 100:.1f}%"

                    # Dessiner le cadre
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Dessiner une étiquette texte au-dessus de la boîte
                    font_scale = 0.6
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    # Fond plein sous le texte pour la lisibilité
                    cv2.rectangle(
                        image,
                        (x1, max(0, y1 - text_h - 10)),
                        (x1 + text_w, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        image,
                        label,
                        (x1, max(text_h, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
                return image

            # Results go on their own layer (undo with Ctrl+Z, or hide/delete the layer)
            self._add_overlay_layer("Deepfake analysis", draw_results)

            report = "\n".join(summary_messages)
            self.status_bar.showMessage("Deepfake analysis complete.")
            QMessageBox.information(self, "Deepfake Analysis Results", report)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run deepfake analysis: {e}")
            self.status_bar.showMessage("Analysis failed.")

    def _add_overlay_layer(self, name, draw):
        """Put what draw(image) draws on a new transparent layer at the top."""
        height, width = self.layers.size
        rgb, alpha = overlay_from_drawing(draw, height, width)
        self._push_history()
        self.layers.add_layer_on_top(name, rgb, alpha)
        self._after_edit()

    def _create_ribbon(self):
        """Ruban en haut de la fenêtre, style Paint : un groupe par famille de fonctions."""
        self.ribbon = Ribbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)

        clipboard = self.ribbon.add_group("Clipboard")
        clipboard.add_large_button(self.paste_act)
        clipboard.add_small_buttons([self.cut_act, self.copy_act])

        image = self.ribbon.add_group("Image")
        image.add_small_buttons([self.rot_ccw_act, self.rot_cw_act,
                                 self.flip_h_act, self.flip_v_act,
                                 self.rot_180_act, self.flip_d_act])
        image.add_large_button(self.resize_act)

        tools = self.ribbon.add_group("Tools")
        tools.add_small_buttons([self.hand_tool_act, self.select_tool_act,
                                 self.crop_tool_act, self.brush_tool_act,
                                 self.eraser_tool_act, self.picker_tool_act,
                                 self.fill_tool_act])

        shapes = self.ribbon.add_group("Shapes")
        shapes.add_small_buttons([self.rect_tool_act, self.ellipse_tool_act,
                                  self.line_tool_act, self.triangle_tool_act,
                                  self.right_triangle_tool_act, self.diamond_tool_act,
                                  self.star_tool_act, self.arrow_tool_act])

        # One colour block for the brush, the shapes and the fill tool
        colors = self.ribbon.add_group("Colors")
        self.color_palette = ColorPalette(self.brush_color, self.fill_color)
        self.color_palette.color_chosen.connect(self._on_palette_color)
        colors.add_widget(self.color_palette)

        layers = self.ribbon.add_group("Layers")
        layers.add_large_button(self.layers_act)

        filters = self.ribbon.add_group("Filters")
        filters.add_small_buttons([self.blur_act, self.sobel_act,
                                   self.threshold_act, self.otsu_act])

        detection = self.ribbon.add_group("Detection")
        detection.add_large_button(self.face_detect_act)
        detection.add_large_button(self.deepfake_act)

    def _on_palette_color(self, slot, rgb):
        """A colour was picked in the Colors block (palette or colour wheel)."""
        if slot == SLOT_COLOR:
            self.brush_color = tuple(rgb)
            self._update_color_icon()
        else:
            self.fill_color = tuple(rgb)
            self._update_fill_icon()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------
    def _set_image(self, new_image: np.ndarray) -> None:
        """
        Remplace les pixels du calque actif (sa transparence est gardée), en sauvegardant
        l'état précédent dans l'historique. Utilisé par les filtres.
        Pour dessiner sur un calque, utiliser plutôt _apply_edit (voir plus bas).
        """
        if self.layers is None:
            self.current_image = new_image
            self._after_edit()
            return
        self._push_history()
        self.layers.active_layer.rgb = new_image
        self._after_edit()

    def _push_history(self) -> None:
        """Save the current state of all layers (cheap: the arrays are shared, not copied)."""
        self.history.push(self.layers.snapshot())

    def _after_edit(self) -> None:
        self._update_display()
        self._update_undo_redo_state()
        self._refresh_layers_panel()

    def undo(self) -> None:
        if self.layers is None:
            return
        self.deselect()
        previous = self.history.undo(self.layers.snapshot())
        if previous is None:
            return
        self.layers = previous
        self._after_edit()
        self.status_bar.showMessage("Undo", 2000)

    def redo(self) -> None:
        if self.layers is None:
            return
        self.deselect()
        following = self.history.redo(self.layers.snapshot())
        if following is None:
            return
        self.layers = following
        self._after_edit()
        self.status_bar.showMessage("Redo", 2000)

    def _update_undo_redo_state(self) -> None:
        """Grise les boutons Undo / Redo quand il n'y a rien à annuler / rétablir."""
        self.undo_act.setEnabled(self.history.can_undo())
        self.redo_act.setEnabled(self.history.can_redo())

    # ------------------------------------------------------------------
    # Layers panel
    # ------------------------------------------------------------------
    def _create_layers_panel(self):
        """
        Side bar on the right of the image (fixed width, not movable),
        shown / hidden with the "Layers" button of the ribbon (or F7).
        """
        self.layers_panel = LayersPanel(self)
        self.layers_panel.setFixedWidth(220)
        self.layers_panel.hide()

        # Size / opacity bar on the left, shown when a drawing tool is selected
        self.tool_options = ToolOptionsBar(self)
        self.tool_options.set_size(self.brush_size)
        self.tool_options.set_opacity(round(self.brush_opacity * 100))
        self.tool_options.size_changed.connect(self.set_brush_size)
        self.tool_options.opacity_changed.connect(self.set_brush_opacity)

        # Central area = [ tool options | canvas | layers panel ]
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tool_options)
        layout.addWidget(self.canvas, 1)       # the canvas takes all the remaining space
        layout.addWidget(self.layers_panel)
        self.setCentralWidget(central)

        # Checkable button that shows / hides the panel
        self.layers_act = self._make_action("Layers", None, "layers",
                                            "Show / hide the layers panel (F7)", "F7")
        self.layers_act.setCheckable(True)
        self.layers_act.toggled.connect(self.layers_panel.setVisible)

        self.layers_panel.add_requested.connect(self.add_layer)
        self.layers_panel.delete_requested.connect(self.delete_layer)
        self.layers_panel.active_changed.connect(self._on_active_layer_changed)
        self.layers_panel.visibility_changed.connect(self._on_layer_visibility_changed)
        self.layers_panel.opacity_changed.connect(self._on_layer_opacity_changed)
        self.layers_panel.renamed.connect(self._on_layer_renamed)

    def _refresh_layers_panel(self) -> None:
        if hasattr(self, "layers_panel"):
            self.layers_panel.set_layers(self.layers)

    def add_layer(self) -> None:
        if self.layers is None:
            return
        self._push_history()
        layer = self.layers.add_layer()
        self._after_edit()
        self.status_bar.showMessage(f"Added {layer.name}", 2000)

    def delete_layer(self) -> None:
        if self.layers is None:
            return
        if len(self.layers) == 1:
            QMessageBox.warning(self, "Warning", "The image needs at least one layer.")
            return
        self._push_history()
        name = self.layers.active_layer.name
        self.layers.delete_layer(self.layers.active)
        self._after_edit()
        self.status_bar.showMessage(f"Deleted {name}", 2000)

    def _on_active_layer_changed(self, index: int) -> None:
        if self.layers is not None and 0 <= index < len(self.layers):
            self.layers.active = index
            self.status_bar.showMessage(f"Drawing on: {self.layers.active_layer.name}", 2000)

    def _on_layer_visibility_changed(self, index: int, visible: bool) -> None:
        layer = self.layers.layers[index]
        if layer.visible != visible:
            layer.visible = visible
            self._update_display()

    def _on_layer_opacity_changed(self, index: int, percent: int) -> None:
        self.layers.layers[index].opacity = percent / 100
        self._update_display()

    def _on_layer_renamed(self, index: int, name: str) -> None:
        name = name.strip()
        if name:
            self.layers.layers[index].name = name

    def _create_bottom_toolbar(self):
        self.bottom_toolbar = QToolBar("Quick Tools", self)
        self.bottom_toolbar.setIconSize(QSize(24, 24))
        self.bottom_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.bottom_toolbar)

        # Séparateur élastique pour repousser le zoom à droite
        # (les rotations sont maintenant dans le ruban, groupe "Image")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.bottom_toolbar.addWidget(spacer)

        # Contrôle de Zoom à droite
        self.zoom_control = ZoomControl(self)
        self.bottom_toolbar.addWidget(self.zoom_control)
        self.zoom_control.setEnabled(False)  # grisé tant qu'aucune image n'est ouverte

        # Connexions Canvas <-> ZoomControl
        self.zoom_control.zoom_in_requested.connect(self.canvas.zoom_in)
        self.zoom_control.zoom_out_requested.connect(self.canvas.zoom_out)
        self.zoom_control.fit_requested.connect(self.canvas.fit_to_window)
        self.zoom_control.zoom_percent_changed.connect(self.canvas.set_zoom_percentage)
        self.canvas.zoom_changed.connect(self.zoom_control.set_percent)

    def _update_display(self):
        if self.layers is not None:
            image = self._composite()
            pixmap = QPixmap.fromImage(numpy_to_qimage(image))
            h, w, _ = image.shape
            if (h, w) == self._displayed_size:
                # Same size: just replace the picture and keep the user's zoom
                self.canvas.show_preview(pixmap)
            else:
                # New size (open, new, rotate, crop, resize...): fit to the window
                self.canvas.set_pixmap(pixmap)
                self._displayed_size = (h, w)
            self.zoom_control.setEnabled(True)
            self.status_bar.showMessage(f"Image: {w}x{h} px | Layers: {len(self.layers)}")

    def new_image(self) -> None:
        width, ok = QInputDialog.getInt(self, "New image", "Width (px):", 800, 1, 10000)
        if not ok:
            return
        height, ok = QInputDialog.getInt(self, "New image", "Height (px):", 600, 1, 10000)
        if not ok:
            return

        self.current_image = new_img(width, height)
        self.current_file_path = None  # not saved yet -> Save will ask for a file name
        self.history.clear()
        self._update_undo_redo_state()
        self._update_display()
        self.status_bar.showMessage(f"New image: {width}x{height} px")

    def show_properties(self) -> None:
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        props = img_properties(self._composite(), self.current_file_path)
        if self.current_file_path is None:
            props["File"] = "Not saved yet"
        text = "\n".join(f"{key}: {value}" for key, value in props.items())
        QMessageBox.information(self, "Image Properties", text)

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        try:
            self.current_image = read_img(file_path)
            self.current_file_path = file_path
            # Nouvelle image : on repart d'un historique vide
            self.history.clear()
            self._update_undo_redo_state()
            
            # Au lieu de self.canvas.display_image(self.current_image) :
            self._update_display()
            
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        if self.current_file_path:
            self._write_image_to_file(self.current_file_path)
        else:
            self.save_image_as()

    def save_image_as(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image As", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp)"
        )
        if not file_path:
            return

        # Si l'utilisateur n'a pas tapé d'extension, on prend celle du filtre choisi
        if not os.path.splitext(file_path)[1]:
            if "JPEG" in selected_filter:
                file_path += ".jpg"
            elif "Bitmap" in selected_filter:
                file_path += ".bmp"
            else:
                file_path += ".png"

        # On ne retient le nouveau chemin que si l'enregistrement a réussi
        if self._write_image_to_file(file_path):
            self.current_file_path = file_path

    def _write_image_to_file(self, path: str) -> bool:
        """Enregistre l'image courante. Renvoie True si ça a marché."""
        try:
            # save_img(image, path) : l'image NumPy d'abord, le chemin ensuite
            # All visible layers are merged into one image when saving
            save_img(self._composite(), path)
            self.status_bar.showMessage(f"Saved: {os.path.basename(path)}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
            return False

    def _apply_edit(self, operation, *args, erase=False):
        """
        Apply operation(image, *args) and save the previous state for undo.
        - Geometric operations (rotate, flip, resize, crop) are applied to all layers.
        - Anything else (brush, shapes, fill...) draws on the active layer only;
          with erase=True the painted pixels become transparent instead (eraser).
        """
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        try:
            if operation in GEOMETRIC_OPS:
                new_stack = self.layers.snapshot()
                new_stack.apply_to_all(operation, args)
            else:
                rgb, alpha = self.layers.edit_active(operation, args, erase)
        except ValueError as error:
            QMessageBox.warning(self, "Warning", str(error))
            return

        self._push_history()
        if operation in GEOMETRIC_OPS:
            self.deselect()
            self.layers = new_stack
        else:
            self.layers.active_layer.rgb, self.layers.active_layer.alpha = rgb, alpha
        self._after_edit()

    def _show_edit_preview(self, operation, args, erase=False):
        """Live preview of a drawing operation on the active layer, without saving it."""
        if self.layers is None:
            return
        try:
            override = self.layers.edit_active(operation, args, erase)
        except ValueError:
            return
        self._show_image(self.layers.composite(override))

    def _on_tool_pressed(self, tool, pos):
        if tool in SHAPE_TOOLS:
            self._shape_start = pos
            self._shape_view = None
        elif tool == TOOL_CROP:
            self._crop_start = pos
        elif tool in (TOOL_BRUSH, TOOL_ERASER):
            self._begin_stroke(tool, pos)
        elif tool == TOOL_PICKER:
            self._pick_color(pos)
        elif tool == TOOL_FILL:
            self._fill_at(pos)
        elif tool == TOOL_SELECT:
            if self._inside_selection(pos):
                self._begin_move(pos)
            else:
                self.deselect()
                self._pick_layer_at(pos)
                piece = self._piece_box_at(pos)
                if piece is not None:
                    self._set_selection(piece)
                    self._begin_move(pos)
                else:
                    self._select_start = pos

    def _on_tool_dragged(self, tool, pos):
        if tool in SHAPE_TOOLS and self._shape_start is not None:
            self._preview_shape(tool, pos)
        elif tool == TOOL_CROP and self._crop_start is not None:
            self.canvas.show_selection(self._crop_start, pos)
        elif tool in (TOOL_BRUSH, TOOL_ERASER) and self._stroke_mask is not None:
            self._continue_stroke(pos)
        elif tool == TOOL_SELECT and self._lifted is not None:
            self._preview_move(pos)
        elif tool == TOOL_SELECT and self._select_start is not None:
            self.canvas.show_selection(self._select_start, pos)

    def _on_tool_released(self, tool, pos):
        if tool in SHAPE_TOOLS and self._shape_start is not None:
            self._apply_edit(self._draw_current_shape, tool, pos)
            self._shape_start = None
            self._shape_view = None
            return
        if tool in (TOOL_BRUSH, TOOL_ERASER) and self._stroke_mask is not None:
            self._end_stroke()
            return
        if tool == TOOL_SELECT:
            self._end_select(pos)
            return
        if tool != TOOL_CROP or self._crop_start is None:
            return
        start, self._crop_start = self._crop_start, None
        self.canvas.clear_selection()
        box = (start.x(), start.y(), pos.x(), pos.y())
        self._apply_edit(crop, box)

    def _draw_current_shape(self, image, tool, end):
        start = (self._shape_start.x(), self._shape_start.y())
        return draw_shape(image, tool, start, (end.x(), end.y()),
                          self.brush_color, self.brush_size)

    def _show_image(self, image):
        self.canvas.show_preview(QPixmap.fromImage(numpy_to_qimage(image)))

    def _update_fill_icon(self):
        swatch = QPixmap(32, 32)
        swatch.fill(QColor(*self.fill_color))
        self.fill_color_act.setIcon(QIcon(swatch))
        if hasattr(self, "color_palette"):   # keep the Colors block in sync
            self.color_palette.set_color(SLOT_FILL, self.fill_color)

    def choose_fill_color(self):
        color = QColorDialog.getColor(QColor(*self.fill_color), self, "Fill color")
        if color.isValid():
            self.fill_color = (color.red(), color.green(), color.blue())
            self._update_fill_icon()

    def _fill_at(self, pos):
            x, y = int(pos.x()), int(pos.y())
            height, width = self.current_image.shape[:2]
            if 0 <= x < width and 0 <= y < height:  # ignore clicks outside the image
                self._apply_edit(flood_fill, x, y, self.fill_color, 60)

    def _pick_color(self, pos):
        try:
            color = get_pixel_color(self._composite(), pos.x(), pos.y())  # colour as seen on screen
        except ValueError:
            return  # clicked outside the image: do nothing
        self.brush_color = color
        self._update_color_icon()
        self.status_bar.showMessage(f"Picked color RGB{color}", 3000)
        self.brush_tool_act.trigger()

    def _stroke_style(self, tool):
        if tool == TOOL_ERASER:
            return (255, 255, 255), self.brush_size
        return self.brush_color, self.brush_size

    def _update_color_icon(self):
        swatch = QPixmap(32, 32)
        swatch.fill(QColor(*self.brush_color))
        self.color_act.setIcon(QIcon(swatch))
        if hasattr(self, "color_palette"):   # keep the Colors block in sync (e.g. after the picker)
            self.color_palette.set_color(SLOT_COLOR, self.brush_color)

    def _update_size_icon(self):
        icon = QPixmap(32, 32)
        icon.fill(Qt.GlobalColor.transparent)
        diameter = max(3, min(self.brush_size, 30))
        painter = QPainter(icon)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().color(QPalette.ColorRole.WindowText))
        painter.drawEllipse((32 - diameter) / 2, (32 - diameter) / 2, diameter, diameter)
        painter.end()
        self.size_act.setIcon(QIcon(icon))

    def choose_brush_color(self):
        color = QColorDialog.getColor(QColor(*self.brush_color), self, "Color")
        if color.isValid():  # False if the user pressed Cancel
            self.brush_color = (color.red(), color.green(), color.blue())
            self._update_color_icon()

    def choose_brush_size(self):
        size, ok = QInputDialog.getInt(self, "Size", "Size (px):", self.brush_size, 1, 200)
        if ok:
            self.set_brush_size(size)

    def set_brush_size(self, size: int) -> None:
        """Single place that changes the size (dialog, slider or typed value)."""
        self.brush_size = size
        self.size_act.setText(f"Size: {size}")
        self._update_size_icon()
        self.canvas.set_brush_size(size)
        self.tool_options.set_size(size)

    def set_brush_opacity(self, percent: int) -> None:
        self.brush_opacity = percent / 100
        self.tool_options.set_opacity(percent)

    # ------------------------------------------------------------------
    # Brush / eraser strokes
    # ------------------------------------------------------------------
    # While the mouse moves, only the new piece of the stroke (from the previous mouse
    # position to the new one) is drawn, and only its small rectangle of the screen is
    # recomputed. This keeps drawing smooth whatever the image size, the length of the
    # stroke or the number of layers.
    #
    # The stroke is first drawn as a coverage mask (0 = untouched, 255 = fully covered,
    # with soft anti-aliased edges), then mixed with the active layer using the colour
    # and the opacity of the brush (or removed from the layer for the eraser).

    def _begin_stroke(self, tool, pos):
        if self.layers is None:
            return
        self._stroke_tool = tool
        self._stroke_mask = np.zeros(self.layers.size, dtype=np.uint8)
        self._stroke_view = self._composite().copy()   # what the screen shows during the stroke
        self._stroke_box = None                        # rectangle covered by the stroke so far
        point = (pos.x(), pos.y())
        self._stroke_points = [point]
        self._add_stroke_segment(point, point)

    def _continue_stroke(self, pos):
        point = (pos.x(), pos.y())
        self._add_stroke_segment(self._stroke_points[-1], point)
        self._stroke_points.append(point)

    def _end_stroke(self):
        box, self._stroke_box = self._stroke_box, None
        if box is not None:
            y0, y1, x0, x1 = box
            region = (slice(y0, y1), slice(x0, x1))
            rgb, alpha = self._stroke_result(region)
            layer = self.layers.active_layer
            new_rgb, new_alpha = layer.rgb.copy(), layer.alpha.copy()
            new_rgb[region], new_alpha[region] = rgb, alpha
            self._push_history()
            layer.rgb, layer.alpha = new_rgb, new_alpha
            # The preview shown during the stroke is exactly the final image: reuse it
            self.layers.remember_composite(self._stroke_view)
            self._after_edit()
        self._stroke_mask = None
        self._stroke_view = None
        self._stroke_points = []

    def _add_stroke_segment(self, p0, p1):
        height, width = self.layers.size
        margin = self.brush_size // 2 + 2
        x0 = max(0, int(min(p0[0], p1[0])) - margin)
        x1 = min(width, int(max(p0[0], p1[0])) + margin + 1)
        y0 = max(0, int(min(p0[1], p1[1])) - margin)
        y1 = min(height, int(max(p0[1], p1[1])) + margin + 1)
        if x0 >= x1 or y0 >= y1:
            return  # completely outside the image
        region = (slice(y0, y1), slice(x0, x1))

        # Draw the new piece on the coverage mask with the project's brush function
        points = [(p0[0] - x0, p0[1] - y0)]
        if p1 != p0:
            points.append((p1[0] - x0, p1[1] - y0))
        mask = self._stroke_mask[region]
        drawn = draw_brush_stroke(np.dstack([mask] * 3), points, (255, 255, 255), self.brush_size)
        self._stroke_mask[region] = np.maximum(mask, drawn[:, :, 0])

        # Update only this rectangle of the screen
        rgb, alpha = self._stroke_result(region)
        self._stroke_view[region] = self.layers.blend_region(region, rgb, alpha)
        self._show_image(self._stroke_view)

        if self._stroke_box is None:
            self._stroke_box = (y0, y1, x0, x1)
        else:
            b = self._stroke_box
            self._stroke_box = (min(b[0], y0), max(b[1], y1), min(b[2], x0), max(b[3], x1))

    def _stroke_result(self, region):
        """Active layer inside `region` with the current stroke applied: (rgb, alpha)."""
        layer = self.layers.active_layer
        coverage = self._stroke_mask[region].astype(np.float32)[:, :, None] / 255.0
        alpha = layer.alpha[region].astype(np.float32)[:, :, None] / 255.0

        if self._stroke_tool == TOOL_ERASER:
            # Eraser: make the layer transparent where the stroke passes
            new_alpha = alpha * (1.0 - coverage)
            return layer.rgb[region], (new_alpha[:, :, 0] * 255 + 0.5).astype(np.uint8)

        # Brush: paint the colour "over" the layer with the chosen opacity
        paint = coverage * self.brush_opacity
        color = np.array(self.brush_color, dtype=np.float32)
        rgb = layer.rgb[region].astype(np.float32)
        new_alpha = paint + alpha * (1.0 - paint)
        new_rgb = (color * paint + rgb * alpha * (1.0 - paint)) / np.maximum(new_alpha, 1e-6)
        return (np.clip(new_rgb + 0.5, 0, 255).astype(np.uint8),
                (new_alpha[:, :, 0] * 255 + 0.5).astype(np.uint8))

    def _preview_shape(self, tool, pos):
        """Live preview of a shape: only the rectangle around the shape is recomputed."""
        if self.layers is None:
            return
        if self._shape_view is None:
            self._shape_view = self._composite()
        height, width = self.layers.size
        sx, sy = self._shape_start.x(), self._shape_start.y()
        ex, ey = pos.x(), pos.y()
        margin = self.brush_size + 3
        x0 = max(0, int(min(sx, ex)) - margin)
        x1 = min(width, int(max(sx, ex)) + margin + 1)
        y0 = max(0, int(min(sy, ey)) - margin)
        y1 = min(height, int(max(sy, ey)) + margin + 1)
        view = self._shape_view.copy()
        if x0 < x1 and y0 < y1:
            region = (slice(y0, y1), slice(x0, x1))

            def draw(image):
                return draw_shape(image, tool, (sx - x0, sy - y0), (ex - x0, ey - y0),
                                  self.brush_color, self.brush_size)

            try:
                rgb, alpha = self.layers.edit_region(draw, region)
                view[region] = self.layers.blend_region(region, rgb, alpha)
            except ValueError:
                return
        self._show_image(view)


    # Select, Cut, Copy, Paste
    def _set_selection(self, box) -> None:
        self._selection = box
        left, top, right, bottom = box
        self.canvas.show_selection(QPointF(left, top), QPointF(right, bottom))

    def deselect(self) -> None:
        self._floating = None
        self._selection = None
        self._select_start = None
        self.canvas.clear_selection()

    def select_all(self) -> None:
        if self.layers is None:
            return
        height, width = self.layers.size
        self.select_tool_act.setChecked(True)
        self._on_tool_clicked(self.select_tool_act)
        self._set_selection((0, 0, width, height))

    def _pick_layer_at(self, pos) -> None:
        x, y = int(pos.x()), int(pos.y())
        height, width = self.layers.size
        if not (0 <= x < width and 0 <= y < height):
            return
        for index in range(len(self.layers) - 1, -1, -1):  # from the top layer down
            layer = self.layers.layers[index]
            if layer.visible and layer.opacity > 0 and layer.alpha[y, x] > 0:
                if index != self.layers.active:
                    self.layers.active = index
                    self._refresh_layers_panel()
                    self.status_bar.showMessage(f"Selecting area on: {layer.name}", 2000)
                return
                if self.layers.active != 0:
                    self.layers.active = 0
                    self._refresh_layers_panel()

    def _piece_at(self, pos) -> bool:
        if self.layers is None:
            return False
        x, y = int(pos.x()), int(pos.y())
        height, width = self.layers.size
        if not (0 <= x < width and 0 <= y < height):
            return False
        for layer in self.layers.layers[1:]:
            if layer.visible and layer.opacity > 0 and layer.alpha[y, x] > 0:
                return True
        return False

    def _piece_box_at(self, pos):
        layer = self.layers.active_layer
        x, y = int(pos.x()), int(pos.y())
        height, width = self.layers.size
        if self.layers.active == 0 or not (0 <= x < width and 0 <= y < height) or layer.alpha[y, x] == 0:
            return None
        rows = np.flatnonzero(layer.alpha.any(axis=1))
        cols = np.flatnonzero(layer.alpha.any(axis=0))
        return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1

    def _inside_selection(self, pos) -> bool:
        if self._selection is None:
            return False
        left, top, right, bottom = self._selection
        return left <= pos.x() < right and top <= pos.y() < bottom

    def _end_select(self, pos) -> None:
        if self._lifted is not None:
            self._end_move(pos)
            return
        if self._select_start is None:
            return
        start, self._select_start = self._select_start, None
        try:
            box = selection_box(self.current_image, (start.x(), start.y(), pos.x(), pos.y()))
        except ValueError:
            self.deselect()
            return
        self._set_selection(box)
        self.status_bar.showMessage(f"Area Selected {box[2] - box[0]}x{box[3] - box[1]} pixels "
                                    "(Copy: Ctrl + C, Cut: Ctrl + X, or drag inside to move)", 4000)

    @staticmethod
    def _alpha3(alpha):
        return np.dstack([alpha] * 3)

    def _selected_pixels(self):
        layer = self.layers.active_layer
        rgb = crop(layer.rgb, self._selection)
        alpha = crop(self._alpha3(layer.alpha), self._selection)[:, :, 0]
        return rgb, alpha

    def _layer_without_selection(self):
        layer = self.layers.active_layer
        rgb = empty_region(layer.rgb, self._selection, (255, 255, 255))
        alpha = empty_region(self._alpha3(layer.alpha), self._selection, (0, 0, 0))[:, :, 0]
        return rgb, alpha

    def _begin_move(self, pos) -> None:
        if self._floating is not None and self.layers.active_layer.rgb is self._floating[0]:
            self._lifted, self._lift_origin = self._floating[1], self._floating[2]
        else:
            self._lift_origin = self._selection[:2]
            piece_rgb, piece_alpha = self._selected_pixels()
            base_rgb, base_alpha = self._layer_without_selection()
            self._lifted = (piece_rgb, self._alpha3(piece_alpha), base_rgb, self._alpha3(base_alpha))
        self._move_start = pos

    def _moved_layer(self, pos):
        piece_rgb, piece_alpha3, base_rgb, base_alpha3 = self._lifted
        dx = int(round(pos.x() - self._move_start.x()))
        dy = int(round(pos.y() - self._move_start.y()))
        left, top = self._lift_origin[0] + dx, self._lift_origin[1] + dy
        rgb = paste_img(base_rgb, piece_rgb, left, top)
        alpha = np.ascontiguousarray(paste_img(base_alpha3, piece_alpha3, left, top)[:, :, 0])
        return rgb, alpha, dx, dy

    def _preview_move(self, pos) -> None:
        rgb, alpha, dx, dy = self._moved_layer(pos)
        self._show_image(self.layers.composite((rgb, alpha)))
        left, top = self._lift_origin[0] + dx, self._lift_origin[1] + dy
        piece_h, piece_w = self._lifted[0].shape[:2]
        self.canvas.show_selection(QPointF(left, top), QPointF(left + piece_w, top + piece_h))

    def _end_move(self, pos) -> None:
        rgb, alpha, dx, dy = self._moved_layer(pos)
        lifted, self._lifted = self._lifted, None
        if dx == 0 and dy == 0:
            self._set_selection(self._selection)
            return
        new_origin = (self._lift_origin[0] + dx, self._lift_origin[1] + dy)
        piece_rgb, piece_alpha3, base_rgb, base_alpha3 = lifted

        self._push_history()
        layer = self.layers.active_layer
        if self.layers.active == 0:
            layer.rgb, layer.alpha = base_rgb, np.ascontiguousarray(base_alpha3[:, :, 0])
            height, width = self.layers.size
            empty = np.zeros((height, width, 3), np.uint8)
            lifted = (piece_rgb, piece_alpha3, empty, empty)
            rgb = paste_img(empty, piece_rgb, *new_origin)
            alpha = np.ascontiguousarray(paste_img(empty, piece_alpha3, *new_origin)[:, :, 0])
            layer = self.layers.add_layer("Piece")
        layer.rgb, layer.alpha = rgb, alpha
        self._after_edit()

        # Remember the piece and where it is now, in case it is moved again
        piece_h, piece_w = piece_rgb.shape[:2]
        try:  # the selection follows the pixels (cut off at the image border)
            self._set_selection(selection_box(rgb, (new_origin[0], new_origin[1],
                                                    new_origin[0] + piece_w, new_origin[1] + piece_h)))
            self._floating = (rgb, lifted, new_origin)
        except ValueError:
            self.deselect()  # moved completely outside the image
    def _need_selection(self) -> bool:
        if self.layers is None or self._selection is None:
            self.status_bar.showMessage("Choose an area using the Select Tool", 3000)
            return False
        return True

    def copy_selection(self) -> None:
        if not self._need_selection():
            return
        self._clipboard = self._selected_pixels()
        self._clipboard_pos = self._selection[:2]
        height, width = self._clipboard[1].shape
        self.status_bar.showMessage(f"Copied area: {width}x{height} pixels", 3000)

    def delete_selection(self) -> None:
        if not self._need_selection():
            return
        rgb, alpha = self._layer_without_selection()
        self._push_history()
        self.layers.active_layer.rgb, self.layers.active_layer.alpha = rgb, alpha
        self.deselect()
        self._after_edit()

    def cut_selection(self) -> None:
        if not self._need_selection():
            return
        self.copy_selection()
        self.delete_selection()
        self.status_bar.showMessage("Cut", 3000)

    def paste_clipboard(self) -> None:
        if self.layers is None:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        if self._clipboard is None:
            self.status_bar.showMessage("Nothing to paste. Copy an area first (Ctrl+C)", 3000)
            return
        piece_rgb, piece_alpha = self._clipboard
        left, top = self._selection[:2] if self._selection is not None else self._clipboard_pos
        height, width = self.layers.size
        rgb = paste_img(np.zeros((height, width, 3), np.uint8), piece_rgb, left, top)
        alpha = paste_img(np.zeros((height, width, 3), np.uint8), self._alpha3(piece_alpha), left, top)

        self._push_history()
        layer = self.layers.add_layer("Pasted")
        layer.rgb, layer.alpha = rgb, np.ascontiguousarray(alpha[:, :, 0])
        self._after_edit()

        self.hand_tool_act.setChecked(True)
        self._on_tool_clicked(self.hand_tool_act)
        piece_h, piece_w = piece_alpha.shape
        try:
            self._set_selection(selection_box(rgb, (left, top, left + piece_w, top + piece_h)))
        except ValueError:
            self.deselect()
        self.status_bar.showMessage("New layer created. Drag to move the image", 4000)


    def resize_image(self):
            if self.current_image is None:
                QMessageBox.warning(self, "Warning", "Open an image first.")
                return

            percent, ok = QInputDialog.getInt(self, "Resize", "Scale (%):", 50, 1, 400)
            if ok:
                # 1. Applique le redimensionnement NumPy
                self._apply_edit(resize_scale, percent / 100)
                
                # 2. Réinitialise l'affichage à l'échelle 1:1 (100 %)
                #    pour visualiser la nouvelle taille réelle sans l'étirer
                if hasattr(self.canvas, "set_zoom_percentage"):
                    self.canvas.set_zoom_percentage(100)
                elif hasattr(self.canvas, "reset_zoom"):
                    self.canvas.reset_zoom()
