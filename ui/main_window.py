import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QStatusBar, QMessageBox, QToolBar,
    QFileDialog, QWidget, QSizePolicy, QInputDialog
)
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap
from PySide6.QtCore import Qt, QSize
from ui.utils import load_icon, numpy_to_qimage
from ui.canvas import ImageCanvas, TOOL_HAND, TOOL_SELECT, TOOL_CROP, TOOL_BRUSH
from ui.zoom_control import ZoomControl
from ui.history import ImageHistory
from ui.ribbon import Ribbon
from core.transform import rotate_left, rotate_right, flip_horizontal, flip_vertical, resize_scale
from core.image_io import read_img, save_img
from detection.face_detector import detect_faces
from detection.pipeline import analyze_deepfake
from core.filters import (
    binary_threshold,
    gaussian_blur,
    histogram_threshold,
    sobel_filter,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Machine Vision Photo Editor & Deepfake Detector")
        self.resize(1100, 700)

        # État interne partagé : NumPy RGB uint8
        self.current_image: np.ndarray | None = None
        self.current_file_path: str | None = None

        # Historique pour Undo / Redo (20 étapes max)
        self.history = ImageHistory(max_steps=20)

        # Composant central (Canvas)
        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open an image to start (Ctrl+O)")

        # Initialisation UI
        self._create_actions()
        self._create_menus()
        self._create_ribbon()
        self._create_bottom_toolbar()
        self._update_undo_redo_state()

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
        self.open_act = self._make_action("&Open...", self.open_image,
                                          shortcut=QKeySequence.StandardKey.Open)
        self.save_act = self._make_action("&Save", self.save_image,
                                          shortcut=QKeySequence.StandardKey.Save)
        self.save_as_act = self._make_action("Save &As...", self.save_image_as,
                                             shortcut=QKeySequence.StandardKey.SaveAs)
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
        self.tool_group.setExclusive(True)

        self.hand_tool_act = self._add_tool("Hand", TOOL_HAND, "hand", "Hand: drag to scroll the image when zoomed in")
        self.select_tool_act = self._add_tool("Select", TOOL_SELECT, "select", "Select an area (coming soon)")
        self.crop_tool_act = self._add_tool("Crop", TOOL_CROP, "crop", "Crop the image (coming soon)")
        self.brush_tool_act = self._add_tool("Brush", TOOL_BRUSH, "brush", "Draw on the image (coming soon)")

        self.hand_tool_act.setChecked(True)
        for placeholder in (self.select_tool_act, self.crop_tool_act, self.brush_tool_act):
            placeholder.setEnabled(False)

        self.tool_group.triggered.connect(lambda act: self.canvas.set_tool(act.data()))

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
        file_menu.addAction(self.open_act)
        file_menu.addSeparator()
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.save_as_act)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_act)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_act)
        edit_menu.addAction(self.redo_act)

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
            boxes = detect_faces(self.current_image)
            if not boxes:
                QMessageBox.information(self, "Face Detection", "No faces detected in the image.")
                return

            # On fait une copie de travail pour dessiner sans altérer définitivement l'original si besoin
            annotated_image = self.current_image.copy()
            for (x1, y1, x2, y2) in boxes:
                # Rectangle bleu (RGB: 0, 150, 255) avec épaisseur proportionnelle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 150, 255), 2)

            self._set_image(annotated_image)
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
            results = analyze_deepfake(self.current_image)
            if not results:
                QMessageBox.information(
                    self, "Deepfake Detection", "No faces detected to analyze."
                )
                self.status_bar.showMessage("Analysis complete: No faces detected.")
                return

            annotated_image = self.current_image.copy()
            summary_messages = []

            for i, res in enumerate(results, start=1):
                x1, y1, x2, y2 = res["bbox"]
                prob = res["manipulation_probability"]
                is_fake = prob >= 0.5

                # Vert (réel) si < 50%, Rouge (deepfake) si >= 50%
                color = (255, 50, 50) if is_fake else (50, 205, 50)
                label = f"Fake: {prob * 100:.1f}%" if is_fake else f"Real: {(1 - prob) * 100:.1f}%"

                # Dessiner le cadre
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # Dessiner une étiquette texte au-dessus de la boîte
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                # Fond plein sous le texte pour la lisibilité
                cv2.rectangle(
                    annotated_image,
                    (x1, max(0, y1 - text_h - 10)),
                    (x1 + text_w, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(text_h, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

                verdict = "Manipulated (Fake)" if is_fake else "Authentic (Real)"
                summary_messages.append(f"Face #{i}: {verdict} — {prob * 100:.1f}% manipulation probability")

            # Mettre à jour l'image affichée (annulable avec Ctrl+Z)
            self._set_image(annotated_image)

            report = "\n".join(summary_messages)
            self.status_bar.showMessage("Deepfake analysis complete.")
            QMessageBox.information(self, "Deepfake Analysis Results", report)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run deepfake analysis: {e}")
            self.status_bar.showMessage("Analysis failed.")

    def _create_ribbon(self):
        """Ruban en haut de la fenêtre, style Paint : un groupe par famille de fonctions."""
        self.ribbon = Ribbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)

        history = self.ribbon.add_group("History")
        history.add_small_buttons([self.undo_act, self.redo_act])

        tools = self.ribbon.add_group("Tools")
        tools.add_small_buttons([self.hand_tool_act, self.select_tool_act,
                                 self.crop_tool_act, self.brush_tool_act])

        image = self.ribbon.add_group("Image")
        image.add_small_buttons([self.rot_ccw_act, self.rot_cw_act,
                                 self.flip_h_act, self.flip_v_act])
        image.add_large_button(self.resize_act)

        filters = self.ribbon.add_group("Filters")
        filters.add_small_buttons([self.blur_act, self.sobel_act,
                                   self.threshold_act, self.otsu_act])

        detection = self.ribbon.add_group("Detection")
        detection.add_large_button(self.face_detect_act)
        detection.add_large_button(self.deepfake_act)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------
    def _set_image(self, new_image: np.ndarray) -> None:
        """
        Remplace l'image courante en sauvegardant l'ancienne dans l'historique.
        Toute modification de l'image doit passer par ici pour être annulable.
        """
        if self.current_image is not None:
            self.history.push(self.current_image)
        self.current_image = new_image
        self._update_display()
        self._update_undo_redo_state()

    def undo(self) -> None:
        if self.current_image is None:
            return
        previous = self.history.undo(self.current_image)
        if previous is None:
            return
        self.current_image = previous
        self._update_display()
        self._update_undo_redo_state()
        self.status_bar.showMessage("Undo", 2000)

    def redo(self) -> None:
        if self.current_image is None:
            return
        following = self.history.redo(self.current_image)
        if following is None:
            return
        self.current_image = following
        self._update_display()
        self._update_undo_redo_state()
        self.status_bar.showMessage("Redo", 2000)

    def _update_undo_redo_state(self) -> None:
        """Grise les boutons Undo / Redo quand il n'y a rien à annuler / rétablir."""
        self.undo_act.setEnabled(self.history.can_undo())
        self.redo_act.setEnabled(self.history.can_redo())

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
        if self.current_image is not None:
            qimg = numpy_to_qimage(self.current_image)
            pixmap = QPixmap.fromImage(qimg)
            self.canvas.set_pixmap(pixmap)
            self.zoom_control.setEnabled(True)
            h, w, _ = self.current_image.shape
            self.status_bar.showMessage(f"Image: {w}x{h} px | Format: RGB uint8")

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
            save_img(self.current_image, path)
            self.status_bar.showMessage(f"Saved: {os.path.basename(path)}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
            return False

    def _apply_edit(self, operation, *args):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        try:
            new_image = operation(self.current_image, *args)
        except ValueError as error:
            QMessageBox.warning(self, "Warning", str(error))
            return
        self._set_image(new_image)

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
