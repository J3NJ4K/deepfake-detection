import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QStatusBar, QMessageBox, QToolBar, 
    QFileDialog, QWidget, QSizePolicy, QInputDialog
)
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtCore import Qt, QSize
from ui.utils import load_icon, numpy_to_qimage
from ui.canvas import ImageCanvas
from ui.zoom_control import ZoomControl
from ui.utils import load_icon, numpy_to_qimage, read_image_unicode, write_image_unicode
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

        # Composant central (Canvas)
        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - No image loaded")

        # Initialisation UI
        self._create_actions()
        self._create_menus()
        self._create_bottom_toolbar()

    def _create_actions(self):
        # --- File Menu ---
        self.open_act = QAction("&Open...", self)
        self.open_act.setShortcut(QKeySequence.StandardKey.Open)
        self.open_act.triggered.connect(self.open_image)

        self.save_act = QAction("&Save", self)
        self.save_act.setShortcut(QKeySequence.StandardKey.Save)
        self.save_act.triggered.connect(self.save_image)

        self.save_as_act = QAction("Save &As...", self)
        self.save_as_act.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_act.triggered.connect(self.save_image_as)

        self.quit_act = QAction("&Quit", self)
        self.quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_act.triggered.connect(self.close)

        # --- Filters Menu ---
        self.filter_gaussian_act = QAction("&Gaussian Filter", self)
        self.filter_gaussian_act.triggered.connect(self._action_placeholder)

        self.filter_sobel_act = QAction("&Sobel Filter", self)
        self.filter_sobel_act.triggered.connect(self._action_placeholder)

        self.filter_thresh_act = QAction("&Binary Thresholding", self)
        self.filter_thresh_act.triggered.connect(self._action_placeholder)

        # --- Detection Menu ---
        self.face_detect_act = QAction("&Detect Faces", self)
        self.face_detect_act.triggered.connect(self.run_face_detection)

        self.deepfake_act = QAction("&Run Deepfake Analysis", self)
        self.deepfake_act.triggered.connect(self.run_deepfake_analysis)

        # --- Rotation Actions ---
        icon_ccw = load_icon("rotate_left")
        self.rot_ccw_act = QAction(icon_ccw, "", self) if icon_ccw else QAction("↺", self)
        self.rot_ccw_act.setToolTip("Rotate 90° Counter-Clockwise (Ctrl+Shift+R)")
        self.rot_ccw_act.setShortcut("Ctrl+Shift+R")
        self.rot_ccw_act.triggered.connect(lambda: self._apply_edit(rotate_left))

        icon_cw = load_icon("rotate_right")
        self.rot_cw_act = QAction(icon_cw, "", self) if icon_cw else QAction("↻", self)
        self.rot_cw_act.setToolTip("Rotate 90° Clockwise (Ctrl+R)")
        self.rot_cw_act.setShortcut("Ctrl+R")
        self.rot_cw_act.triggered.connect(lambda: self._apply_edit(rotate_right))

        # --- Image Menu ---
        self.flip_h_act = QAction("Flip &Horizontal", self)
        self.flip_h_act.triggered.connect(lambda: self._apply_edit(flip_horizontal))

        self.flip_v_act = QAction("Flip &Vertical", self)
        self.flip_v_act.triggered.connect(lambda: self._apply_edit(flip_vertical))

        self.resize_act = QAction("&Resize...", self)
        self.resize_act.triggered.connect(self.resize_image)

    def _create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_act)
        file_menu.addSeparator()
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.save_as_act)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_act)

        image_menu = menubar.addMenu("&Image")
        image_menu.addAction(self.rot_ccw_act)
        image_menu.addAction(self.rot_cw_act)
        image_menu.addSeparator()
        image_menu.addAction(self.flip_h_act)
        image_menu.addAction(self.flip_v_act)
        image_menu.addSeparator()
        image_menu.addAction(self.resize_act)

# ... tes autres menus (File, Edit, Detection...) ...

        # Menu Filters
        filters_menu = self.menuBar().addMenu("&Filters")

        # Gaussian Blur
        blur_action = filters_menu.addAction("Gaussian Blur")
        blur_action.triggered.connect(self.apply_gaussian_blur)

        # Sobel Filter
        sobel_action = filters_menu.addAction("Sobel Filter (Edge Detection)")
        sobel_action.triggered.connect(self.apply_sobel_filter)

        filters_menu.addSeparator()

        # Binary Threshold
        binary_action = filters_menu.addAction("Binary Threshold")
        binary_action.triggered.connect(self.apply_binary_threshold)

        # Histogram / Otsu Threshold
        otsu_action = filters_menu.addAction(
            "Automatic Threshold (Otsu)"
        )
        otsu_action.triggered.connect(self.apply_histogram_threshold)

        detection_menu = menubar.addMenu("&Detection")
        detection_menu.addAction(self.face_detect_act)
        detection_menu.addAction(self.deepfake_act)

    def apply_gaussian_blur(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self.current_image = gaussian_blur(self.current_image, kernel_size=5)
            self._update_display()
            self.statusBar().showMessage("Gaussian Blur appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur flou: {e}", 3000)

    def apply_sobel_filter(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self.current_image = sobel_filter(self.current_image)
            self._update_display()
            self.statusBar().showMessage("Filtre Sobel appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur Sobel: {e}", 3000)

    def apply_binary_threshold(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self.current_image = binary_threshold(self.current_image, threshold=128)
            self._update_display()
            self.statusBar().showMessage("Seuillage binaire appliqué", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur seuillage: {e}", 3000)

    def apply_histogram_threshold(self):
        if self.current_image is None:
            self.statusBar().showMessage("Aucune image chargée", 3000)
            return

        try:
            self.current_image = histogram_threshold(self.current_image)
            self._update_display()
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

            self.current_image = annotated_image
            self._update_display()
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
                summary_messages.append(f"Face #{i}: {verdict} ({prob * 100:.1f}% probability)")

            # Mettre à jour l'image affichée
            self.current_image = annotated_image
            self._update_display()

            report = "\n".join(summary_messages)
            self.status_bar.showMessage("Deepfake analysis complete.")
            QMessageBox.information(self, "Deepfake Analysis Results", report)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run deepfake analysis: {e}")
            self.status_bar.showMessage("Analysis failed.")

    def _create_bottom_toolbar(self):
        self.bottom_toolbar = QToolBar("Quick Tools", self)
        self.bottom_toolbar.setIconSize(QSize(24, 24))
        self.bottom_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.bottom_toolbar)

        # 1. Rotations à gauche
        self.bottom_toolbar.addAction(self.rot_ccw_act)
        self.bottom_toolbar.addAction(self.rot_cw_act)

        # 2. Séparateur élastique pour repousser le zoom à droite
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.bottom_toolbar.addWidget(spacer)

        # 3. Contrôle de Zoom à droite
        self.zoom_control = ZoomControl(self)
        self.bottom_toolbar.addWidget(self.zoom_control)

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
            self._write_image_file(self.current_file_path)
        else:
            self.save_image_as()

    def save_image_as(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp)"
        )
        if file_path:
            self._write_image_file(file_path)
            self.current_file_path = file_path

    def _write_image_to_file(self, path: str) -> None:
        try:
            # Utilise directement save_img qui gère la conversion vers BGR en interne
            save_img(path, self.current_image)
            self.status_bar.showMessage(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def _apply_edit(self, operation, *args):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        try:
            self.current_image = operation(self.current_image, *args)
        except ValueError as error:
            QMessageBox.warning(self, "Warning", str(error))
            return
        self._update_display()

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

    def _action_placeholder(self):
        source_action = self.sender().text().replace("&", "")
        QMessageBox.information(
            self, "Feature in Progress", f"Action triggered: '{source_action}'"
        )