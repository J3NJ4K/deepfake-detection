import cv2
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
        self.deepfake_act = QAction("&Run Deepfake Analysis", self)
        self.deepfake_act.triggered.connect(self._action_placeholder)

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

        filters_menu = menubar.addMenu("&Filters")
        filters_menu.addAction(self.filter_gaussian_act)
        filters_menu.addAction(self.filter_sobel_act)
        filters_menu.addAction(self.filter_thresh_act)

        detection_menu = menubar.addMenu("&Detection")
        detection_menu.addAction(self.deepfake_act)

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

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*.*)"
        )
        if not file_path:
            return

        # Lecture universelle compatible avec les accents et espaces
        image = read_image_unicode(file_path)
        if image is None:
            QMessageBox.critical(self, "Error", f"Failed to load image from:\n{file_path}")
            return

        self.current_image = image
        self.current_file_path = file_path
        self._update_display()

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

    def _write_image_file(self, path: str):
        # Écriture universelle
        success = write_image_unicode(path, self.current_image)
        if success:
            self.status_bar.showMessage(f"Saved successfully: {path}", 4000)
        else:
            QMessageBox.critical(self, "Error", f"Failed to save image to:\n{path}")

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
            self._apply_edit(resize_scale, percent / 100)

    def _action_placeholder(self):
        source_action = self.sender().text().replace("&", "")
        QMessageBox.information(
            self, "Feature in Progress", f"Action triggered: '{source_action}'"
        )