import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QIcon, QImage, QPainter, QPalette, QPixmap, QColor, QGuiApplication
from PySide6.QtSvg import QSvgRenderer

# Tailles pré-calculées pour les icônes (couvre les petits/grands boutons et les écrans en 125-200%)
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96)


def _theme_icon_color() -> QColor:
    """Couleur du texte du thème : blanc en thème sombre, noir en thème clair."""
    return QGuiApplication.palette().color(QPalette.ColorRole.WindowText)


def _tinted_svg_icon(path: str, color: QColor) -> QIcon:
    """
    Dessine le SVG puis le repeint entièrement dans `color` en gardant sa forme
    (sa transparence). Marche pour les icônes d'une seule couleur.
    """
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return QIcon(path)

    # Garde les proportions du SVG (au cas où il ne serait pas carré)
    view = renderer.viewBoxF()
    ratio = view.width() / view.height() if view.height() > 0 else 1.0

    icon = QIcon()
    for size in ICON_SIZES:
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)

        if ratio >= 1:
            w, h = size, size / ratio
        else:
            w, h = size * ratio, size
        target = QRectF((size - w) / 2, (size - h) / 2, w, h)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, target)
        # "SourceIn" : on remplit avec la couleur uniquement là où l'icône est dessinée
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(image.rect(), color)
        painter.end()

        icon.addPixmap(QPixmap.fromImage(image))
    return icon


def load_icon(name: str, color: QColor | None = None) -> QIcon | None:
    """
    Charge assets/<name>.svg (ou .png).
    Les SVG sont recolorés automatiquement avec la couleur du thème
    (blanc sur thème sombre, noir sur thème clair). On peut forcer une couleur
    avec `color`, par ex. load_icon("deepfake", QColor("red")).
    """
    # Si assets est à la racine du projet :
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")

    svg_path = os.path.join(assets_dir, f"{name}.svg")
    if os.path.exists(svg_path):
        return _tinted_svg_icon(svg_path, color or _theme_icon_color())

    png_path = os.path.join(assets_dir, f"{name}.png")
    if os.path.exists(png_path):
        return QIcon(png_path)  # les PNG sont gardés tels quels

    print(f"[ERREUR] Fichier introuvable pour '{name}' dans {assets_dir}")
    return None

def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convertit un tableau NumPy (H, W, 3) RGB uint8 en QImage pour affichage."""
    if array is None or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("L'image doit être un tableau NumPy de dimension (H, W, 3) RGB.")

    # Assure que le tableau est contigu en mémoire pour Qt
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)

    height, width, channels = array.shape
    bytes_per_line = channels * width

    # Format_RGB888 correspond exactement à uint8 (0-255) RGB
    qimage = QImage(
        array.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888
    )
    return qimage.copy()

def read_image_unicode(file_path: str) -> np.ndarray | None:
    """
    Lit une image depuis n'importe quel chemin Windows (accents, espaces)
    et la retourne au format NumPy RGB uint8 (H, W, 3).
    """
    try:
        raw_bytes = np.fromfile(file_path, dtype=np.uint8)
        bgr_image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return None
        # Règle d'or du groupe : conversion immédiate en RGB uint8
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def write_image_unicode(file_path: str, rgb_image: np.ndarray) -> bool:
    """
    Enregistre une image NumPy RGB uint8 vers n'importe quel dossier,
    en prenant en compte les chemins Unicode et l'extension du fichier.
    """
    try:
        # Conversion RGB vers BGR pour OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Détection de l'extension
        extension = os.path.splitext(file_path)[1].lower()
        if not extension:
            extension = ".png"
            file_path += extension

        # Encodage en mémoire
        success, buffer = cv2.imencode(extension, bgr_image)
        if not success:
            return False

        # Écriture brute sur le disque
        buffer.tofile(file_path)
        return True
    except Exception:
        return False