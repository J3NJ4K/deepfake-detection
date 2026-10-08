import os
import cv2
import numpy as np
from PySide6.QtGui import QIcon, QImage
from PySide6 import QtSvg

def load_icon(name: str) -> QIcon | None:
    # Si assets est à la racine du projet :
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")

    for ext in (".svg", ".png"):
        path = os.path.join(assets_dir, f"{name}{ext}")
        if os.path.exists(path):
            print(f"[OK] Icône trouvée : {path}")
            return QIcon(path)
    
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