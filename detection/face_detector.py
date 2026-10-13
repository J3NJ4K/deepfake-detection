from pathlib import Path
import cv2
import numpy as np

_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_detector = None


def detect_faces(image: np.ndarray, score_threshold: float = 0.6) -> list[tuple[int, int, int, int]]:
    """RGB uint8 in, list with (x1, y1, x2, y2) out."""
    # creates the detector once and reuses it, but updates the image size every time
    global _detector
    h, w = image.shape[:2]

    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (w, h), score_threshold)
    _detector.setInputSize((w, h))

    # YuNet expects BGR, and returns None (not an empty array) when no faces are found
    _, faces = _detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if faces is None:
        return []

    # converts YuNet's (x, y, w, h) to (x1, y1, x2, y2) and keeps the boxes inside the image
    boxes = []
    for x, y, bw, bh in faces[:, :4]:
        boxes.append((max(0, int(x)), max(0, int(y)), min(w, int(x + bw)), min(h, int(y + bh))))
    return boxes
