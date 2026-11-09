from pathlib import Path
import cv2
import numpy as np

_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_detector = None

MAX_SIDE = 1280  # YuNet splits up or misses faces that are too large

def detect_faces(image: np.ndarray, score_threshold: float = 0.75) -> list[tuple[int, int, int, int]]:

    global _detector
    h, w = image.shape[:2]

    # detects on a smaller copy if the image is large, boxes are scaled back after
    scale = min(1.0, MAX_SIDE / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    sh, sw = small.shape[:2]

    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (sw, sh), score_threshold)
    _detector.setInputSize((sw, sh))

    # YuNet expects BGR, and returns None when no faces are found
    _, faces = _detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None:
        return []

    # back to original image coordinates, (x, y, w, h) -> (x1, y1, x2, y2), kept inside the image
    boxes = []
    for x, y, bw, bh in faces[:, :4] / scale:
        boxes.append((max(0, int(x)), max(0, int(y)), min(w, int(x + bw)), min(h, int(y + bh))))
    return boxes



