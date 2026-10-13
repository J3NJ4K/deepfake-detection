import numpy as np
from detection.face_detector import detect_faces
from detection.preprocess import crop_face, to_tensor
from detection.classifier import load_model, predict

_model = None


def analyze_deepfake(image: np.ndarray) -> list[dict]:
    """RGB uint8 inn, one dict per face out: {"bbox": (x1, y1, x2, y2), "manipulation_probability": float}"""
    # loads the model the first time it's needed, so the app starts fast
    global _model
    if _model is None:
        _model = load_model()

    # for each detected face: crops it, converts to a tensor and gets the probability
    results = []
    for bbox in detect_faces(image):
        prob = predict(_model, to_tensor(crop_face(image, bbox)))
        results.append({"bbox": bbox, "manipulation_probability": prob})
    return results
