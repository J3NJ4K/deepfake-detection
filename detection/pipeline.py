import numpy as np
from detection.face_detector import detect_faces
from detection.preprocess import crop_box, crop_face, to_tensor
from detection.classifier import load_model, predict
from detection.gradcam import grad_cam

_model = None

def analyze_deepfake(image: np.ndarray) -> list[dict]:
    # loads the model the first time it's needed, so the app starts fast
    global _model
    if _model is None:
        _model = load_model()

    # for each detected face: crops it, converts to a tensor and gets the probability
    results = []
    for bbox in detect_faces(image):
        tensor = to_tensor(crop_face(image, bbox))
        results.append({
            "bbox": bbox,
            "manipulation_probability": predict(_model, tensor),
            "heatmap": grad_cam(_model, tensor),
            "crop_box": crop_box(image, bbox),
        })
    return results