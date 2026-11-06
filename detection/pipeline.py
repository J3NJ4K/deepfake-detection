import numpy as np
from detection.face_detector import detect_faces
from detection.preprocess import crop_box, crop_face, to_tensor
from detection.classifier import load_model, predict
from detection.gradcam import grad_cam
from detection.commfor import load_commfor, predict_commfor, to_commfor_tensor

GENERATED_THRESHOLD = 0.5

_xception = None
_commfor = None

def analyze_deepfake(image: np.ndarray) -> dict:
    # loads the models the first time they're needed, so the app starts fast
    global _xception, _commfor
    if _xception is None:
        _xception = load_model()
    if _commfor is None:
        _commfor = load_commfor()

    # for each detected face: crops it, converts to a tensor and gets the manipulation probability
    faces = []
    for bbox in detect_faces(image):
        face = crop_face(image, bbox)
        tensor = to_tensor(face)
        faces.append({
            "bbox": bbox,
            "manipulation_probability": predict(_xception, tensor),
            "heatmap": grad_cam(_xception, tensor),
            "crop_box": crop_box(image, bbox),
        })

    # commfor only on the whole image
    return {
        "generated_probability": predict_commfor(_commfor, to_commfor_tensor(image)),
        "faces": faces,
    }
