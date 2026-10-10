import cv2
import numpy as np
import torch

INPUT_SIZE = 256
MARGIN = 1.3


def crop_face(image: np.ndarray, bbox: tuple[int, int, int, int], margin: float = MARGIN) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * margin
    h, w = image.shape[:2]
    nx1, ny1 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    nx2, ny2 = int(min(w, cx + size / 2)), int(min(h, cy + size / 2))
    return image[ny1:ny2, nx1:nx2]


def to_tensor(face: np.ndarray) -> torch.Tensor:
    face = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(face).float() / 255.0
    x = (x - 0.5) / 0.5
    return x.permute(2, 0, 1).unsqueeze(0)