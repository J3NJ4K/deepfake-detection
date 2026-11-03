import cv2
import numpy as np

from core.validation import check_img

def draw_brush_stroke(image, points, color=(255, 0, 0), size=5):
    check_img(image)
    if size <= 0:
        raise ValueError(f"brush size must be positive, got {size}")
    if not points:
        raise ValueError("Cannot draw a stroke without any points")

    result = image.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in points]

    if len(pts) == 1:
        cv2.circle(result, pts[0], max(1, size // 2), color, -1, lineType=cv2.LINE_AA)
    else:
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(result, p1, p2, color, size, lineType=cv2.LINE_AA)
    return result
