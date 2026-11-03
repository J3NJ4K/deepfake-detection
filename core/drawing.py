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


def get_pixel_color(image, x, y):
    check_img(image)
    x, y = int(x), int(y)
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Point ({x}, {y}) is outside the {width}x{height} image")
    r, g, b = image[y, x]
    return int(r), int(g), int(b)


SHAPES = ("rectangle", "ellipse", "line")


def draw_shape(image, shape, start, end, outline_color=(255, 0, 0), thickness=3, fill_color=None):
    check_img(image)
    if shape not in SHAPES:
        raise ValueError(f"Invalid shape '{shape}'. Choose from {SHAPES}")
    if thickness <= 0:
        raise ValueError(f"Invalid thickness: {thickness}. Must be positive.")

    result = image.copy()
    x1, y1 = int(round(start[0])), int(round(start[1]))
    x2, y2 = int(round(end[0])), int(round(end[1]))

    if shape == "line":
        cv2.line(result, (x1, y1), (x2, y2), outline_color, thickness, lineType=cv2.LINE_AA)
        return result

    if shape == "rectangle":
        if fill_color is not None:
            cv2.rectangle(result, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(result, (x1, y1), (x2, y2), outline_color, thickness, lineType=cv2.LINE_AA)
    else:  # ellipse that fits inside the dragged box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        if fill_color is not None:
            cv2.ellipse(result, center, axes, 0, 0, 360, fill_color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(result, center, axes, 0, 0, 360, outline_color, thickness, lineType=cv2.LINE_AA)
    return result
