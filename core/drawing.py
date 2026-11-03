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


SHAPES = ("rectangle", "ellipse", "line", "triangle", "right_triangle", "diamond", "star", "arrow")


def _shape_points(shape, left, top, right, bottom):
    w, h = right - left, bottom - top
    cx, cy = (left + right) / 2, (top + bottom) / 2
    if shape == "triangle":
        pts = [(cx, top), (right, bottom), (left, bottom)]
    elif shape == "right_triangle":
        pts = [(left, top), (right, bottom), (left, bottom)]
    elif shape == "diamond":
        pts = [(cx, top), (right, cy), (cx, bottom), (left, cy)]
    elif shape == "star":
        pts = []
        for i in range(10):  # 5 outer points and 5 inner points, taking turns
            angle = -np.pi / 2 + i * np.pi / 5
            r = 1.0 if i % 2 == 0 else 0.4
            pts.append((cx + r * w / 2 * np.cos(angle), cy + r * h / 2 * np.sin(angle)))
    else:  # arrow pointing right
        head = left + 0.6 * w
        pts = [(left, cy - h / 4), (head, cy - h / 4), (head, top), (right, cy),
               (head, bottom), (head, cy + h / 4), (left, cy + h / 4)]
    return np.array(pts).round().astype(np.int32)


def draw_shape(image, shape, start, end, outline_color=(255, 0, 0), thickness=3, fill_color=None):
    check_img(image)
    if shape not in SHAPES:
        raise ValueError(f"Invalid shape '{shape}'. Choose from {SHAPES}")
    if thickness <= 0:
        raise ValueError(f"Thickness must be positive, got {thickness}")

    result = image.copy()
    x1, y1 = int(round(start[0])), int(round(start[1]))
    x2, y2 = int(round(end[0])), int(round(end[1]))

    if shape == "line":
        cv2.line(result, (x1, y1), (x2, y2), outline_color, thickness, lineType=cv2.LINE_AA)
    elif shape == "rectangle":
        if fill_color is not None:
            cv2.rectangle(result, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(result, (x1, y1), (x2, y2), outline_color, thickness, lineType=cv2.LINE_AA)
    elif shape == "ellipse":
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        if fill_color is not None:
            cv2.ellipse(result, center, axes, 0, 0, 360, fill_color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(result, center, axes, 0, 0, 360, outline_color, thickness, lineType=cv2.LINE_AA)
    else:
        pts = _shape_points(shape, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if fill_color is not None:
            cv2.fillPoly(result, [pts], fill_color, lineType=cv2.LINE_AA)
        cv2.polylines(result, [pts], True, outline_color, thickness, lineType=cv2.LINE_AA)
    return result


def flood_fill(image, x, y, color, tolerance=20):
    check_img(image)
    x, y = int(x), int(y)
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Coordinates ({x}, {y}) are outside the image dimensions {width}x{height}.")
    result = image.copy()
    mask = np.zeros((height + 2, width + 2), np.uint8)
    diff = (tolerance, tolerance, tolerance)
    cv2.floodFill(result, mask, (x, y), color, loDiff=diff, upDiff=diff,
                  flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    return result
