import cv2
import numpy as np
from core.validation import check_img

def selection_box(image, box):
    check_img(image)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if left >= right or top >= bottom:
        raise ValueError("No area selected")
    return left, top, right, bottom


def empty_region(image, box, color=(255,255,255)):
    left, top, right, bottom = selection_box(image, box)
    result = image.copy()
    result[top:bottom, left:right] = color
    return result


def paste_img(image, piece, x, y, mask=None):
    check_img(image)
    check_img(piece)
    result = image.copy()
    height, width = image.shape[:2]
    piece_h, piece_w = piece.shape[:2]
    x, y = int(round(x)), int(round(y))

    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + piece_w), min(height, y + piece_h)
    if left < right and top < bottom:
        part = piece[top - y:bottom - y, left - x:right - x]
        if mask is None:
            result[top:bottom, left:right] = part
        else:
            keep = mask[top - y:bottom - y, left - x:right - x] > 0
            result[top:bottom, left:right][keep] = part[keep]
    return result


def lasso_mask(image, points):
    check_img(image)
    if len(points) < 3:
        raise ValueError("Draw a shape around the area to select it")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    outline = np.array(points, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(mask, [outline], 255)
    if not mask.any():
        raise ValueError("No area selected")
    return mask


def mask_box(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("No area selected")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def fill_mask(image, mask, color=(255, 255, 255)):
    check_img(image)
    result = image.copy()
    result[mask > 0] = color
    return result


def smart_cut_mask(image, box, iterations=5):
    left, top, right, bottom = selection_box(image, box)
    if right - left < 10 or bottom - top < 10:
        raise ValueError("Draw a bigger box around the object")

    height, width = image.shape[:2]
    margin_x, margin_y = max(10, (right - left) // 4), max(10, (bottom - top) // 4)
    x0, y0 = max(0, left - margin_x), max(0, top - margin_y)
    x1, y1 = min(width, right + margin_x), min(height, bottom + margin_y)
    part = np.ascontiguousarray(image[y0:y1, x0:x1])

    labels = np.zeros(part.shape[:2], np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    object_model = np.zeros((1, 65), np.float64)
    rect = (left - x0, top - y0, right - left, bottom - top)
    try:
        cv2.grabCut(part, labels, rect, background_model, object_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        raise ValueError("Could not cut out the object: leave some background around it")

    found = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(found, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object found in the box")
    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1, offset=(x0, y0))
    outline = [(int(x) + x0, int(y) + y0) for x, y in largest.reshape(-1, 2)]
    return mask, outline