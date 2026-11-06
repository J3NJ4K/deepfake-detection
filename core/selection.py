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
        raise ValueError("Draw around the area you want to select")
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