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


def paste_img(image, piece, x, y):
    check_img(image)
    check_img(piece)
    result = image.copy()
    height, width = image.shape[:2]
    piece_h, piece_w = piece.shape[:2]
    x, y = int(round(x)), int(round(y))

    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + piece_w), min(height, y + piece_h)
    if left < right and top < bottom:
        result[top:bottom, left:right] = piece[top - y:bottom - y, left - x:right - x]
    return result