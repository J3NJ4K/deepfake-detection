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


def smart_cut_mask(image, box, iterations=5):
    left, top, right, bottom = selection_box(image, box)
    if right - left < 10 or bottom - top < 10:
        raise ValueError("Try selecting a larger area")

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
        raise ValueError("Unable to cut out the object. Include some background")

    found = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(found, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not detect an object")
    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1, offset=(x0, y0))
    outline = [(int(x) + x0, int(y) + y0) for x, y in largest.reshape(-1, 2)]
    return mask, outline