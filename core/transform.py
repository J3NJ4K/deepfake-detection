import cv2
from core.validation import check_img

def rotate_left(image):
    check_img(image)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_right(image):
    check_img(image)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def flip_horizontal(image):
    check_img(image)
    return cv2.flip(image, 1)

def flip_vertical(image):
    check_img(image)
    return cv2.flip(image, 0)

def flip_diagonal(image):
    check_img(image)
    return cv2.transpose(image)

def rotate_180(image):
    check_img(image)
    return cv2.rotate(image, cv2.ROTATE_180)

def crop(image, bbox):
    check_img(image)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)

    # sort the corners in the right order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # keep the box inside the image
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox} for image size {width}x{height}")
    return image[y1:y2, x1:x2].copy()


def resize(image, new_width, new_height):
    check_img(image)
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Invalid image size: {new_width}x{new_height}")

    height, width = image.shape[:2]
    if new_width * new_height < width * height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(image, (int(new_width), int(new_height)), interpolation = interpolation)


def resize_scale(image, scale):
    check_img(image)
    if scale <= 0:
        raise ValueError(f"Invalid image scale: {scale}")
    height, width = image.shape[:2]
    return resize(image, max(1, round(width * scale)), max(1, round(height * scale)))

