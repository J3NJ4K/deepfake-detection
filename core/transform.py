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