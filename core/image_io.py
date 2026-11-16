import os
import cv2
import numpy as np

from core.validation import check_img

def read_img(path):

    # does the file exist?
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist")

    # read the file and decode it into BGR
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # can opencv read the image?
    if bgr is None:
        raise ValueError(f"Failed to read image from {path}")

    # BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def save_img(image, path):

    # is the image valid?
    check_img(image)

    # get file extension
    ext = os.path.splitext(path)[1].lower()

    # is the file supported?
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension {ext}")

    # RGB to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # encode image
    ok, buffer = cv2.imencode(ext, bgr)

    # did the encoding work?
    if not ok:
        raise ValueError(f"Failed to encode image {path}")

    # save image
    buffer.tofile(path)


def new_img(width, height, color=(255, 255, 255)):
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    if len(color) != 3 or not all(0 <= c <= 255 for c in color):
        raise ValueError(f"Color must be (R, G, B) with values 0-255, got {color}")

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def img_properties(image, path=None):
    check_img(image)
    height, width, channels = image.shape
    props = {
        "Width": f"{width} px",
        "Height": f"{height} px",
        "Channels": f"{channels} (RGB)",
    }
    if path and os.path.isfile(path):
        props["File name"] = os.path.basename(path)
        props["Format"] = os.path.splitext(path)[1].lstrip(".").upper()
        props["File size"] = f"{os.path.getsize(path) / 1024:.1f} KB"
    return props
