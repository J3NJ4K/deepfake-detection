import numpy as np
from core.validation import check_img


def brightness_contrast(image, brightness=0, contrast=0):
    check_img(image)
    if not (-100 <= brightness <= 100 and -100 <= contrast <= 100):
        raise ValueError("Brightness and contrast values must be from -100 to 100")

    factor = (contrast + 100) / 100
    shift = brightness * 255 / 100
    result = (image.astype(np.float32) - 128) * factor + 128 + shift
    return np.clip(result + 0.5, 0, 255).astype(np.uint8)