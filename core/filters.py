import cv2
import numpy as np

from core.validation import check_img


def gaussian_blur(image, kernel_size=5, sigma=0):
    """
    Apply Gaussian blur to an RGB uint8 image.

    Args:
        image: RGB uint8 NumPy image
        kernel_size: Positive odd integer, e.g. 3, 5, 7.
        sigma: Gaussian standard deviation. 0 lets OpenCV choose it.

    Returns:
        Blurred RGB uint8 NumPy image.
    """
    check_img(image)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )

def sobel_filter(image):
    """
    Detect edges in an RGB uint8 image using the Sobel operator.

    Returns:
        Edge image as an RGB uint8 NumPy image.
    """
    check_img(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(
        grad_x.astype(np.float32),
        grad_y.astype(np.float32)
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    edges = magnitude.astype(np.uint8)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def binary_threshold(image, threshold=128):
    """
    Convert an RGB uint8 image into a binary black-and-white image.

    Args:
        image: RGB uint8 NumPy image.
        threshold: Intensity threshold between 0 and 255.

    Returns:
        Binary image as an RGB uint8 NumPy image.
    """
    check_img(image)

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def histogram_threshold(image):
    """
    Apply automatic binary thresholding using Otsu's method.

    Args:
        image: RGB uint8 NumPy image.

    Returns:
        Binary image as an RGB uint8 NumPy image.
    """
    check_img(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)