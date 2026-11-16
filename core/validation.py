import numpy as np

def check_img(image) -> None:
    # check that image is a NumPy array -> TypeError
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a numpy array, it got {type(image).__name__}")

    # check that image has 3 color channels -> ValueError
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have the shape of (H, W, 3), it got {image.shape}")

    # check that the data type is uint8 -> ValueError
    if image.dtype != np.uint8:
        raise ValueError(f"Image must have the dtype uint8, it got {image.dtype}")

    # check that the image is not empty -> ValueError
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image must not be empty, it got {image.shape}")
