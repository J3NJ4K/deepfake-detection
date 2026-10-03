import os
import cv2
import numpy as np

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

