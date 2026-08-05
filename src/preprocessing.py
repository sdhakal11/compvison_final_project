"""
Preprocessing steps for the camera images.
Order: crop -> gaussian blur -> RGB to YUV -> resize -> normalize.
The 200x66 YUV input is what the Nvidia model expects.
"""

import cv2
import numpy as np
from src import config


def crop(img):
    # keep only the road part of the frame (see Figure 6 in the handout)
    return img[60:135, :, :]


def rgb_to_yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def resize(img):
    return cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))


def preprocess(img):
    img = crop(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = rgb_to_yuv(img)
    img = resize(img)
    img = img / 255.0          # normalize to 0-1
    return img
