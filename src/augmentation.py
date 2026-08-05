"""
Data augmentation so the model sees more varied images and generalizes better.
Applied randomly to only part of the data, and ONLY to the training set.
Flipping also flips the steering angle (multiply by -1).
"""

import cv2
import numpy as np
import random


def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    ratio = random.uniform(0.5, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_zoom(img):
    h, w = img.shape[:2]
    scale = random.uniform(1.0, 1.3)
    nh, nw = int(h * scale), int(w * scale)
    zoomed = cv2.resize(img, (nw, nh))
    # crop back to the original size from the top-left
    return zoomed[0:h, 0:w]


def random_pan(img):
    h, w = img.shape[:2]
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def random_flip(img, steering):
    img = cv2.flip(img, 1)
    steering = -steering          # flipped image = opposite steering
    return img, steering


def random_augment(img, steering):
    # each augmentation has a 50% chance of being applied
    if random.random() < 0.5:
        img = random_brightness(img)
    if random.random() < 0.5:
        img = random_zoom(img)
    if random.random() < 0.5:
        img = random_pan(img)
    if random.random() < 0.5:
        img, steering = random_flip(img, steering)
    return img, steering
