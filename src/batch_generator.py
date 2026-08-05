"""
Batch generator used during training. It yields batches of preprocessed images
and their steering angles. Training batches get random augmentation; validation
batches don't (we want to evaluate on clean images).
"""

import cv2
import numpy as np
from src.preprocessing import preprocess
from src.augmentation import random_augment


def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def batch_generator(image_paths, steerings, batch_size, is_training):
    while True:
        batch_imgs = []
        batch_steers = []
        for _ in range(batch_size):
            i = np.random.randint(len(image_paths))
            img = load_image(image_paths[i])
            steer = steerings[i]

            if is_training:
                img, steer = random_augment(img, steer)

            img = preprocess(img)
            batch_imgs.append(img)
            batch_steers.append(steer)

        yield np.asarray(batch_imgs), np.asarray(batch_steers)
