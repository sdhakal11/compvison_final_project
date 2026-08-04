"""
Loads driving_log.csv, balances the steering-angle distribution, and splits
into training and validation sets. We only use the center camera + steering.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src import config


COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def load_log(data_dir=config.DATA_DIR):
    path = os.path.join(data_dir, config.LOG_FILE)
    df = pd.read_csv(path, names=COLUMNS)
    # keep just the filename of the center image, not the full path
    df["center"] = df["center"].apply(lambda p: os.path.basename(str(p).strip()))
    return df


def show_histogram(steerings, title="Steering distribution"):
    plt.hist(steerings, bins=config.NUM_BINS)
    plt.title(title)
    plt.xlabel("steering angle")
    plt.ylabel("count")
    plt.show()


def balance_data(df):
    # the car drives straight most of the time, so the 0-steering bin is huge.
    # trim each over-full bin down to SAMPLES_PER_BIN so it's not biased.
    bins = np.linspace(-1, 1, config.NUM_BINS + 1)
    keep = []
    for i in range(config.NUM_BINS):
        in_bin = df[(df["steering"] >= bins[i]) & (df["steering"] < bins[i + 1])]
        if len(in_bin) > config.SAMPLES_PER_BIN:
            in_bin = in_bin.sample(config.SAMPLES_PER_BIN)
        keep.append(in_bin)
    return pd.concat(keep).reset_index(drop=True)


def load_data(data_dir=config.DATA_DIR):
    df = load_log(data_dir)
    df = balance_data(df)
    img_dir = os.path.join(data_dir, "IMG")
    image_paths = df["center"].apply(lambda f: os.path.join(img_dir, f)).values
    steerings = df["steering"].values.astype(np.float32)
    return train_test_split(
        image_paths, steerings, test_size=config.TEST_SIZE, random_state=42
    )
