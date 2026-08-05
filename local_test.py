"""
Offline model inference tester for saved Udacity driving data.

Run this when the Udacity simulator is unavailable to verify that the model
loads and produces steering predictions on your collected images.

Example:
    python local_test.py --num 10
"""

import argparse
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from src.data_loader import load_log
from src.preprocessing import preprocess
from src import config


def parse_args():
    parser = argparse.ArgumentParser(description="Offline inference tester for the saved model.")
    parser.add_argument(
        "--model",
        default=config.MODEL_OUT,
        help="Path to the saved Keras model file.",
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="Root data directory containing driving_log.csv and IMG/.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of sample images to run through the model.",
    )
    return parser.parse_args()


def load_sample_paths(data_dir, max_samples):
    df = load_log(data_dir)
    img_dir = os.path.join(data_dir, "IMG")
    paths = []
    steerings = []

    for _, row in df.iterrows():
        filename = os.path.basename(str(row["center"]).strip())
        image_path = os.path.join(img_dir, filename)
        if not os.path.isfile(image_path):
            continue
        paths.append(image_path)
        steerings.append(float(row["steering"]))
        if len(paths) >= max_samples:
            break

    return paths, np.array(steerings, dtype=np.float32)


def load_preprocessed_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as img:
            image = np.asarray(img)
        images.append(preprocess(image))
    return np.stack(images, axis=0)


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = load_model(args.model, compile=False)
    paths, steerings = load_sample_paths(args.data_dir, args.num)

    if len(paths) == 0:
        raise RuntimeError(
            f"No sample images found in {args.data_dir}/IMG. "
            "Check that your driving_log.csv and IMG/ files exist."
        )

    images = load_preprocessed_images(paths)
    predictions = model.predict(images, verbose=0).flatten()

    print(f"Loaded {len(paths)} sample images from {args.data_dir}/IMG")
    print(f"Model: {args.model}\n")
    print("Sample predictions:")
    print("#  image file                       actual   predicted")
    print("-- -------------------------------- -----   ---------")
    for index, path in enumerate(paths):
        actual = steerings[index]
        pred = float(predictions[index])
        print(f"{index + 1:2d} {os.path.basename(path):30s} {actual:7.4f} {pred:10.4f}")

    mae = float(np.mean(np.abs(predictions - steerings)))
    print(f"\nMean absolute error on {len(paths)} samples: {mae:.4f}")


if __name__ == "__main__":
    main()
