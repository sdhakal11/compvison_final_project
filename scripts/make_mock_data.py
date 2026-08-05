"""
Generates a small synthetic dataset (fake center/left/right jpgs + driving_log.csv)
so the pipeline can be exercised end-to-end without the real simulator.
Not part of the deliverable - just a local testing helper.
"""

import os
import numpy as np
import cv2

OUT_DIR = "data"
IMG_DIR = os.path.join(OUT_DIR, "IMG")
NUM_SAMPLES = 300


def make_fake_frame(steering):
    img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
    # draw a line whose position reflects the steering angle, so a model has
    # something non-random to learn during the smoke test
    x = int(160 + steering * 120)
    cv2.line(img, (x, 0), (x, 159), (255, 255, 255), 8)
    return img


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []

    for i in range(NUM_SAMPLES):
        steering = float(np.clip(rng.normal(0, 0.3), -1, 1))
        center = make_fake_frame(steering)
        left = make_fake_frame(min(steering + 0.1, 1.0))
        right = make_fake_frame(max(steering - 0.1, -1.0))

        center_name = f"center_{i:04d}.jpg"
        left_name = f"left_{i:04d}.jpg"
        right_name = f"right_{i:04d}.jpg"

        cv2.imwrite(os.path.join(IMG_DIR, center_name), cv2.cvtColor(center, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(IMG_DIR, left_name), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(IMG_DIR, right_name), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

        throttle = 0.5
        brake = 0.0
        speed = 10.0
        rows.append(
            f"IMG/{center_name},IMG/{left_name},IMG/{right_name},"
            f"{steering},{throttle},{brake},{speed}"
        )

    log_path = os.path.join(OUT_DIR, "driving_log.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {NUM_SAMPLES} samples to {IMG_DIR} and {log_path}")


if __name__ == "__main__":
    main()
