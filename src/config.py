"""
Central config for the self-driving car project.
Paths and hyperparameters live here so the other modules stay clean.
"""

# where the simulator saved the data (change to your own path)
DATA_DIR = "data"
LOG_FILE = "driving_log.csv"

# Nvidia model input size (width x height)
IMG_WIDTH = 200
IMG_HEIGHT = 66

# dataset balancing
NUM_BINS = 25
SAMPLES_PER_BIN = 400        # cap so the straight-driving (0 steering) bin
                            # doesn't dominate the histogram

# training
TEST_SIZE = 0.2
BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_OUT = "models/model.h5"
