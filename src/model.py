"""
The Nvidia CNN architecture from Figure 7 of the handout.
Input is a 66x200 YUV image, output is a single steering angle (regression),
so the loss is mean squared error.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from src import config


def build_model():
    model = Sequential()

    # 5 convolutional layers (elu activation)
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu",
                     input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))

    model.add(Flatten())

    # fully connected layers (1164 -> 100 -> 50 -> 10 -> 1) to match Figure 7
    model.add(Dense(1164, activation="elu"))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))          # single steering value

    model.compile(loss="mse", optimizer=Adam(learning_rate=config.LEARNING_RATE))
    return model


if __name__ == "__main__":
    build_model().summary()
