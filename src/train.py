"""
Main training script. Loads and balances the data, builds the Nvidia model,
trains it with the batch generator, plots the training/validation loss, and
saves the final model.

Run from the project root:  python -m src.train
"""

import matplotlib.pyplot as plt
from src import config
from src.data_loader import load_data
from src.batch_generator import batch_generator
from src.model import build_model


def plot_history(history):
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title("Training / Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.legend()
    plt.show()


def main():
    x_train, x_val, y_train, y_val = load_data()
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

    model = build_model()
    model.summary()

    history = model.fit(
        batch_generator(x_train, y_train, config.BATCH_SIZE, is_training=True),
        steps_per_epoch=len(x_train) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=batch_generator(x_val, y_val, config.BATCH_SIZE, is_training=False),
        validation_steps=len(x_val) // config.BATCH_SIZE,
        verbose=1,
    )

    model.save(config.MODEL_OUT)
    print(f"Model saved to {config.MODEL_OUT}")
    plot_history(history)


if __name__ == "__main__":
    main()
