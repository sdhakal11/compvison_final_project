# Self-Driving Car Simulation using CNN

DPS920 Final Project. A convolutional neural network (Nvidia architecture) that
predicts steering angles from front-camera images to drive a car autonomously
in the Udacity self-driving car simulator.

## Pipeline
1. **Data collection** - drive manually in the simulator's Training Mode to
   generate `IMG/` frames and `driving_log.csv`.
2. **Balancing** - trim the over-represented straight-driving samples so the
   steering histogram is balanced.
3. **Augmentation** - random flip (with steering negated), brightness, zoom,
   pan. Training set only.
4. **Preprocessing** - crop road area -> Gaussian blur -> RGB to YUV ->
   resize to 200x66 -> normalize.
5. **Training** - Nvidia CNN, MSE loss, Adam optimizer.
6. **Testing** - `TestSimulation.py` drives the car in Autonomous Mode.

## Project structure
```
compvison_final_project/
|-- README.md
|-- .gitignore
|-- package_list.txt          # provided by the course - environment setup
|-- data/                     # (gitignored) IMG/ + driving_log.csv
|-- models/                   # (gitignored) saved model.h5
|-- src/
|   |-- config.py             # paths + hyperparameters
|   |-- data_loader.py        # load csv, balance, train/val split
|   |-- augmentation.py       # flip / brightness / zoom / pan
|   |-- preprocessing.py      # crop / YUV / resize / normalize
|   |-- batch_generator.py    # yields training/validation batches
|   |-- model.py              # Nvidia CNN (Figure 7)
|   |-- train.py              # training entry point
|-- TestSimulation.py         # runs the model in the simulator
```

## How to run
1. Set up the environment from `package_list.txt` (TensorFlow GPU on Windows 10).
2. Put your collected data in `data/` (so you have `data/IMG/` and
   `data/driving_log.csv`).
3. Train:  `python -m src.train`
4. Test:   `python TestSimulation.py`, then launch the simulator in
   Autonomous Mode.

## Team
- (name) - lead / data pipeline
- (name) - model + training
- (name) - testing + docs
