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
|-- scripts/
|   |-- make_mock_data.py     # generates synthetic data for offline testing
|-- TestSimulation.py         # runs the model in the simulator
```

## How to run
1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   (`package_list.txt` is the course-provided conda/Python 3.8 alternative if
   you specifically need that environment instead.)
2. Get the Udacity Term 1 simulator (search "udacity self driving car
   simulator" if you don't already have it) and drive in Training Mode to
   collect data, or use `scripts/make_mock_data.py` to generate a small
   synthetic dataset for testing the pipeline without the simulator.
3. Put your collected data in `data/` (so you have `data/IMG/` and
   `data/driving_log.csv`).
4. Train:  `.venv\Scripts\python.exe -m src.train`
5. Test:   `.venv\Scripts\python.exe TestSimulation.py`, then launch the
   simulator and pick Autonomous Mode.

### Note on TestSimulation.py dependency versions
The simulator is an older Unity app that speaks an older Socket.IO wire
protocol. `requirements.txt` pins `python-socketio`, `python-engineio`,
`flask`, and Flask's dependency chain (Jinja2/Werkzeug/click/etc.) to older
versions to match it - with current versions of those packages, Autonomous
Mode connects but the car never receives steering commands (looks like
nothing is happening). `eventlet` is pinned separately, just high enough to
actually import on Python 3.12 (older eventlet releases use APIs Python
removed).

### Offline model sanity check
If the Udacity simulator is not working, verify the saved model locally:
    .venv\Scripts\python.exe local_test.py --num 10

This will load `models/model.h5`, preprocess sample images from `data/IMG/`,
run predictions, and print actual vs predicted steering values.

## Team
- Daniel
- Shaswot
- Nishnath
