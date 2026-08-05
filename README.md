# Self-Driving Car Simulation using CNN
DPS920 Final Project

A convolutional neural network (Nvidia architecture) that predicts steering angles from front-camera images to drive a car autonomously in the Udacity self-driving car simulator.

## Approach
1. Data collection: drive manually in simulator Training Mode to generate image frames and steering labels.
2. Data balancing: reduce the over-representation of straight-driving samples so the model doesn't become biased toward small steering angles.
3. Augmentation: apply random flips, brightness changes, zoom, and pan to improve robustness.
4. Preprocessing: crop the road area, apply Gaussian blur, convert from RGB to YUV, resize to 200x66, and normalize the img before training.
5. Training: train the CNN using mean squared error loss and the Adam optimizer.
6. Testing: run the saved model in the simulator through the provided testing script.
   
## Project structure
```
compvison_final_project/
|-- README.md
|-- .gitignore
|-- package_list.txt          # provided by the course - environment setup
|-- local_test.py             # test the model locally
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

## Environment setup and running the model
1. Create and activate a Python environment
2. Install dependencies after setting up environment:
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r package_list.txt
3. Prepare the dataset by placing the collected simulator data in the data folder and the imgs in the data/IMG folder
4. Train the model by running `python -m src.train`
5. Test in the simulator by running `python TestSimulation.py`
6. ???
7. Profit

## Challenges and solutions
- The simulator wasn't working for some of us so we added an offline inference script to stress test the saved model to show predicted path vs actual path.
- Straight-driving samples lead to imbalanced steering data so we addressed this by balancing the training data so that the model would learn a more useful steering distribution. 

## Team
- Daniel Fu
- Shaswot Dhakal
- Nishnath Bandari
