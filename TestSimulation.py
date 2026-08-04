"""
Connects the trained model to the Udacity simulator in Autonomous Mode.
The simulator sends the current center-camera frame over a socket; we
preprocess it, predict a steering angle, and send back steering + throttle.

Run this, then launch the simulator and pick Autonomous Mode:
    python TestSimulation.py
"""

import base64
from io import BytesIO

import numpy as np
import socketio
import eventlet
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

from src.preprocessing import preprocess
from src import config

sio = socketio.Server()
app = Flask(__name__)
model = None

MAX_SPEED = 15


def send_control(steering, throttle):
    sio.emit("steer", data={
        "steering_angle": str(steering),
        "throttle": str(throttle),
    })


@sio.on("connect")
def connect(sid, environ):
    print("Connected to simulator")
    send_control(0, 0)


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        return
    speed = float(data["speed"])
    img = Image.open(BytesIO(base64.b64decode(data["image"])))
    img = np.asarray(img)
    img = preprocess(img)
    img = np.array([img])

    steering = float(model.predict(img, verbose=0))
    # simple throttle: slow down as we approach max speed
    throttle = 1.0 - (speed / MAX_SPEED)
    send_control(steering, throttle)


if __name__ == "__main__":
    model = load_model(config.MODEL_OUT)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
