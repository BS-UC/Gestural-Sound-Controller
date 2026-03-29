# Gestural-Sound-Controller
A basic Python program that allows you to control the volume (with your right hand) and the pitch (with your left hand) of a built-in sine wave sound, need to access your camera.

## Features

- **Pitch Control**: Raise and lower your left hand to change the pitch.
- **Volume Control**: Raise and lower your right hand to control the volume.
- **Auto-Calibration**: The program automatically calibrates to your initial hand positions after 3 seconds of stability.
- **Smoothing**: Provides smooth pitch transitions for subtle movements, but not really responsive to fast movements.

## Requirements

- Python 3
- `opencv-python`
- `mediapipe`
- `pyaudio`
- `numpy`

## Installation
Download the files except the README, put them in a folder
1.  **Navigate (cd) to this folder in your terminal:**

2.  **Create a virtual environment (if you haven't already):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    On macOS, you may need to install `portaudio` first if you don't have it.
    ```bash
    brew install portaudio
    ```
    Then install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    The `hand_landmarker.task` model file from MediaPipe is already included in the directory.

## Usage

Run the `theremin.py` script from your activated virtual environment in your terminal:

```bash
python3 theremin.py
```

- Hold your hands steady in front of the camera for about 3 seconds to calibrate. The on-screen text will notify you when calibration is complete.
- Move your left hand up and down to control the pitch.
- Move your right hand up and down to control the volume.
- Press 'q' to quit the program.
