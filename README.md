# Facial Emotion Recognition (ResNet-18)

This project performs **facial emotion recognition** on images / webcam frames using a **ResNet-18** classifier trained on **7 emotion classes**.

## What’s in this repo

- `models/fer_resnet18.pth`: trained ResNet-18 weights (required to run inference)
- `data/train/`, `data/test/`: image folders in `torchvision.datasets.ImageFolder` format
- `predict.py`: real-time webcam demo (face detection + emotion label)
- `evaluate.py`: classification report + confusion matrix on `data/test`
- `accuracy.py`: prints train/test accuracy
- `grad.py`: saves Grad-CAM visualizations for a few samples per class

## Dataset layout

The code expects this directory structure:

```
data/
  train/
    angry/ ...
    disgust/ ...
    fear/ ...
    happy/ ...
    neutral/ ...
    sad/ ...
    surprise/ ...
  test/
    angry/ ...
    disgust/ ...
    fear/ ...
    happy/ ...
    neutral/ ...
    sad/ ...
    surprise/ ...
  models/
    haarcascade_frontalface_default.xml
```

Class labels are taken from the folder names (alphabetical order).

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:

- If `torch` / `torchvision` installation fails on your machine, install them following the official PyTorch instructions for your platform, then install the remaining packages.
- `predict.py` requires `opencv-python` (included in `requirements.txt`).

## Run

### Evaluate on the test set (report + confusion matrix)

```bash
python evaluate.py
```

### Print train/test accuracy

```bash
python accuracy.py
```

### Webcam demo (press `q` to quit)

```bash
python predict.py
```

This uses OpenCV’s Haar cascade at `data/models/haarcascade_frontalface_default.xml`.

On macOS, you may see a warning about Continuity Cameras / `AVCaptureDeviceTypeExternal` being deprecated. This is emitted by the OS camera stack (not your model) and can usually be ignored.

### Generate Grad-CAM visualizations

```bash
python grad.py
```

Outputs are written to `gradcam_results/`.

## Hardware / device

This repo is optimized for **MacBooks (Apple Silicon)** using **PyTorch MPS** acceleration when available. If MPS isn’t available, the scripts automatically fall back to **CPU**.
