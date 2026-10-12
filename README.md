# Deepfake Detector Photo Editor

Machine Vision project for IKT213.

A desktop photo-editing application that combines standard image processing tools with face detection and pretrained deepfake classification

===

## Installation and Setup

## Requirements


Before starting, make sure to have the following installed:

- Python 3
- Git

===

### 1. Clone the repository
```bash
git clone https://github.com/J3NJ4K/deepfake-detection.git
cd deepfake-detection
git checkout first-release
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
- macOS/linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install project dependencies

In active virtual environment run:
```bash
python -m pip install -r requirements.txt
```
The required libraries installed include PySide6, OpenCV, NumPy, and PyTorch.

### 4. Download pretrained model files

Large binary weight files are excluded from git via .gitignore to prevent repository bloat

First, Make sure models/ directory exists.

#### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force models
```

#### macOS / Linux

```bash
mkdir -p models
```

The following two files must be placed inside 'models/':

- 'face_detection_yunet_20203mar.ohnx'
- 'xception_best.pth'

#### YuNet face detector

On Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models/face_detection_yunet_2023mar.onnx"
```

On MacOS / Linux:
```bash
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

#### Xception deepfake classifier

Download 'xception_best.pth' from DeepfakeBench v1.0.1 release:

https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1 

placed into `models/`.

```text
models/xception_best.pth
```

## Running the Application

Make sure the virtual environment is activated and the required model files are inside `models/`.

Run the app from the repository root:

```bash
python main.py
```

The application window should open.

## Basic Usage

1. Open an image using `File -> Open`.
2. Use the available image-editing tools.
3. Apply filters through the Filters menu.
4. Run the face detection or deepfake analysis features on the opened image.

===

## First Release Features

The first prototype release includes:

- Desktop photo editor interface
- Image loading and display
- Basic image transformations
- Gaussian Blur
- Sobel Edge Detection
- Binary Thresholding
- Histogram / Otsu Thresholding
- YuNet face detection
- Pretrained Xception deepfake classification
- Bounding boxes on faces and classification probabilities