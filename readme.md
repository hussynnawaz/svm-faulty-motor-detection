# Faulty Motor Detection via Video Classification

This project provides a Flask-based API for detecting abnormal (faulty) and normal motor behavior from video files using deep learning and machine learning models. It supports two main models: ResNet50 (with SVM) and DenseNet121.

## Features

- **REST API** for video classification.
- Supports `.mp4` and `.mov` video files.
- Uses pre-trained ResNet50 for feature extraction and SVM for classification.
- DenseNet121 model for direct video classification.
- Returns prediction label, probability, and confidence.

## Project Structure

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txtpython densenet_api.py
   import requests

url = "http://localhost:5000/predict"
with open("media/AbnormalTest.mp4", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
print(response.json())import requests

url = "http://localhost:5000/predict"
with open("media/AbnormalTest.mp4", "rb") as f:
    files = {"video": f}
    response = requests.post(url, files=files)
print(response.json())
