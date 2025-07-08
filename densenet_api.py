from flask import Flask, request, jsonify
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from werkzeug.utils import secure_filename

# ---------------- Load Model ---------------- #
model = load_model("models/densenet_video_classifier.h5")

# ---------------- Flask App ---------------- #
app = Flask(__name__)

# ---------------- Video to Tensor ---------------- #
def read_video_tensor(file_bytes, max_frames=16):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERROR] OpenCV could not open video file.")
            return None

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = preprocess_input(frame.astype(np.float32))
            frames.append(frame)

        cap.release()
        os.remove(video_path)

        if len(frames) == 0:
            return None

        video_tensor = np.stack(frames)  # Shape: (T, H, W, C)
        return video_tensor

    except Exception as e:
        print(f"[EXCEPTION] Failed to read video: {e}")
        return None

# ---------------- API Endpoint ---------------- #
@app.route('/predict', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    file_bytes = video_file.read()

    video_tensor = read_video_tensor(file_bytes, max_frames=16)

    if video_tensor is None:
        return jsonify({'error': 'Could not extract frames from video'}), 400

    # Prepare batch for model: shape (N, 224, 224, 3)
    batch = np.array(video_tensor)

    # Predict on all frames, then average
    preds = model.predict(batch, verbose=0)
    avg_pred = float(np.mean(preds))

    label = 'Abnormal' if avg_pred > 0.5 else 'Normal'
    confidence = round(avg_pred if label == 'Abnormal' else 1 - avg_pred, 4)

    response = {
        'filename': filename,
        'frames_used': len(video_tensor),
        'prediction': label,
        'probability': round(avg_pred, 4),
        'confidence': confidence,
        'model': 'DenseNet121',
        'status': 'success'
    }

    return jsonify(response)

# ---------------- Run Server ---------------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
