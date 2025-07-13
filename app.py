# =================== IMPORTS ===================
import os
import cv2
import json
import uuid
import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# =================== APP INITIALIZATION ===================
app = Flask(__name__)

# =================== LOAD MODELS ===================
svm_model = joblib.load("models/resnet50model.pkl")
resnet = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(224, 224, 3))

# =================== FEATURE EXTRACTOR ===================
def extract_resnet_features_from_bytes(file_bytes, num_frames=5):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(file_bytes)

    cap = cv2.VideoCapture(temp_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        os.remove(temp_video_path)
        return None

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frame_features = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = img_to_array(frame)
        frame = preprocess_input(frame)
        frame_features.append(frame)

    cap.release()
    os.remove(temp_video_path)

    if not frame_features:
        return None

    frames_array = np.array(frame_features)
    features = resnet.predict(frames_array, verbose=0)
    return np.mean(features, axis=0).reshape(1, -1)

# =================== SAVE RESULT ===================
def save_result_json(data, folder="results"):
    os.makedirs(folder, exist_ok=True)
    filename = f"result_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# =================== PREDICTION ENDPOINT ===================
@app.route("/predict", methods=["POST"])
def predict_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_bytes = file.read()
        feature = extract_resnet_features_from_bytes(file_bytes)

        if feature is None:
            return jsonify({"error": "Could not extract features from the video."}), 500

        pred = svm_model.predict(feature)[0]
        label = "Normal" if pred == 0 else "Abnormal"

        try:
            proba = svm_model.predict_proba(feature)[0][int(pred)]
        except:
            decision = svm_model.decision_function(feature)
            proba = 1 / (1 + np.exp(-decision))[0]

        confidence = round(float(proba), 4)

        result = {
            "prediction": int(pred),
            "label": label,
            "probability": confidence,
            "confidence": confidence,
            "filename": file.filename
        }

        save_result_json(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =================== RUN SERVER ===================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
