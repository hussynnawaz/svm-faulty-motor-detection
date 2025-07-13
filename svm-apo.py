import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from sklearn.svm import SVC
import joblib

app = Flask(__name__)
model = joblib.load("models/svm_model.joblib")

# ========== CONFIG ==========
FRAME_SIZE = (64, 64)
NUM_FRAMES = 5
ALPHA = 30

# ========== FEATURE EXTRACTOR ==========
def extract_feature(path, num_frames=NUM_FRAMES, alpha=ALPHA):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frames.append(frame)
    cap.release()
    if len(frames) < 2: return None

    frames = np.array(frames)
    fft = np.fft.fft(frames, axis=0)
    fft[3:-3] = 0
    filtered = np.fft.ifft(fft, axis=0).real
    amplified = frames + alpha * filtered
    return amplified.mean(axis=0).flatten()

# ========== ROUTES ==========
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("temp.mp4")
    file.save(filepath)

    vec = extract_feature(filepath)
    os.remove(filepath)

    if vec is None or vec.shape[0] != model.n_features_in_:
        return jsonify({"error": f"Invalid feature size: {None if vec is None else vec.shape[0]}"})

    pred = model.predict([vec])[0]
    prob = model.predict_proba([vec])[0][int(pred)]

    return jsonify({
        "prediction": "Abnormal" if pred == 1 else "Normal",
        "confidence": round(float(prob), 4)
    })

# ========== RUN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
