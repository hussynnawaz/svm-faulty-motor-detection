import os, cv2, numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model("models/resnet_model.h5")

def extract_frames(video_bytes, num_frames=5):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f: f.write(video_bytes)
    cap = cv2.VideoCapture(temp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); os.remove(temp_path); return None
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(img_to_array(frame).astype(np.float32))
        frames.append(frame)
    cap.release(); os.remove(temp_path)
    return np.array(frames) if frames else None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        frames = extract_frames(file.read(), num_frames=5)
        if frames is None: return jsonify({"error": "Frame extraction failed"}), 500
        preds = model.predict(frames, verbose=0)
        avg_pred = float(np.mean(preds))
        label = "Abnormal" if avg_pred > 0.5 else "Normal"
        confidence = round(avg_pred if label == "Abnormal" else 1 - avg_pred, 4)
        return jsonify({
            "prediction": int(avg_pred > 0.5),
            "label": label,
            "confidence": confidence,
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
