import os, cv2, json, uuid, numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model("models/efficientnet_model.h5")

os.makedirs("results", exist_ok=True)

def extract_frames(file_bytes, num_frames=3):
    path = "temp_video.mp4"
    with open(path, "wb") as f: f.write(file_bytes)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); os.remove(path); return None
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(img_to_array(frame))
        frames.append(frame)
    cap.release(); os.remove(path)
    if not frames: return None
    return np.array(frames)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        data = extract_frames(file.read())
        if data is None: return jsonify({"error": "Frame extraction failed"}), 500
        preds = model.predict(data, verbose=0)
        avg_pred = np.mean(preds)
        label = "Abnormal" if avg_pred > 0.5 else "Normal"
        confidence = round(float(avg_pred if label == "Abnormal" else 1 - avg_pred), 4)
        result = {
            "id": str(uuid.uuid4()),
            "prediction": int(avg_pred > 0.5),
            "label": label,
            "confidence": confidence,
            "filename": file.filename
        }

        # Save result
        with open(f"results/{result['id']}.json", "w") as f:
            json.dump(result, f, indent=2)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
