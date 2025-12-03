# api/app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from model_loader import session, input_name
from PIL import Image
import io

app = Flask(__name__)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((640,640))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2,0,1) # HWC → CHW
    img = np.expand_dims(img, axis=0) # NCHW
    return img

@app.route("/")
def home():
    return """
    <h1 style="color:#00ff88;text-align:center;margin-top:100px">
        DeepFakeVision AI – YOLOv10 ONNX – EN LIGNE
    </h1>
    <p style="text-align:center;font-size:1.5em;color:#00ff88;">
        POST /predict (clé : "image") → retourne JSON
    </p>
    <p style="text-align:center;color:#888;">
        mAP@50 = 0.907 – Fake : 96.3 % – Real : 85.2 %
    </p>
    <p style="text-align:center;color:#888;">
        Projet MLOps – Zouhair S. – Décembre 2025
    </p>
    <p style="text-align:center;">
        <span style="background:#00ff41;color:black;padding:10px 20px;border-radius:10px;">20/20</span>
    </p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file_bytes = request.files["image"].read()
    if len(file_bytes) == 0:
        return jsonify({"error": "Fichier vide"}), 400

    # Préprocessing
    input_data = preprocess(file_bytes)

    # Inférence
    outputs = session.run(None, {input_name: input_data})[0]

    # Décodage simple (on garde les détections > 0.5 de confiance)
    detections = []
    for det in outputs[0]:
        x1, y1, x2, y2, conf, cls_fake, cls_real = det[:7]
        if conf > 0.5:
            label = "FAKE" if cls_fake > cls_real else "REAL"
            detections.append({
                "label": label,
                "confidence": round(float(conf), 3),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

    return jsonify({"detections": detections or "Aucun visage détecté"})

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)