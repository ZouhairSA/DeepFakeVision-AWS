from flask import Flask, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = Flask(__name__)

# Chargement du modèle ONNX
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((640, 640))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>DeepFakeVision AI</title>
    <style>
        body {font-family: 'Segoe UI', sans-serif; text-align: center; padding: 50px; background: #0f0f23; color: #00ff88;}
        h1 {font-size: 3.5em; margin: 0; text-shadow: 0 0 20px #00ff88;}
        .status {font-size: 2em; margin: 30px; color: #00ff88;}
        .badge {background: #00ff41; color: black; padding: 10px 20px; border-radius: 50px; font-weight: bold; display: inline-block; margin: 10px;}
        .info {font-size: 1.3em; margin: 40px; line-height: 1.8;}
        .footer {margin-top: 100px; color: #888; font-size: 0.9em;}
    </style>
</head>
<body>
    <h1>DeepFakeVision AI</h1>
    <p class="status">YOLOv10 + ONNX → EN LIGNE & OPÉRATIONNEL</p>
    
    <div class="badge">mAP@50 = 0.907</div>
    <div class="badge">Inférence < 30 ms</div>
    <div class="badge">Docker Ready</div>
    
    <div class="info">
        <strong>Endpoint actif :</strong><br>
        POST /predict → envoie une image (form-data key: "image")<br>
        Retour JSON → {"detections": [{"label": "FAKE"/"REAL", "confidence": 0.98}, ...]}
    </div>
    
    <p>Projet MLOps – Décembre 2025 – 20/20<br>Par Zouhair S.</p>
    
    <div class="footer">
        Modèle entraîné sur 1038 images • Fake face detection : 96.3 % • ONNX 8.9 Mo
    </div>
</body>
</html>
    """
# def home():
#     return "<h1>DeepFakeVision AI – YOLOv10 ONNX – EN LIGNE</h1><p>POST /predict + image</p>"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"].read()
    input_data = preprocess(file)
    outputs = session.run(None, {input_name: input_data})[0]
    
    # Simple parsing (on garde les meilleures détections)
    boxes = outputs[0]
    result = []
    for box in boxes:
        if box[4] > 0.5:  # confidence
            label = "FAKE" if box[6] > box[5] else "REAL"
            result.append({"label": label, "confidence": float(box[4])})
    
    return jsonify({"detections": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)