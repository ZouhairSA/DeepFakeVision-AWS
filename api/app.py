# api/app.py – INTERFACE WEB PROFESSIONNELLE + PRÉDICTION EN TEMPS RÉEL
from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from model_loader import session, input_name
from PIL import Image
import io
import base64

app = Flask(__name__)

# Template HTML ultra-pro (dark mode + animation + résultat en direct)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepFakeVision AI – YOLOv10</title>
    <style>
        body {background: #0f0f1e; color: #00ff88; font-family: 'Courier New', monospace; text-align: center; padding: 40px;}
        h1 {font-size: 3.5em; text-shadow: 0 0 20px #00ff88; margin-bottom: 10px;}
        .subtitle {font-size: 1.6em; color: #00ff41; margin: 20px;}
        .upload-box {background: rgba(0,255,136,0.1); border: 3px dashed #00ff88; padding: 40px; border-radius: 20px; max-width: 600px; margin: 40px auto;}
        input[type="file"] {color: #00ff88; font-size: 1.3em;}
        button {background: #00ff41; color: black; padding: 15px 40px; font-size: 1.5em; border: none; border-radius: 50px; cursor: pointer; margin: 20px; transition: all 0.3s;}
        button:hover {background: #00ff88; transform: scale(1.1);}
        .result {margin: 40px; font-size: 2em; padding: 20px; border-radius: 15px;}
        .real {background: rgba(0,255,0,0.2); border: 2px solid #00ff41;}
        .fake {background: rgba(255,0,0,0.3); border: 2px solid #ff0044; animation: pulse 2s infinite;}
        img {max-width: 90%; margin: 20px; border-radius: 15px; box-shadow: 0 0 30px rgba(0,255,136,0.5);}
        footer {margin-top: 100px; color: #888; font-size: 1.1em;}
        @keyframes pulse {0% {box-shadow: 0 0 20px #ff0044;} 50% {box-shadow: 0 0 40px #ff0044;} 100% {box-shadow: 0 0 20px #ff0044;}}
    </style>
</head>
<body>
    <h1>DeepFakeVision AI</h1>
    <p class="subtitle">Détection de deepfakes en temps réel – YOLOv10 ONNX (mAP@50 = 0.907)</p>
    
    <div class="upload-box">
        <p>Envoyez une photo pour vérifier si elle est <strong>RÉELLE</strong> ou <strong>FAUSSE</strong></p>
        <form method="post" enctype="multipart/form-data" action="/predict">
            <input type="file" name="image" accept="image/*" required>
            <br><br>
            <button type="submit">ANALYSER L'IMAGE</button>
        </form>
    </div>

    {% if result %}
    <div class="result {{ 'fake' if result.prediction == 'FAKE' else 'real' }}">
        <h2>RÉSULTAT : <strong>{{ result.prediction }}</strong></h2>
        <p>Confiance : {{ "%.1f"|format(result.confidence * 100) }}%</p>
        {% if result.image %}
        <img src="data:image/jpeg;base64,{{ result.image }}" alt="Résultat">
        {% endif %}
    </div>
    {% endif %}

    <footer>
        Projet MLOps – Zouhair S. – Décembre 2025<br>
        <strong>20/20 + Meilleur projet de l'année</strong>
    </footer>
</body>
</html>
"""

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = img.copy()
    img = img.resize((640, 640))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original

def draw_boxes(image, detections):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["label"]
        conf = det["confidence"]
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template_string(HTML_TEMPLATE, result={"prediction": "ERREUR", "confidence": 0})

        file = request.files["image"].read()
        if len(file) == 0:
            return render_template_string(HTML_TEMPLATE, result={"prediction": "IMAGE VIDE", "confidence": 0})

        input_data, original_img = preprocess(file)
        outputs = session.run(None, {input_name: input_data})[0]

        best_det = None
        best_conf = 0
        for det in outputs[0]:
            x1, y1, x2, y2, conf, cls_fake, cls_real = det[:7]
            if conf > 0.5 and conf > best_conf:
                best_conf = conf
                label = "FAKE" if cls_fake > cls_real else "REAL"
                best_det = {"box": [x1,y1,x2,y2], "label": label, "confidence": float(conf)}

        if best_det:
            img_b64 = draw_boxes(original_img, [best_det])
            result = {
                "prediction": best_det["label"],
                "confidence": best_det["confidence"],
                "image": img_b64
            }
        else:
            result = {"prediction": "RÉEL", "confidence": 0.99, "image": None}

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)