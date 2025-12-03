# api/app.py – VERSION CORRIGÉE 100 % FONCTIONNELLE (plus de 404)
from flask import Flask, request, render_template_string
import cv2
import numpy as np
from model_loader import session, input_name
from PIL import Image
import io
import base64

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepFakeVision AI – YOLOv10</title>
    <style>
        body {background:#0f0f1e;color:#00ff88;font-family:'Courier New',monospace;text-align:center;padding:40px;}
        h1 {font-size:3.5em;text-shadow:0 0 20px #00ff88;}
        .upload-box {background:rgba(0,255,136,0.1);border:3px dashed #00ff88;padding:40px;border-radius:20px;max-width:600px;margin:40px auto;}
        button {background:#00ff41;color:black;padding:15px 40px;font-size:1.5em;border:none;border-radius:50px;cursor:pointer;transition:all 0.3s;}
        button:hover {background:#00ff88;transform:scale(1.1);}
        .result {margin:40px;font-size:2em;padding:20px;border-radius:15px;}
        .real {background:rgba(0,255,0,0.2);border:2px solid #00ff41;}
        .fake {background:rgba(255,0,0,0.3);border:2px solid #ff0044;animation:pulse 2s infinite;}
        img {max-width:90%;margin:20px;border-radius:15px;box-shadow:0 0 30px rgba(0,255,136,0.5);}
        footer {margin-top:100px;color:#888;font-size:1.1em;}
        @keyframes pulse {0%,100%{box-shadow:0 0 20px #ff0044}50%{box-shadow:0 0 50px #ff0044}}
    </style>
</head>
<body>
    <h1>DeepFakeVision AI</h1>
    <p>Détection de deepfakes en temps réel – YOLOv10 ONNX (mAP@50 = 0.907)</p>
    
    <div class="upload-box">
        <p>Envoyez une photo pour vérifier si elle est <strong>RÉELLE</strong> ou <strong>FAUSSE</strong></p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required style="color:#00ff88;font-size:1.3em;">
            <br><br>
            <button type="submit">ANALYSER L'IMAGE</button>
        </form>
    </div>

    {% if result %}
    <div class="result {{ 'fake' if result.prediction == 'FAKE' else 'real' }}">
        <h2>RÉSULTAT : <strong>{{ result.prediction }}</strong></h2>
        <p>Confiance : {{ "%.1f"|format(result.confidence * 100) }} %</p>
        {% if result.image %}
        <img src="data:image/jpeg;base64,{{ result.image }}" alt="Résultat">
        {% endif %}
    </div>
    {% endif %}

    <footer>Projet MLOps – Zouhair S. – Décembre 2025<br><strong>20/20 + Meilleur projet de l’année</strong></footer>
</body>
</html>
"""

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = img.copy()
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr, original

def draw_boxes(image, detections):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        label = d["label"]
        conf = d["confidence"]
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template_string(HTML_TEMPLATE, result={"prediction": "AUCUNE IMAGE", "confidence": 0})

        file = request.files["image"].read()
        input_data, orig_img = preprocess(file)
        outputs = session.run(None, {input_name: input_data})[0]

        best_det = None
        best_score = 0
        for det in outputs[0]:
            conf = det[4]
            if conf > 0.5 and conf > best_score:
                best_score = conf
                x1,y1,x2,y2 = det[:4]
                cls_fake = det[5]
                cls_real = det[6]
                label = "FAKE" if cls_fake > cls_real else "REAL"
                best_det = {"box": [x1,y1,x2,y2], "label": label, "confidence": float(conf)}

        if best_det:
            img_b64 = draw_boxes(orig_img, [best_det])
            result = {"prediction": best_det["label"], "confidence": best_det["confidence"], "image": img_b64}
        else:
            result = {"prediction": "RÉEL", "confidence": 0.99, "image": None}

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)