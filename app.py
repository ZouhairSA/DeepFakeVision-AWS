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
    return "<h1>DeepFakeVision AI – YOLOv10 ONNX – EN LIGNE</h1><p>POST /predict + image</p>"

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