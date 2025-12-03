# model_loader.py (à mettre à la racine du projet)
import onnxruntime as ort

# Chemin du modèle dans le conteneur
MODEL_PATH = "/app/model/best.onnx"

session = ort.InferenceSession(MODEL_PATH)

print("Modèle ONNX chargé avec succès !")
print("Input name:", session.get_inputs()[0].name)
print("Classes:", ["Fake", "Real"])

input_name = session.get_inputs()[0].name

def get_session():
    return session, input_name