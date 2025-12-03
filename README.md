# DeepFakeVision-AWS
Détection de deepfakes en temps réel avec YOLOv10 (mAP@50 = 0.907) – Modèle ONNX + API Flask + Docker + AWS Ready – Pipeline MLOps complet – 20/20

**Détection en temps réel de deepfakes – YOLOv10 (SOTA 2025)**  
mAP@50 = **0.907** | Inférence ONNX < 30 ms | API Flask + Docker + AWS Ready  


## Résultats
- mAP@50 : **0.907**  
- Fake face : **96.3 %**  
- Real face : 85.2 %  
- Modèle ONNX : 8.9 Mo  

## Lancer en 1 commande
```bash
docker run -d -p 5000:5000 zouhairsa/deepfakevision-yolov10
curl http://localhost:5000
# → DeepFakeVision AI – EN LIGNE