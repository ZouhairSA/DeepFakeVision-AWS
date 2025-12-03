# Dockerfile – Version finale 100 % fonctionnelle (testée le 03/12/2025)
FROM python:3.11-slim

WORKDIR /app

# Dépendances système CORRIGÉES pour OpenCV + ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY api/ ./api/
COPY model/best.onnx ./model/best.onnx
COPY model_loader.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"]


# FROM python:3.11-slim

# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .

# EXPOSE 5000
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 5000

# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]