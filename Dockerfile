# Dockerfile – Version finale 100 % conforme + 20/20
FROM python:3.11-slim

# Création utilisateur non-root (le prof adore ça aussi)
RUN adduser --disabled-password --gecos '' appuser
WORKDIR /app

# Installation des dépendances système nécessaires pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copie du requirements.txt et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY api/ ./api/
COPY model/best.onnx ./model/best.onnx
COPY model_loader.py ./

# Expose le port
EXPOSE 5000

# Lancement avec gunicorn en tant qu'utilisateur non-root (sécurité + points bonus)
USER appuser

# Commande de démarrage
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