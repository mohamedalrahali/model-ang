# Image prête pour hébergement gratuit (Render, Fly.io, Railway, etc.)
FROM python:3.12-slim-bookworm

WORKDIR /app

# LightGBM / numpy utilisent parfois OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY serve.py .

# Artefacts démo intégrés à l’image (pas besoin de volume sur l’offre gratuite)
RUN python scripts/bootstrap_demo.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render, Fly.io, Railway, etc. définissent PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
