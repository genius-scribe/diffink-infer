# DiffInk RunPod Serverless Worker
#
# Build:
#   docker build --platform linux/amd64 -t your-user/diffink-infer:latest .
# Push:
#   docker push your-user/diffink-infer:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── PyTorch 2.4.1 + CUDA 12.4 ─────────────────────────────────────────────
# (Last release with SM_70 / V100 support)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python dependencies ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod \
    "numpy>=1.24" \
    einops \
    einx \
    "x-transformers>=1.42" \
    pyyaml \
    tqdm \
    "opencv-python-headless>=4.11" \
    scipy \
    "gdown>=5.0"

# ── Application source ─────────────────────────────────────────────────────
COPY diffink/ ./diffink/
COPY handler.py ./

# ── Download pretrained weights and vocabulary from Google Drive ───────────
# File IDs from https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB
RUN mkdir -p checkpoints data

# Vocabulary (small — download first to fail fast)
RUN python -m gdown 1yQpL0oxC5dv8yXTdWuHsdkaZpAI4XYeQ -O data/All_zi.json \
 && python -m gdown 1V7g1tTXQzuuri28mVQfSY8o2lbZRF9a3 -O data/selected_400_100.json

# InkVAE checkpoint (~171 MB)
RUN python -m gdown 11fprScAKJnML2Dv_BFZ5JDSQ1cQm341t -O checkpoints/vae_epoch_100.pt

# InkDiT checkpoint (~2.8 GB) — gdown auto-confirms large-file prompt
RUN python -m gdown 13sApjo9rqFHdfNnWmiWaRjFVWewJqECY -O checkpoints/dit_epoch_1.pt

# ── Entrypoint ─────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
