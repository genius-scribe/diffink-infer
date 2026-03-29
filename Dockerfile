# DiffInk RunPod Serverless Worker
#
# Base image is pre-cached on all RunPod worker nodes — only the diff
# (pip deps + source) is pulled, keeping cold-start image pulls to ~200 MB.
#
# Build:
#   docker build --platform linux/amd64 -t your-user/diffink-infer:latest .
# Push:
#   docker push your-user/diffink-infer:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── Extra system libs for opencv ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (torch already provided by base image) ─────────────
RUN pip install --no-cache-dir \
    runpod \
    "numpy>=1.24" \
    einops \
    einx \
    "x-transformers>=1.42" \
    pyyaml \
    tqdm \
    "opencv-python-headless>=4.11"

# ── Application source ─────────────────────────────────────────────────────
COPY diffink/ ./diffink/
COPY handler.py ./

# ── Entrypoint ─────────────────────────────────────────────────────────────
# Checkpoints (~3 GB total) are downloaded at worker startup by handler.py
# from Cloudflare R2 public bucket if not already present on the container's disk.
CMD ["python", "-u", "handler.py"]
