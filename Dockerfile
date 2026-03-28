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
        python3 python3-pip \
        libgl1 libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── uv ────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# ── Python dependencies ────────────────────────────────────────────────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# ── Application source ─────────────────────────────────────────────────────
COPY diffink/ ./diffink/
COPY handler.py ./

# ── Download pretrained weights and vocabulary from Google Drive ───────────
# File IDs from https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB
RUN mkdir -p checkpoints data

# Character vocabulary (~44 KB) — download first to fail fast
RUN python -m gdown 1yQpL0oxC5dv8yXTdWuHsdkaZpAI4XYeQ -O data/All_zi.json

# InkVAE checkpoint (~171 MB)
RUN python -m gdown 11fprScAKJnML2Dv_BFZ5JDSQ1cQm341t -O checkpoints/vae_epoch_100.pt

# InkDiT checkpoint (~2.8 GB)
RUN python -m gdown 13sApjo9rqFHdfNnWmiWaRjFVWewJqECY -O checkpoints/dit_epoch_1.pt

# ── Entrypoint ─────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
