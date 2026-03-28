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

# ── Entrypoint ─────────────────────────────────────────────────────────────
# Checkpoints (~3 GB total) are downloaded at worker startup by handler.py
# via gdown if not already present on the container's disk.
CMD ["python", "-u", "handler.py"]
