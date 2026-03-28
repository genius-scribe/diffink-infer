# diffink-infer

RunPod Serverless inference worker for [DiffInk](https://github.com/awei669/DiffInk) — a glyph- and style-aware latent diffusion transformer for Chinese online handwriting generation (ICLR 2026).

## How it works

Two models run in sequence at inference time:

1. **InkVAE** — 1D convolutional VAE. Encodes a pen-stroke sequence `[T, 5]` into a latent `[T/8, 384]` with 8× temporal compression. Decodes back to GMM parameters (20-mixture, 3 pen states = 123-dim output per timestep).
2. **InkDiT** — Diffusion Transformer. Given the VAE latent as conditioning, generates new latent vectors for the non-prefix region via DDIM sampling.

The **style prefix** is the first ~30% of characters from the input stroke sequence. These are held fixed; only the remaining characters are regenerated, inheriting the writer's style from the prefix.

## Stroke format

All stroke data uses a `[T, 5]` float32 array with columns:

| Index | Name | Description |
|-------|------|-------------|
| 0 | x | Pen x position |
| 1 | y | Pen y position |
| 2 | is_next | 1 = continue current stroke, 0 = end of stroke |
| 3 | is_new_stroke | 1 = start of new stroke |
| 4 | is_new_char | 1 = end of character (combined with is_next=0, is_new_stroke=0) |

Arrays are passed over the API as **base64-encoded raw float32 bytes** (little-endian, row-major).

## RunPod API

The endpoint follows the standard [RunPod Serverless](https://docs.runpod.io/serverless/overview) request/response envelope.

### Submit a job

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

```json
{
  "input": {
    "text": "春风又绿江南岸",
    "style_strokes": "<base64-encoded float32 array>",
    "char_points_idx": [42, 91, 138, 187, 241, 290, 344],
    "sampling_timesteps": 20,
    "cfg_scale": 1.0,
    "temperature": 0.1,
    "greedy": true,
    "output_image": true
  }
}
```

**Response:**

```json
{ "id": "abc123", "status": "IN_QUEUE" }
```

### Poll for result

```
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
Authorization: Bearer {API_KEY}
```

### Input fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Chinese characters to generate. Every character must be present in the vocabulary (`data/All_zi.json`). |
| `style_strokes` | string | yes | — | Base64-encoded `[T, 5]` float32 array. Provides both the style reference and the prefix used for conditioning. |
| `char_points_idx` | list[int] | no | auto | Index of the last point of each character in `style_strokes`. Auto-derived from the `is_new_char` column if omitted. |
| `sampling_timesteps` | int | no | `20` | Number of DDIM denoising steps. Higher = slightly better quality, slower. |
| `cfg_scale` | float | no | `1.0` | Classifier-free guidance scale. `1.0` = no guidance. |
| `temperature` | float | no | `0.1` | GMM sampling temperature. Lower = less variation. |
| `greedy` | bool | no | `true` | Use greedy (mean) GMM sampling instead of stochastic. |
| `output_image` | bool | no | `true` | Include a rendered PNG in the response. |

### Output fields

| Field | Type | Description |
|-------|------|-------------|
| `strokes` | string | Base64-encoded `[seq_len, 5]` float32 array of generated pen strokes. |
| `seq_len` | int | Number of valid stroke points (excluding padding). |
| `shape` | list[int] | Shape of the strokes array, e.g. `[802, 5]`. |
| `image` | string | Base64-encoded PNG of the rendered line (omitted if `output_image=false`). |
| `error` | string | Present only on failure; describes what went wrong. |

### Decode strokes (Python)

```python
import base64, numpy as np

strokes = np.frombuffer(base64.b64decode(response["output"]["strokes"]), dtype=np.float32)
strokes = strokes.reshape(response["output"]["shape"])
# strokes: [T, 5] — (x, y, is_next, is_new_stroke, is_new_char)
```

### Decode image (Python)

```python
import base64
from PIL import Image
import io

img = Image.open(io.BytesIO(base64.b64decode(response["output"]["image"])))
img.save("output.png")
```

## Local development

```bash
# Install deps (requires uv)
uv sync --extra local

# Download checkpoints
python download_checkpoints.py

# Generate a test payload from the validation set
python make_test_input.py --h5 data/val.h5 --idx 0 --out test_input.json

# Run the handler locally
python handler.py
# In another terminal:
# curl -s http://localhost:8000 -d @test_input.json | python -m json.tool
```

## Docker

```bash
# Build
docker build --platform linux/amd64 -t your-user/diffink-infer:latest .

# Push
docker push your-user/diffink-infer:latest
```

The image is based on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, which is pre-cached on RunPod worker nodes. Only the ~200 MB diff (pip deps + source) is pulled at deploy time.

Pretrained checkpoints (~3 GB total) are downloaded from Google Drive on first worker startup and cached on the container's disk for the lifetime of that worker.

## Requirements

- GPU with CUDA compute capability ≥ 7.0 (V100 or newer), ≥ 6 GB VRAM
- PyTorch 2.4.x + CUDA 12.4
