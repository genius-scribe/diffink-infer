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
uv run python download_checkpoints.py

# Generate a test payload from the validation set
uv run python make_test_input.py --h5 data/val.h5 --idx 0 --out test_input.json

# Run the handler locally
uv run python handler.py
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

## AutoDL Pro deployment

In addition to RunPod Serverless, this code is deployed on AutoDL Pro instance `pro-7772d41fb379` ("diffink") for hands-on debugging. Project tree on the instance:

```
/root/suchuan/
├── diffink-infer/        # this repo
├── inksight-torch/       # InkSight stroke-recovery weights
├── fontdiffuser-inference/
└── 严寒_kvenjoy_严寒.{gfont,json}
```

Quick verification on the Pro instance (uses uv, no miniconda):

```bash
cd /root/suchuan/diffink-infer
uv run python test_local.py \
  --test_input test/test_input_short.json \
  --output_dir /tmp/verify
```

Expected: ckpt loads + 10-step DDIM sampling completes in ~1 s on RTX 4080 SUPER / vGPU-32GB; outputs `diffink_output.{png,npy}`.

> **First-time setup on a fresh Pro:** install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv sync --extra local --python 3.12`. To download torch CUDA wheels faster from inside China, temporarily swap the index in `pyproject.toml` to the aliyun mirror:
> ```bash
> sed -i 's|https://download.pytorch.org/whl/cu124|https://mirrors.aliyun.com/pytorch-wheels/cu124|; /explicit = true/a format = "flat"' pyproject.toml
> uv sync --extra local --python 3.12
> # then revert pyproject.toml before committing
> ```

## Experiment archive

This repo also serves as an archive of the experiments behind the bug-finding work. Documents and test data are preserved as evidence:

- [`docs/DiffInk_VAE_泄漏问题.md`](docs/) — VAE boundary leakage analysis (the headline finding)
- [`docs/DiffInk_数据格式与gfont转换说明.md`](docs/) — data format + gfont conversion notes (with reference plots)
- [`docs/DiffInk_训练损失与联合训练说明.md`](docs/) — training loss + joint training notes
- [`docs/DiffInk应用中出现的问题.pdf`](docs/) — bound PDF report
- [`docs/试验总结.md`](docs/) — experiment summary (sections 1-6 of the report, narrative form)
- [`test/README.md`](test/README.md) — test-data manifest (which input → which output, parameters, dates)

Two reproducible inference runs are kept under `test/test_outputs_*/`:

| Test | Reference | Target | Output |
|------|-----------|--------|--------|
| 8.3 sentence | `天地黄宇宙洪荒严寒永` (10 chars, gfont) | `春风又绿江南岸明月何时照我还` (14 chars) | `test/test_outputs_sentence_v3/` |
| 8.2 InkSight | `人生得意须尽` (6 chars, InkSight-recovered) | `春夏秋冬东南西北上下` (10 random chars) | `test/test_outputs_inksight_6s10t/` |

Both runs use the post-fix mask (see below).

## Bug fixes & history

### handler.py mask — fixed 2026-04-27

Earlier handler used `mask = torch.ones(1, T)`, treating every position (including 8-multiple padding) as a real stroke. The paper's `ValDataset.collate_fn` uses `mask = ~(data == pad_val).all(dim=-1)` — only real positions are `True`. The all-ones mask let DiT attend to padding latents at the tail; the difference is small in practice (pad < 1 latent in tested cases) but structurally wrong.

Current code matches the paper:

```python
mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
mask[0, :orig_len] = True
```

> ⚠️ Commit `e269ba6 "fix: mask must be all-1s ..."` is a **misnamed regression** — its title is wrong and its change is reverted by the current code. Kept in history as evidence of the original mistake.

The mask logic in independent experiment scripts (e.g. `test_val_h5.py`, removed during cleanup) always used the paper-consistent form, so historical experiment outputs under `test/test_outputs_*` (now archived locally only) are still valid analysis.
