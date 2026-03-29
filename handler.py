"""
RunPod Serverless handler for DiffInk online handwriting generation.

Input schema (event["input"]):
  text              str   — Chinese characters to generate (must be in vocabulary)
  style_strokes     str   — Base64-encoded float32 array, shape [T, 5].
                            Columns: (x, y, is_next, is_new_stroke, is_new_char).
                            Used as the style reference; the first ~30% of characters
                            act as a conditioning prefix, the rest are regenerated.
  char_points_idx   list  — (optional) int indices of the last point of each character.
                            Auto-derived from is_new_char column if omitted.
  sampling_timesteps int  — DDIM steps (default 20)
  cfg_scale         float — classifier-free guidance scale (default 1.0)
  temperature       float — GMM sampling temperature (default 0.1)
  greedy            bool  — use greedy (mean) GMM sampling (default true)
  output_image      bool  — include base64 PNG in response (default true)

Output schema:
  strokes           str   — Base64-encoded float32 array, shape [seq_len, 5]
  image             str   — Base64-encoded PNG (if output_image=true)
  seq_len           int   — number of valid stroke points
  shape             list  — [seq_len, 5]
"""

import base64
import io
import json
import os
import tempfile

import subprocess

import numpy as np
import runpod
import torch

from diffink.model.diffusion import Diffusion
from diffink.model.dit import DiT
from diffink.model.gmm import get_mixture_coef, sample_from_params
from diffink.model.vae import VAE
from diffink.utils.mask import build_prefix_mask_from_char_points
from diffink.utils.utils import ModelConfig
from diffink.utils.visual import plot_line_cv2

# ---------------------------------------------------------------------------
# Paths — prefer network volume (/runpod-volume), fall back to local
# ---------------------------------------------------------------------------
_VOLUME = "/runpod-volume"

def _resolve(rel: str) -> str:
    """Return volume path if volume is mounted, else local path."""
    vol_path = os.path.join(_VOLUME, rel)
    if os.path.exists(vol_path):
        return vol_path
    if os.path.ismount(_VOLUME):
        return vol_path
    return rel

VAE_CKPT   = os.environ.get("VAE_CKPT",   _resolve("checkpoints/vae_epoch_100.pt"))
DIT_CKPT   = os.environ.get("DIT_CKPT",   _resolve("checkpoints/dit_epoch_1.pt"))
VOCAB_PATH = os.environ.get("VOCAB_PATH", _resolve("data/All_zi.json"))

# ---------------------------------------------------------------------------
# Download checkpoints if not present (runs once per worker lifetime)
# Uses Cloudflare R2 public bucket (no rate limits, fast CDN)
# Downloads to network volume when available so files persist across restarts
# ---------------------------------------------------------------------------
_R2_BASE = "https://pub-233b8b390a4b4668b0e5fd1f4cd5a2bf.r2.dev/diffink"

_R2_FILES = [
    (VOCAB_PATH, f"{_R2_BASE}/meta/All_zi.json"),                      # ~44 KB
    (VAE_CKPT,   f"{_R2_BASE}/checkpoints/vae_epoch_100.pt"),           # ~171 MB
    (DIT_CKPT,   f"{_R2_BASE}/checkpoints/dit_epoch_1-001.pt"),         # ~2.8 GB
]

for _path, _url in _R2_FILES:
    if not os.path.exists(_path):
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        print(f"Downloading {_url} → {_path} ...")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", _path, _url],
            check=True,
        )

# ---------------------------------------------------------------------------
# Load character vocabulary
# ---------------------------------------------------------------------------
print("Loading vocabulary...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    _vocab_data = json.load(f)
CHAR_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_vocab_data.keys())}
NUM_TEXT_EMBEDDING = len(CHAR_TO_IDX) + 1  # index 0 reserved for padding

# ---------------------------------------------------------------------------
# Model config (must match the pretrained checkpoints)
# ---------------------------------------------------------------------------
_config_dict = {
    # InkVAE
    "in_channels": 5,
    "latent_dim": 384,
    "hidden_dims": [128, 256, 384],
    "decoder_dims": [384, 256, 128],
    "decoder_output_dim": 123,
    "num_text_embedding": NUM_TEXT_EMBEDDING,
    "trans_hidden_dim": 256,
    "trans_num_heads": 4,
    "trans_num_layers": 3,
    "ocr_hidden_dim": 384,
    "ocr_num_heads": 4,
    "ocr_num_layers": 3,
    "num_writer": 90,
    "style_classifier_dim": 384,
    # InkDiT
    "dim": 896,
    "depth": 16,
    "heads": 14,
    "dim_head": 64,
    "dropout": 0.1,
    "ff_mult": 4,
    "text_dim": 512,
    "text_mask_padding": True,
    "conv_layers": 3,
    "long_skip_connection": False,
}
CONFIG = ModelConfig(_config_dict)

# ---------------------------------------------------------------------------
# Load models (once, at worker startup)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


print("Loading InkVAE...")
VAE_MODEL = VAE(CONFIG).to(DEVICE).eval()
_sd = _load_state_dict(VAE_CKPT)
_model_sd = VAE_MODEL.state_dict()
_filtered = {k: v for k, v in _sd.items() if k in _model_sd and v.shape == _model_sd[k].shape}
VAE_MODEL.load_state_dict(_filtered, strict=False)
print(f"  loaded {len(_filtered)} / {len(_model_sd)} params")

print("Loading InkDiT...")
DIT_MODEL = DiT(CONFIG).to(DEVICE).eval()
DIT_MODEL.load_state_dict(_load_state_dict(DIT_CKPT))

DIFFUSION = Diffusion(noise_steps=1000, schedule_type="cosine", device=DEVICE)
print("Models ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_to_indices(text: str) -> list[int]:
    indices = []
    for c in text:
        if c not in CHAR_TO_IDX:
            raise ValueError(f"Character '{c}' (U+{ord(c):04X}) not in vocabulary")
        indices.append(CHAR_TO_IDX[c])
    # Trailing marker consistent with ValDataset (appends '、')
    indices.append(CHAR_TO_IDX.get("、", 0))
    return indices


def _derive_char_points_idx(strokes: np.ndarray) -> list[int]:
    """Find character-end positions from pen-state column 4 (is_new_char)."""
    is_next      = strokes[:, 2].astype(int)
    is_new_stroke = strokes[:, 3].astype(int)
    is_new_char  = strokes[:, 4].astype(int)
    idx = np.where((is_next == 0) & (is_new_stroke == 0) & (is_new_char == 1))[0]
    return idx.tolist()


def _pad_to_multiple_of_8(strokes: np.ndarray):
    """Pad stroke sequence to length divisible by 8 (VAE compression factor)."""
    T = strokes.shape[0]
    pad_len = (8 - T % 8) % 8
    if pad_len == 0:
        return strokes.copy(), T
    pad_val = np.array([[0, 0, 0, 0, 1]] * pad_len, dtype=np.float32)
    return np.vstack([strokes, pad_val]), T  # return original T as seq_len


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    inp = event["input"]

    # --- Required inputs ---
    text: str = inp["text"]
    strokes_b64: str = inp["style_strokes"]

    # --- Optional inputs ---
    char_points_idx = inp.get("char_points_idx")          # list[int] or None
    sampling_timesteps = int(inp.get("sampling_timesteps", 20))
    cfg_scale          = float(inp.get("cfg_scale", 1.0))
    temperature        = float(inp.get("temperature", 0.1))
    greedy             = bool(inp.get("greedy", True))
    output_image       = bool(inp.get("output_image", True))

    # --- Decode style strokes ---
    try:
        raw = base64.b64decode(strokes_b64)
        strokes = np.frombuffer(raw, dtype=np.float32).reshape(-1, 5)
    except Exception as e:
        return {"error": f"Failed to decode style_strokes: {e}"}

    # --- Derive character boundaries ---
    if char_points_idx is None:
        char_points_idx = _derive_char_points_idx(strokes)
    if len(char_points_idx) == 0:
        return {
            "error": (
                "No character boundaries found in style_strokes. "
                "Ensure pen-state column 4 contains end-of-character markers "
                "(is_next=0, is_new_stroke=0, is_new_char=1), or supply char_points_idx."
            )
        }

    # --- Text indices ---
    try:
        text_indices = _text_to_indices(text)
    except ValueError as e:
        return {"error": str(e)}

    # --- Build tensors ---
    strokes_padded, seq_len = _pad_to_multiple_of_8(strokes)
    T = strokes_padded.shape[0]

    data = torch.tensor(strokes_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)   # [1, T, 5]

    pad_val = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    is_pad  = np.all(strokes_padded == pad_val, axis=1)
    mask    = torch.tensor(~is_pad, dtype=torch.bool).unsqueeze(0).to(DEVICE)           # [1, T]

    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, L]

    # --- Inference ---
    with torch.no_grad():
        feat = VAE_MODEL.encode(data.permute(0, 2, 1))[0].permute(0, 2, 1)  # [1, T_lat, D]

        latent_mask, latent_padding_mask, _ = build_prefix_mask_from_char_points(
            char_points_idx=[char_points_idx],
            mask=mask,
            compression_factor=8,
            prefix_ratio=0.3,
        )
        final_noise_mask = latent_mask * latent_padding_mask  # [1, T_lat]

        x_pred = DIFFUSION.ddim_sample(
            dit_model=DIT_MODEL,
            n=1,
            cond=feat,
            text=text_tensor,
            cond_mask=latent_mask,
            padding_mask=latent_padding_mask,
            sampling_timesteps=sampling_timesteps,
            eta=0.0,
            cfg_scale=cfg_scale,
        )

        x_mix = (
            x_pred.permute(0, 2, 1) * final_noise_mask.unsqueeze(1)
            + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)
        )  # [1, D, T_lat]

        gmm_out = VAE_MODEL.decode(x_mix)  # [1, 123, T_lat]
        pi, mu1, mu2, sigma1, sigma2, corr, pen, _ = get_mixture_coef(gmm_out, num_mixture=20)

        params = [
            pi[0].cpu(), mu1[0].cpu(), mu2[0].cpu(),
            sigma1[0].cpu(), sigma2[0].cpu(), corr[0].cpu(),
            pen[0].cpu(),
        ]
        recon = sample_from_params(params, temp=temperature, max_seq_len=seq_len, greedy=greedy)

    # --- Encode output ---
    recon_f32 = recon.astype(np.float32)
    result = {
        "strokes":  base64.b64encode(recon_f32.tobytes()).decode("utf-8"),
        "seq_len":  int(seq_len),
        "shape":    list(recon_f32.shape),
    }

    if output_image:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_line_cv2(recon_f32, tmp, canvas_height=256, padding=20,
                          line_thickness=2, max_dist=200)
            with open(tmp, "rb") as f:
                result["image"] = base64.b64encode(f.read()).decode("utf-8")
        finally:
            os.unlink(tmp)

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
