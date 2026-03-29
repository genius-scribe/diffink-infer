"""
RunPod Serverless handler for DiffInk — online handwriting generation.

## Usage

    {
      "reference_strokes": "<base64 float32 [T,5]: x y is_next is_new_stroke is_new_char>",
      "reference_text":    "the full text that the reference strokes represent",
      "target_text":       "the text you want to generate",
      "num_style_chars":   9   // how many leading ref chars to keep as style anchor (default 9)
    }

## How it works

1. Trim reference strokes to the first N characters (style anchor)
2. Use target_text as the generation text
3. Model keeps 30% of the style anchor as prefix, generates the rest conditioned
   on the style + text embedding

The model was trained with prefix_ratio=0.3.  By trimming the reference to a
short style sample, we keep the model in its comfort zone.
"""

import base64
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
# Paths
# ---------------------------------------------------------------------------
_VOLUME = "/runpod-volume"

def _resolve(rel: str) -> str:
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
# Download checkpoints
# ---------------------------------------------------------------------------
_R2_BASE = "https://pub-233b8b390a4b4668b0e5fd1f4cd5a2bf.r2.dev/diffink"
_R2_FILES = [
    (VOCAB_PATH, f"{_R2_BASE}/meta/All_zi.json", 44043),
    (VAE_CKPT,   f"{_R2_BASE}/checkpoints/vae_epoch_100.pt", 178524954),
    (DIT_CKPT,   f"{_R2_BASE}/checkpoints/dit_epoch_1-001.pt", 2914621632),
]

for _path, _url, _min_size in _R2_FILES:
    if not os.path.exists(_path) or os.path.getsize(_path) < _min_size * 0.9:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        _dl = _path + ".downloading"
        while os.path.exists(_dl):
            print(f"[wait] {_path} being downloaded ...")
            import time; time.sleep(5)
            if os.path.exists(_path) and os.path.getsize(_path) >= _min_size * 0.9:
                break
        else:
            if os.path.exists(_path) and os.path.getsize(_path) >= _min_size * 0.9:
                continue
            with open(_dl, "w") as f: f.write(str(os.getpid()))
            try:
                print(f"Downloading {_url} → {_path} ...")
                subprocess.run(["wget", "--show-progress", "-O", _path, _url], check=True)
            finally:
                os.remove(_dl)
    else:
        print(f"[skip] {_path} ({os.path.getsize(_path)} bytes)")

# ---------------------------------------------------------------------------
# Vocabulary & config
# ---------------------------------------------------------------------------
print("Loading vocabulary...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    _vocab = json.load(f)
CHAR_TO_IDX = {k: i for i, k in enumerate(_vocab.keys())}
_NUM_EMB = len(CHAR_TO_IDX) + 1

_cfg = {
    "in_channels": 5, "latent_dim": 384,
    "hidden_dims": [128, 256, 384], "decoder_dims": [384, 256, 128],
    "decoder_output_dim": 123, "num_text_embedding": _NUM_EMB,
    "trans_hidden_dim": 256, "trans_num_heads": 4, "trans_num_layers": 3,
    "ocr_hidden_dim": 384, "ocr_num_heads": 4, "ocr_num_layers": 3,
    "num_writer": 90, "style_classifier_dim": 384,
    "dim": 896, "depth": 16, "heads": 14, "dim_head": 64,
    "dropout": 0.1, "ff_mult": 4, "text_dim": 512,
    "text_mask_padding": True, "conv_layers": 3, "long_skip_connection": False,
}
CONFIG = ModelConfig(_cfg)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def _load_sd(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd

print("Loading InkVAE...")
VAE_MODEL = VAE(CONFIG).to(DEVICE).eval()
_sd = _load_sd(VAE_CKPT)
_ms = VAE_MODEL.state_dict()
_fl = {k: v for k, v in _sd.items() if k in _ms and v.shape == _ms[k].shape}
VAE_MODEL.load_state_dict(_fl, strict=False)

print("Loading InkDiT...")
DIT_MODEL = DiT(CONFIG).to(DEVICE).eval()
DIT_MODEL.load_state_dict(_load_sd(DIT_CKPT))

DIFFUSION = Diffusion(noise_steps=1000, schedule_type="cosine", device=DEVICE)
print("Models ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_to_indices(text):
    indices = []
    for c in text:
        if c not in CHAR_TO_IDX:
            raise ValueError(f"Character '{c}' (U+{ord(c):04X}) not in vocabulary")
        indices.append(CHAR_TO_IDX[c])
    indices.append(CHAR_TO_IDX.get("、", 0))
    return indices

def _derive_char_points_idx(strokes):
    mask = (strokes[:, 2].astype(int) == 0) & \
           (strokes[:, 3].astype(int) == 0) & \
           (strokes[:, 4].astype(int) == 1)
    return np.where(mask)[0].tolist()

def _pad_to_multiple_of_8(strokes):
    T = strokes.shape[0]
    pad_len = (8 - T % 8) % 8
    if pad_len == 0:
        return strokes.copy(), T
    pad_val = np.array([[0, 0, 0, 0, 1]] * pad_len, dtype=np.float32)
    return np.vstack([strokes, pad_val]), T

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event):
    inp = event["input"]

    # --- Inputs ---
    ref_strokes_b64 = inp["reference_strokes"]
    reference_text  = inp["reference_text"]   # text of the style reference
    target_text     = inp["target_text"]
    num_style_chars = int(inp.get("num_style_chars", 9))

    # --- Options ---
    char_points_idx    = inp.get("char_points_idx")
    sampling_timesteps = int(inp.get("sampling_timesteps", 20))
    cfg_scale          = float(inp.get("cfg_scale", 1.0))
    temperature        = float(inp.get("temperature", 0.1))
    greedy             = bool(inp.get("greedy", True))
    output_image       = bool(inp.get("output_image", True))

    # --- Decode reference strokes ---
    try:
        raw = base64.b64decode(ref_strokes_b64)
        ref_strokes = np.frombuffer(raw, dtype=np.float32).reshape(-1, 5)
    except Exception as e:
        return {"error": f"Failed to decode reference_strokes: {e}"}

    # --- Character boundaries ---
    if char_points_idx is None:
        char_points_idx = _derive_char_points_idx(ref_strokes)
    if len(char_points_idx) == 0:
        return {"error": "No character boundaries found in reference strokes"}

    # --- Trim to first N characters as style anchor ---
    if num_style_chars > len(char_points_idx):
        num_style_chars = len(char_points_idx)
    trim_end = char_points_idx[num_style_chars - 1]  # last point of Nth char
    style_strokes = ref_strokes[:trim_end]
    style_char_idx = char_points_idx[:num_style_chars]

    # --- Text: style reference + target ---
    # The model was trained with text matching the style strokes.
    # We need the style text so the model knows what the prefix is "saying".
    style_text = reference_text[:num_style_chars]
    full_text = style_text + target_text

    bad = [c for c in full_text if c not in CHAR_TO_IDX]
    if bad:
        return {"error": f"Characters not in vocabulary ({len(bad)}): {''.join(bad[:20])}"}

    text_indices = _text_to_indices(full_text)
    num_target = len(target_text)

    print(f"style: {num_style_chars} chars ({trim_end} pts), "
          f"text: '{full_text}' ({len(full_text)} chars)")

    # --- Build tensors ---
    padded, seq_len = _pad_to_multiple_of_8(style_strokes)
    T = padded.shape[0]
    data = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, T, dtype=torch.bool, device=DEVICE)
    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        feat = VAE_MODEL.encode(data.permute(0, 2, 1))[0].permute(0, 2, 1)

        latent_mask, latent_padding_mask, _ = build_prefix_mask_from_char_points(
            char_points_idx=[style_char_idx],
            mask=mask,
            compression_factor=8,
            prefix_ratio=0.3,
        )
        final_noise_mask = latent_mask * latent_padding_mask

        x_pred = DIFFUSION.ddim_sample(
            dit_model=DIT_MODEL, n=1, cond=feat,
            text=text_tensor,
            cond_mask=latent_mask,
            padding_mask=latent_padding_mask,
            sampling_timesteps=sampling_timesteps,
            eta=0.0, cfg_scale=cfg_scale,
        )

        x_mix = (
            x_pred.permute(0, 2, 1) * final_noise_mask.unsqueeze(1)
            + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)
        )

        gmm_out = VAE_MODEL.decode(x_mix)
        pi, mu1, mu2, sigma1, sigma2, corr, pen, _ = get_mixture_coef(gmm_out, num_mixture=20)

        params = [pi[0].cpu(), mu1[0].cpu(), mu2[0].cpu(),
                  sigma1[0].cpu(), sigma2[0].cpu(), corr[0].cpu(), pen[0].cpu()]
        recon = sample_from_params(params, temp=temperature, max_seq_len=seq_len, greedy=greedy)

    # --- Output ---
    recon_f32 = recon.astype(np.float32)
    result = {
        "strokes":  base64.b64encode(recon_f32.tobytes()).decode("utf-8"),
        "seq_len":  int(seq_len),
        "shape":    list(recon_f32.shape),
        "style_chars": num_style_chars,
        "target_chars": num_target,
    }

    if output_image:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_line_cv2(recon_f32, tmp, canvas_height=256, padding=20,
                          line_thickness=2, max_dist=5000)
            with open(tmp, "rb") as f:
                result["image"] = base64.b64encode(f.read()).decode("utf-8")
        finally:
            os.unlink(tmp)

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
