"""
RunPod Serverless handler for DiffInk — online handwriting generation.

## Usage

    {
      "reference_strokes": "<base64 float32 [T,5]: x y is_next is_new_stroke is_new_char>",
      "reference_text":    "the text that the reference strokes represent",
      "target_text":       "the text you want to generate",
      "temperature":       0.1,
      "sampling_timesteps": 20
    }

## Design notes

The model was trained on (prefix-strokes, text) → (suffix-strokes) where
prefix+suffix = one continuous sentence.  We adapt it for online inference:

1.  full_text  = reference_text + target_text
2.  prefix_ratio = len(ref_chars) / len(full_text)
3.  Estimate target stroke length from reference avg-points-per-char
4.  Pad reference strokes with zero-padding for the target portion
5.  The DiT generates strokes for the suffix (target) portion.

This is an approximation — the model never saw cross-sentence generation
during training — but it works well enough when the style reference is long
enough to give the model a good style anchor.
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
# Paths — prefer network volume (/runpod-volume), fall back to local
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
# Download checkpoints if not present (once per worker lifetime)
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
        _downloading_path = _path + ".downloading"
        while os.path.exists(_downloading_path):
            print(f"[wait] {_path} being downloaded by another worker ...")
            import time; time.sleep(5)
            if os.path.exists(_path) and os.path.getsize(_path) >= _min_size * 0.9:
                print(f"[skip] {_path} ready after waiting")
                break
        else:
            if os.path.exists(_path) and os.path.getsize(_path) >= _min_size * 0.9:
                print(f"[skip] {_path} already exists ({os.path.getsize(_path)} bytes)")
                continue
            with open(_downloading_path, "w") as _f:
                _f.write(str(os.getpid()))
            try:
                print(f"Downloading {_url} → {_path} ...")
                subprocess.run(["wget", "--show-progress", "-O", _path, _url], check=True)
            finally:
                os.remove(_downloading_path)
    else:
        print(f"[skip] {_path} already exists ({os.path.getsize(_path)} bytes)")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
print("Loading vocabulary...")
with open(VOCAB_PATH, "r", encoding="utf-8") as _f:
    _vocab_data = json.load(_f)
CHAR_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_vocab_data.keys())}
_NUM_EMBEDDING = len(CHAR_TO_IDX) + 1

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
_cfg = {
    "in_channels": 5, "latent_dim": 384,
    "hidden_dims": [128, 256, 384], "decoder_dims": [384, 256, 128],
    "decoder_output_dim": 123, "num_text_embedding": _NUM_EMBEDDING,
    "trans_hidden_dim": 256, "trans_num_heads": 4, "trans_num_layers": 3,
    "ocr_hidden_dim": 384, "ocr_num_heads": 4, "ocr_num_layers": 3,
    "num_writer": 90, "style_classifier_dim": 384,
    "dim": 896, "depth": 16, "heads": 14, "dim_head": 64,
    "dropout": 0.1, "ff_mult": 4, "text_dim": 512,
    "text_mask_padding": True, "conv_layers": 3, "long_skip_connection": False,
}
CONFIG = ModelConfig(_cfg)

# ---------------------------------------------------------------------------
# Load models (once, at worker startup)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def _load_sd(path: str) -> dict:
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
print(f"  loaded {len(_fl)} / {len(_ms)} params")

print("Loading InkDiT...")
DIT_MODEL = DiT(CONFIG).to(DEVICE).eval()
DIT_MODEL.load_state_dict(_load_sd(DIT_CKPT))

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
    indices.append(CHAR_TO_IDX.get("、", 0))
    return indices


def _derive_char_points_idx(strokes: np.ndarray) -> list[int]:
    mask = (strokes[:, 2].astype(int) == 0) & \
           (strokes[:, 3].astype(int) == 0) & \
           (strokes[:, 4].astype(int) == 1)
    return np.where(mask)[0].tolist()


def _pad_to_multiple_of_8(strokes: np.ndarray):
    T = strokes.shape[0]
    pad_len = (8 - T % 8) % 8
    if pad_len == 0:
        return strokes.copy(), T
    pad_val = np.array([[0, 0, 0, 0, 1]] * pad_len, dtype=np.float32)
    return np.vstack([strokes, pad_val]), T


def _check_vocab(text: str) -> list[str]:
    return [c for c in text if c not in CHAR_TO_IDX]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    inp = event["input"]

    # --- Inputs ---
    ref_strokes_b64: str = inp["reference_strokes"]
    ref_text: str        = inp["reference_text"]
    target_text: str     = inp["target_text"]

    # --- Options ---
    char_points_idx     = inp.get("char_points_idx")
    sampling_timesteps  = int(inp.get("sampling_timesteps", 20))
    cfg_scale           = float(inp.get("cfg_scale", 1.0))
    temperature         = float(inp.get("temperature", 0.1))
    greedy              = bool(inp.get("greedy", True))
    output_image        = bool(inp.get("output_image", True))

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

    num_ref_chars = len(char_points_idx)

    # --- Verify reference_text matches stroke count ---
    if len(ref_text) != num_ref_chars:
        return {
            "error": f"reference_text length ({len(ref_text)}) != "
                     f"character boundaries in strokes ({num_ref_chars})"
        }

    # --- Full text & vocabulary check ---
    full_text = ref_text + target_text
    bad_chars = _check_vocab(full_text)
    if bad_chars:
        return {"error": f"Characters not in vocabulary ({len(bad_chars)}): {''.join(bad_chars[:20])}"}

    text_indices = _text_to_indices(full_text)
    num_total_chars = len(text_indices) - 1  # exclude trailing marker
    num_target_chars = len(target_text)

    # --- Estimate target stroke length ---
    # avg points per character from the reference
    avg_pts = len(ref_strokes) / num_ref_chars
    target_pts = int(avg_pts * num_target_chars)
    # Pad reference with enough zero-rows for the target portion
    # plus extra to reach a multiple of 8
    target_pts = ((target_pts + 7) // 8) * 8  # round up to multiple of 8

    zero_pad = np.zeros((target_pts, 5), dtype=np.float32)
    # zero_pad[:, 4] = 1  ← DO NOT set is_new_char=1, it pollutes char_points_idx

    full_strokes = np.vstack([ref_strokes, zero_pad])
    full_strokes, total_len = _pad_to_multiple_of_8(full_strokes)

    # --- prefix_ratio ---
    prefix_ratio = num_ref_chars / num_total_chars
    print(f"ref_chars={num_ref_chars} target_chars={num_target_chars} "
          f"total_chars={num_total_chars} prefix_ratio={prefix_ratio:.3f} "
          f"ref_pts={len(ref_strokes)} target_pts≈{target_pts} total_pts={full_strokes.shape[0]}")

    # --- Build tensors ---
    T = full_strokes.shape[0]
    data = torch.tensor(full_strokes, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    mask_ref = np.ones(len(ref_strokes), dtype=bool)
    mask_pad = np.zeros(target_pts, dtype=bool)
    mask_np = np.concatenate([mask_ref, mask_pad])
    # Re-pad to match T
    if len(mask_np) < T:
        mask_np = np.concatenate([mask_np, np.zeros(T - len(mask_np), dtype=bool)])
    mask = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(DEVICE)

    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        feat = VAE_MODEL.encode(data.permute(0, 2, 1))[0].permute(0, 2, 1)

        latent_mask, latent_padding_mask, _ = build_prefix_mask_from_char_points(
            char_points_idx=[char_points_idx],
            mask=mask,
            compression_factor=8,
            prefix_ratio=prefix_ratio,
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
        recon = sample_from_params(params, temp=temperature, max_seq_len=total_len, greedy=greedy)

    # --- Output ---
    recon_f32 = recon.astype(np.float32)
    result = {
        "strokes":  base64.b64encode(recon_f32.tobytes()).decode("utf-8"),
        "seq_len":  int(total_len),
        "shape":    list(recon_f32.shape),
        "prefix_ratio": round(prefix_ratio, 4),
        "ref_chars": num_ref_chars,
        "target_chars": num_target_chars,
        "avg_pts_per_char": round(avg_pts, 1),
    }

    if output_image:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _f:
            tmp = _f.name
        try:
            plot_line_cv2(recon_f32, tmp, canvas_height=256, padding=20,
                          line_thickness=2, max_dist=200)
            with open(tmp, "rb") as _f:
                result["image"] = base64.b64encode(_f.read()).decode("utf-8")
        finally:
            os.unlink(tmp)

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
