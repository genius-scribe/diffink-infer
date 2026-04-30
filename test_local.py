"""DiffInk 本地推理测试（不需要 RunPod）。

用法:
    cd /root/autodl-tmp/suchuan
    python diffink-infer/test_local.py
    python diffink-infer/test_local.py --test_input diffink-infer/test_input.json
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from diffink.model.diffusion import Diffusion
from diffink.model.dit import DiT
from diffink.model.gmm import get_mixture_coef, sample_from_params
from diffink.model.vae import VAE
from diffink.utils.mask import build_prefix_mask_from_char_points
from diffink.utils.utils import ModelConfig
from diffink.utils.visual import plot_line_cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=os.path.join(SCRIPT_DIR, "checkpoints"))
    parser.add_argument("--vocab", default=os.path.join(SCRIPT_DIR, "data", "All_zi.json"))
    parser.add_argument("--test_input", default=os.path.join(SCRIPT_DIR, "test_input.json"),
                        help="test_input.json 路径")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "test_outputs"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载词汇表 ----
    print("Loading vocabulary...")
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_idx = {k: i for i, k in enumerate(vocab.keys())}
    num_emb = len(char_to_idx) + 1

    # ---- 模型配置 ----
    cfg = {
        "in_channels": 5, "latent_dim": 384,
        "hidden_dims": [128, 256, 384], "decoder_dims": [384, 256, 128],
        "decoder_output_dim": 123, "num_text_embedding": num_emb,
        "trans_hidden_dim": 256, "trans_num_heads": 4, "trans_num_layers": 3,
        "ocr_hidden_dim": 384, "ocr_num_heads": 4, "ocr_num_layers": 3,
        "num_writer": 90, "style_classifier_dim": 384,
        "dim": 896, "depth": 16, "heads": 14, "dim_head": 64,
        "dropout": 0.1, "ff_mult": 4, "text_dim": 512,
        "text_mask_padding": True, "conv_layers": 3, "long_skip_connection": False,
    }
    config = ModelConfig(cfg)

    # ---- 加载模型 ----
    print(f"Loading models on {device}...")

    def load_sd(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        return sd

    vae_path = os.path.join(args.ckpt_dir, "vae_epoch_100.pt")
    dit_path = os.path.join(args.ckpt_dir, "dit_epoch_1.pt")

    print("  Loading InkVAE...")
    vae_model = VAE(config).to(device).eval()
    sd = load_sd(vae_path)
    ms = vae_model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    vae_model.load_state_dict(filtered, strict=False)

    print("  Loading InkDiT...")
    dit_model = DiT(config).to(device).eval()
    dit_model.load_state_dict(load_sd(dit_path))

    diffusion = Diffusion(noise_steps=1000, schedule_type="cosine", device=device)
    print("Models ready.\n")

    # ---- 加载测试数据 ----
    if not os.path.exists(args.test_input):
        print(f"test_input.json not found: {args.test_input}")
        print("请先运行 make_test_input.py 生成测试数据")
        return

    with open(args.test_input) as f:
        test_data = json.load(f)

    inp = test_data.get("input", test_data)

    # 兼容新旧两种 API 字段名
    ref_strokes_b64 = inp.get("reference_strokes") or inp.get("style_strokes")
    if not ref_strokes_b64:
        print("ERROR: test_input.json 缺少 reference_strokes 或 style_strokes 字段")
        return

    raw = base64.b64decode(ref_strokes_b64)
    ref_strokes = np.frombuffer(raw, dtype=np.float32).reshape(-1, 5)

    # 旧版用 "text" 表示目标文字，新版用 "reference_text" + "target_text"
    if "reference_text" in inp:
        reference_text = inp["reference_text"]
        target_text = inp["target_text"]
    else:
        # 旧版格式：text 是目标，参考文本从笔迹中推断
        target_text = inp.get("text", "")
        # 用目标文本的前几个字作为参考文本的占位
        reference_text = target_text
    num_style_chars = int(inp.get("num_style_chars", 9))

    print(f"Reference text: '{reference_text}'")
    print(f"Target text: '{target_text}'")
    print(f"Style chars: {num_style_chars}")
    print(f"Reference strokes: {ref_strokes.shape}")

    # ---- 推理逻辑 (和 handler.py 一致) ----
    def text_to_indices(text):
        indices = []
        for c in text:
            if c not in char_to_idx:
                raise ValueError(f"Character '{c}' not in vocabulary")
            indices.append(char_to_idx[c])
        indices.append(char_to_idx.get("、", 0))
        return indices

    def derive_char_points_idx(strokes):
        mask = (strokes[:, 2].astype(int) == 0) & \
               (strokes[:, 3].astype(int) == 0) & \
               (strokes[:, 4].astype(int) == 1)
        return np.where(mask)[0].tolist()

    def pad_to_multiple_of_8(strokes):
        T = strokes.shape[0]
        pad_len = (8 - T % 8) % 8
        if pad_len == 0:
            return strokes.copy(), T
        pad_val = np.array([[0, 0, 0, 0, 1]] * pad_len, dtype=np.float32)
        return np.vstack([strokes, pad_val]), T

    char_points_idx = inp.get("char_points_idx") or derive_char_points_idx(ref_strokes)
    ref_num_chars = len(char_points_idx)
    if num_style_chars > ref_num_chars:
        num_style_chars = ref_num_chars

    style_text = reference_text[:num_style_chars]
    full_text = style_text + target_text
    text_indices = text_to_indices(full_text)
    total_chars = len(full_text)
    prefix_ratio = num_style_chars / total_chars
    num_target = len(target_text)

    total_ref_pts = ref_strokes.shape[0]
    avg_pts_per_char = total_ref_pts / ref_num_chars
    needed_pts = int(avg_pts_per_char * total_chars)

    if needed_pts <= total_ref_pts:
        work_strokes = ref_strokes[:needed_pts].copy()
    else:
        reps = (needed_pts // total_ref_pts) + 1
        tiled = np.tile(ref_strokes, (reps, 1))[:needed_pts]
        work_strokes = tiled.copy()

    style_char_idx = char_points_idx[:num_style_chars]
    prefix_end_pt = style_char_idx[-1] if num_style_chars > 0 else 0
    all_char_idx = list(style_char_idx)
    suffix_pts = needed_pts - prefix_end_pt
    for i in range(num_target):
        idx = prefix_end_pt + int(suffix_pts * (i + 1) / num_target) - 1
        idx = min(idx, needed_pts - 1)
        all_char_idx.append(idx)

    print(f"\nInference: {total_chars} chars, {needed_pts} pts, prefix_ratio={prefix_ratio:.2f}")

    padded, orig_len = pad_to_multiple_of_8(work_strokes)
    T = padded.shape[0]
    data = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    text_tensor = torch.tensor(text_indices, dtype=torch.long).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        feat = vae_model.encode(data.permute(0, 2, 1))[0].permute(0, 2, 1)

        latent_mask, latent_padding_mask, _ = build_prefix_mask_from_char_points(
            char_points_idx=[all_char_idx],
            mask=mask,
            compression_factor=8,
            prefix_ratio=prefix_ratio,
        )
        final_noise_mask = latent_mask * latent_padding_mask

        x_pred = diffusion.ddim_sample(
            dit_model=dit_model, n=1, cond=feat,
            text=text_tensor,
            cond_mask=latent_mask,
            padding_mask=latent_padding_mask,
            sampling_timesteps=args.steps,
            eta=0.0, cfg_scale=args.cfg_scale,
        )

        x_mix = (
            x_pred.permute(0, 2, 1) * final_noise_mask.unsqueeze(1)
            + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)
        )

        gmm_out = vae_model.decode(x_mix)
        pi, mu1, mu2, sigma1, sigma2, corr, pen, _ = get_mixture_coef(gmm_out, num_mixture=20)

        params = [pi[0].cpu(), mu1[0].cpu(), mu2[0].cpu(),
                  sigma1[0].cpu(), sigma2[0].cpu(), corr[0].cpu(), pen[0].cpu()]
        recon = sample_from_params(params, temp=args.temperature, max_seq_len=orig_len, greedy=True)

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.2f}s")
    print(f"Output shape: {recon.shape}")

    # 保存结果
    recon_f32 = recon.astype(np.float32)

    # 渲染 PNG
    out_path = os.path.join(args.output_dir, "diffink_output.png")
    plot_line_cv2(recon_f32, out_path, canvas_height=256, padding=20,
                  line_thickness=2, max_dist=200)
    print(f"Output image: {out_path}")

    # 保存笔迹数据
    np.save(os.path.join(args.output_dir, "diffink_output.npy"), recon_f32)
    print(f"Output data: {args.output_dir}/diffink_output.npy")

    print("\nDone!")


if __name__ == "__main__":
    main()
