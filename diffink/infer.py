"""
DiffInk inference entry point.

Usage:
    python -m diffink.infer [--config configs/inference.yaml] [--output outputs/] [--steps 20] [--cfg 1.0]

Or via the installed script:
    diffink-infer --config configs/inference.yaml
"""

import argparse
import os

import torch
from tqdm import tqdm

from .dataset import build_val_loader
from .model.diffusion import Diffusion
from .model.dit import DiT
from .model.gmm import get_mixture_coef, sample_from_params
from .model.vae import VAE
from .utils.mask import build_prefix_mask_from_char_points
from .utils.utils import ModelConfig, load_config_from_yaml, set_seed
from .utils.visual import plot_line_cv2


def _strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_vae(config: ModelConfig, config_dict: dict, device: torch.device) -> VAE:
    vae = VAE(config).to(device)
    path = config_dict.get("vae_model_path")
    if path and os.path.exists(path):
        print(f"Loading VAE from: {path}")
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = _strip_module_prefix(sd)
        model_sd = vae.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = vae.load_state_dict(filtered, strict=False)
        print(f"  loaded {len(filtered)} params | missing: {len(missing)} | unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {path}")
    return vae


def load_dit(config: ModelConfig, config_dict: dict, device: torch.device) -> DiT:
    dit = DiT(config).to(device)
    path = config_dict.get("dit_resume_ckpt")
    if path and os.path.exists(path):
        print(f"Loading DiT from: {path}")
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = _strip_module_prefix(sd)
        dit.load_state_dict(sd)
    else:
        raise FileNotFoundError(f"DiT checkpoint not found: {path}")
    return dit


@torch.no_grad()
def run_inference(
    dit: DiT,
    vae: VAE,
    val_loader,
    device: torch.device,
    output_dir: str,
    sampling_timesteps: int = 20,
    cfg_scale: float = 1.0,
    prefix_ratio: float = 0.3,
    compression_factor: int = 8,
    max_batches: int | None = None,
):
    """Run DiffInk inference over a DataLoader and save visualisations.

    For each sample the function:
      1. Encodes the full ground-truth sequence with InkVAE (style conditioning).
      2. Builds a prefix mask: the first `prefix_ratio` of characters are kept as
         style context; the remainder is treated as the region to generate.
      3. Runs DDIM sampling with InkDiT conditioned on text + prefix latent.
      4. Mixes the generated suffix back with the encoded prefix.
      5. Decodes the mixed latent with InkVAE to GMM parameters.
      6. Samples strokes from the GMM and saves PNG images.

    Outputs written to `output_dir/`:
      - ``gt_<batch>_<i>.png``    — ground-truth visualisation
      - ``recon_<batch>_<i>.png`` — generated output visualisation
    """
    dit.eval()
    vae.eval()

    diffusion = Diffusion(noise_steps=1000, schedule_type="cosine", device=device)
    os.makedirs(output_dir, exist_ok=True)

    for p, batch in tqdm(enumerate(val_loader), desc="inference", unit="batch"):
        if max_batches is not None and p >= max_batches:
            break

        data, mask, text_idx, char_points_idx, writer_id = batch
        data = data.to(device)
        mask = mask.to(device)
        text_idx = text_idx.to(device)
        batch_size = data.size(0)

        # [B, T, 5] → [B, 5, T]
        data_t = data.permute(0, 2, 1)

        # Encode full sequence (for style prefix)
        feat = vae.encode(data_t)[0].permute(0, 2, 1)  # [B, T_lat, latent_dim]

        # Build prefix / generation masks
        latent_mask, latent_padding_mask, _ = build_prefix_mask_from_char_points(
            char_points_idx=char_points_idx,
            mask=mask,
            compression_factor=compression_factor,
            prefix_ratio=prefix_ratio,
        )
        final_noise_mask = latent_mask * latent_padding_mask  # [B, T_lat]

        # DDIM sampling
        x_pred = diffusion.ddim_sample(
            dit_model=dit,
            n=batch_size,
            cond=feat,
            text=text_idx,
            cond_mask=latent_mask,
            padding_mask=latent_padding_mask,
            sampling_timesteps=sampling_timesteps,
            eta=0.0,
            cfg_scale=cfg_scale,
        )

        # Mix generated suffix with encoded prefix
        x_mix = (
            x_pred.permute(0, 2, 1) * final_noise_mask.unsqueeze(1)
            + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)
        )  # [B, latent_dim, T_lat]

        # Decode
        output = vae.decode(x_mix)  # [B, 123, T_lat] (GMM params)

        # Extract GMM coefficients
        pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = get_mixture_coef(output, num_mixture=20)

        for i in range(batch_size):
            seq_len = int(mask[i].sum().item())
            params = [
                pi[i].cpu(), mu1[i].cpu(), mu2[i].cpu(),
                sigma1[i].cpu(), sigma2[i].cpu(), corr[i].cpu(),
                pen[i].cpu(),
            ]
            recon = sample_from_params(params, temp=0.1, max_seq_len=seq_len, greedy=True)

            gt_path = os.path.join(output_dir, f"gt_{p * batch_size + i}.png")
            recon_path = os.path.join(output_dir, f"recon_{p * batch_size + i}.png")
            plot_line_cv2(data[i], gt_path, canvas_height=256, padding=20, line_thickness=2, max_dist=200)
            plot_line_cv2(recon, recon_path, canvas_height=256, padding=20, line_thickness=2, max_dist=200)

    print(f"Done. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiffInk inference")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--steps", type=int, default=None, help="Override DDIM sampling steps")
    parser.add_argument("--cfg", type=float, default=None, help="Override CFG scale")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches (debugging)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    config_dict = load_config_from_yaml(args.config)

    if args.output is not None:
        config_dict["output_base"] = args.output
    if args.steps is not None:
        config_dict["sampling_timesteps"] = args.steps
    if args.cfg is not None:
        config_dict["cfg_scale"] = args.cfg

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)

    val_loader, config_dict = build_val_loader(config_dict)
    config = ModelConfig(config_dict)

    vae = load_vae(config, config_dict, device)
    dit = load_dit(config, config_dict, device)

    output_dir = config_dict.get("output_base", "./outputs/inference")
    sampling_timesteps = config_dict.get("sampling_timesteps", 20)
    cfg_scale = config_dict.get("cfg_scale", 1.0)

    run_inference(
        dit=dit,
        vae=vae,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        sampling_timesteps=sampling_timesteps,
        cfg_scale=cfg_scale,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
