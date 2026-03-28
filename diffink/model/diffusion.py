import math
import torch
from tqdm import tqdm


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        noise_offset=0,
        beta_start=1e-4,
        beta_end=0.02,
        device=None,
        schedule_type="linear",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_offset = noise_offset
        self.device = device
        self.schedule_type = schedule_type.lower()

        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self):
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule_type == "cosine":
            return self._cosine_beta_schedule(self.noise_steps)
        else:
            raise ValueError(f"Unsupported schedule_type: {self.schedule_type}")

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @torch.no_grad()
    def ddim_sample(
        self,
        dit_model,
        n,
        cond,
        text,
        cond_mask,
        padding_mask,
        sampling_timesteps=20,
        eta=0.0,
        cfg_scale=1.0,
    ):
        """DDIM sampling with classifier-free guidance.

        Args:
            dit_model: DiT model (eval mode expected).
            n: batch size.
            cond: [B, T_lat, latent_dim] — VAE-encoded prefix (style conditioning).
            text: [B, text_len] — character index sequence.
            cond_mask: [B, T_lat] or [B, T_lat, 1] — 1 where to generate, 0 for prefix.
            padding_mask: [B, T_lat] — 1 for valid, 0 for padding.
            sampling_timesteps: number of DDIM steps.
            eta: DDIM stochasticity (0 = deterministic).
            cfg_scale: classifier-free guidance scale.

        Returns:
            x: [B, T_lat, latent_dim] — generated latent.
        """
        if cond_mask.dim() == 2:
            cond_mask = cond_mask.unsqueeze(-1)
        cond_mask = cond_mask.to(dtype=torch.float32, device=self.device)

        dit_model.eval()
        x = torch.randn_like(cond)

        times = torch.linspace(-1, self.noise_steps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            time_tensor = torch.full((n,), time, device=self.device, dtype=torch.long)

            x_uncond = x
            x_cond = cond * (1 - cond_mask) + x * cond_mask

            x_start_cond = dit_model(
                x=x_cond, noise=x_uncond, text=text, time=time_tensor,
                mask=padding_mask, drop_text=False, drop_cond=False,
            )
            x_start_uncond = dit_model(
                x=x_cond, noise=x_uncond, text=text, time=time_tensor,
                mask=padding_mask, drop_text=False, drop_cond=True,
            )
            x_start = x_start_uncond + cfg_scale * (x_start_cond - x_start_uncond)

            alpha_hat_t = self.alpha_hat[time_tensor][:, None, None]

            if time_next < 0:
                x = x_start
                continue

            time_next_tensor = torch.full((n,), time_next, device=self.device, dtype=torch.long)
            alpha_hat_next = self.alpha_hat[time_next_tensor][:, None, None]

            eps = (x - alpha_hat_t.sqrt() * x_start) / (1 - alpha_hat_t).clamp_min(1e-8).sqrt()
            sigma = eta * torch.sqrt(
                ((1 - alpha_hat_next) / (1 - alpha_hat_t).clamp_min(1e-8)
                 * (1 - alpha_hat_t / alpha_hat_next)).clamp_min(0.0)
            )
            c = torch.sqrt((1 - alpha_hat_next - sigma**2).clamp_min(0.0))

            sample_noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = alpha_hat_next.sqrt() * x_start + c * eps + sigma * sample_noise

        return x
