import numpy as np
import torch
import torch.nn.functional as F

INF_MIN = 1e-8


def get_mixture_coef(output, num_mixture=20):
    """Decompose VAE decoder output into GMM parameters.

    Args:
        output: [B, 3 + num_mixture*6, T]

    Returns:
        [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
    """
    pen_logits = output[:, :3, :]
    gmm_params = output[:, 3:, :]
    pi, mu1, mu2, sigma1, sigma2, corr = torch.split(gmm_params, num_mixture, dim=1)

    pi = torch.softmax(pi, dim=1)
    pen = torch.softmax(pen_logits, dim=1)
    sigma1 = torch.exp(sigma1)
    sigma2 = torch.exp(sigma2)
    corr = torch.tanh(corr)

    return [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    s1 *= sqrt_temp**2
    s2 *= sqrt_temp**2
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    return np.random.multivariate_normal([mu1, mu2], cov)


def sample_from_params(params, temp=0.1, max_seq_len=400, greedy=False):
    """Sample a stroke sequence from GMM parameters.

    Args:
        params: list [pi, mu1, mu2, sigma1, sigma2, corr, pen] — all CPU tensors.
        temp: temperature for Gaussian sampling.
        max_seq_len: maximum number of steps.
        greedy: if True, use mean instead of sampling.

    Returns:
        strokes: np.ndarray of shape [seq_len, 5].
    """
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params
    num_mixture, seq_len = o_pi.shape
    strokes = np.zeros((seq_len, 5), dtype=np.float32)

    for step in range(min(max_seq_len, seq_len)):
        idx = torch.distributions.Categorical(o_pi[:, step]).sample().item()
        x1, x2 = sample_gaussian_2d(
            o_mu1[idx, step].item(),
            o_mu2[idx, step].item(),
            o_sigma1[idx, step].item(),
            o_sigma2[idx, step].item(),
            o_corr[idx, step].item(),
            sqrt_temp=np.sqrt(temp),
            greedy=greedy,
        )
        eos = [0, 0, 0]
        eos[np.argmax(o_pen[:, step].cpu().numpy())] = 1
        strokes[step] = [x1, x2] + eos

    return strokes
