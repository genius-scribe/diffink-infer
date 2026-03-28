import torch


def downsample_mask(mask, compression_factor):
    B, T = mask.shape
    valid_T = (T // compression_factor) * compression_factor
    mask = mask[:, :valid_T]
    downsampled = mask.reshape(B, -1, compression_factor).float().mean(dim=2)
    return (downsampled > 0.0).float()


def build_prefix_mask_from_char_points(
    char_points_idx,
    mask,           # [B, T]
    compression_factor=8,
    prefix_ratio=0.3,
    max_label_len=None,
):
    """Build latent-space masks for prefix-conditioned generation.

    Returns:
        latent_mask:    [B, T_latent] — 1 where noise should be generated (suffix)
        latent_pad_mask:[B, T_latent] — 1 for valid (non-padded) positions
        prefix_label_mask: [B, L]    — 1 for suffix characters (to be generated)
    """
    B, T = mask.shape
    device = mask.device
    total_lengths = mask.sum(dim=1).tolist()

    full_mask = torch.zeros(B, T, dtype=torch.float32, device=device)
    max_num_chars = max(len(x) for x in char_points_idx) if max_label_len is None else max_label_len
    prefix_label_mask = torch.zeros(B, max_num_chars, dtype=torch.bool, device=device)

    for b in range(B):
        idx_list = char_points_idx[b]
        total_len = int(total_lengths[b])
        num_chars = len(idx_list)
        num_prefix = max(1, round(num_chars * prefix_ratio))

        if num_prefix >= num_chars:
            prefix_end = total_len
        else:
            prefix_end = idx_list[num_prefix - 1]

        prefix_end = min(prefix_end, total_len)
        full_mask[b, :prefix_end] = 1.0
        prefix_label_mask[b, :num_prefix] = 1

    latent_mask = downsample_mask(full_mask, compression_factor)
    latent_mask = 1.0 - latent_mask  # 1 = to-be-generated (suffix)

    pad_mask = downsample_mask(mask, compression_factor)  # 1 = valid

    return latent_mask, pad_mask, 1.0 - full_mask
