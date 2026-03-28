import torch
import torch.nn as nn


class WriterStyleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_writers: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_writers),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor = None):
        """feat: [B, C, T]; mask: [B, T] with 1=valid, 0=pad → logits [B, num_writers]"""
        x = feat
        if mask is not None:
            mask = mask.to(dtype=feat.dtype)
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
            avg_pool = (x * mask.unsqueeze(1)).sum(dim=2) / mask_sum
            mean = avg_pool.unsqueeze(2)
            var = ((x - mean) ** 2 * mask.unsqueeze(1)).sum(dim=2) / mask_sum
            std_pool = var.sqrt()
            masked_feat = feat.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
            max_pool = masked_feat.max(dim=2).values
        else:
            avg_pool = x.mean(dim=2)
            std_pool = x.std(dim=2, unbiased=False)
            max_pool = x.max(dim=2).values

        avg_pool = self.norm(avg_pool)
        std_pool = self.norm(std_pool)
        max_pool = self.norm(max_pool)

        pooled = torch.cat([avg_pool, max_pool, std_pool], dim=-1)
        pooled = self.dropout(pooled)
        return self.mlp(pooled)
