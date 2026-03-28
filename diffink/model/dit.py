"""
DiffInk DiT — Glyph- and Style-Aware Latent Diffusion Transformer.

ein notation:
  b  - batch
  n  - sequence length (latent)
  nt - text sequence length
  d  - feature dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from ..utils.utils import ModelConfig
from .modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds, text_dim)
        self.mask_padding = mask_padding

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 2000
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):
        text = text + 1  # shift: 0 is filler token
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text.long())

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long, device=text.device)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand_as(text), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand_as(text), 0.0)
            else:
                text = self.text_blocks(text)

        return text


class InputEmbedding(nn.Module):
    def __init__(self, latent_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, noise, text_embed, drop_cond=False):
        if drop_cond:
            x = noise
        x = self.proj(torch.cat((x, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.dim
        latent_dim = config.latent_dim
        num_text_embedding = config.num_text_embedding
        text_dim = config.text_dim
        text_mask_padding = config.text_mask_padding
        conv_layers = config.conv_layers
        dim_head = config.dim_head
        depth = config.depth
        heads = config.heads
        ff_mult = config.ff_mult
        dropout = config.dropout
        long_skip_connection = config.long_skip_connection

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds=num_text_embedding,
            text_dim=text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.input_embed = InputEmbedding(latent_dim=latent_dim, text_dim=text_dim, out_dim=dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, noise, text, time, mask=None, drop_text=False, drop_cond=False):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, noise, text_embed, drop_cond=drop_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
