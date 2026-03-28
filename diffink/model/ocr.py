import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class ChineseHandwritingOCR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(hidden_dim, num_classes)
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        self._init_bias()

    def _init_bias(self):
        with torch.no_grad():
            self.output_fc.bias.data.zero_()
            self.output_fc.bias[0].copy_(torch.tensor(-5.0))

    def forward(self, x):
        """x: [B, C, T] → [T, B, num_classes]"""
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_fc(x)
        return x.permute(1, 0, 2)
