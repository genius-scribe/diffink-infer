import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Encoder, Decoder, TransformerDecoder
from .ocr import ChineseHandwritingOCR
from .writer import WriterStyleClassifier
from ..utils.utils import ModelConfig


class VAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(in_channels=config.in_channels, hidden_dims=config.hidden_dims)
        self.conv_mu = nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=1)

        self.decoder = Decoder(hidden_dims=config.decoder_dims)
        self.transformer_decoder = TransformerDecoder(
            input_dim=config.decoder_dims[-1],
            hidden_dim=config.trans_hidden_dim,
            output_dim=config.decoder_output_dim,
            num_layers=config.trans_num_layers,
            num_heads=config.trans_num_heads,
        )
        self.ocr_model = ChineseHandwritingOCR(
            input_dim=config.latent_dim,
            hidden_dim=config.ocr_hidden_dim,
            num_heads=config.ocr_num_heads,
            num_layers=config.ocr_num_layers,
            num_classes=config.num_text_embedding,
        )
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        self.style_classifier = WriterStyleClassifier(
            input_dim=config.style_classifier_dim,
            num_writers=config.num_writer,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        features = self.encoder(x)
        mu = self.conv_mu(features)
        logvar = self.conv_logvar(features)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, x):
        decoded = self.decoder(x)
        output = self.transformer_decoder(decoded)
        return output

    @torch.no_grad()
    def val(self, data):
        z, mu, logvar = self.encode(data)
        decoded = self.decoder(z)
        output = self.transformer_decoder(decoded)
        return output
