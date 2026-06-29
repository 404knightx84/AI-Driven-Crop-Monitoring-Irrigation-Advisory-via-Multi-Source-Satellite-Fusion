"""
models/crop_classifier/unet_lstm.py
Temporal U-Net: per-timestep U-Net encoder feeds an LSTM to capture
seasonal phenological dynamics, then a U-Net decoder produces the
final crop-type segmentation map.

Classes (default 8):
  0: background  1: wheat   2: rice    3: maize
  4: sugarcane   5: cotton  6: soybean 7: other
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List


CROP_CLASSES = [
    "background", "wheat", "rice", "maize",
    "sugarcane", "cotton", "soybean", "other",
]


class TemporalEncoder(nn.Module):
    """
    Wraps a shared U-Net encoder and applies it across T time steps.
    Returns a sequence of bottleneck feature maps for the LSTM.
    """

    def __init__(self, in_channels: int, encoder_name: str = "resnet34"):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=4,
            weights=None,
        )
        self.feature_dim = self.encoder.out_channels[-1]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, C, H, W)
        returns: (B, T, feature_dim, H//16, W//16)
        """
        B, T, C, H, W = x_seq.shape
        features = []
        for t in range(T):
            enc_feats = self.encoder(x_seq[:, t])   # list of feature maps
            features.append(enc_feats[-1])           # take deepest level
        return torch.stack(features, dim=1)          # (B, T, D, h, w)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell for spatial temporal fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial: tuple, device):
        h = torch.zeros(batch_size, self.hidden_dim, *spatial, device=device)
        return h, h.clone()


class CropClassifier(nn.Module):
    """
    Full model: TemporalEncoder → ConvLSTM → U-Net decoder → segmentation head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 8,
        encoder_name: str = "resnet34",
        lstm_hidden: int = 256,
        sequence_length: int = 6,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        self.temporal_encoder = TemporalEncoder(in_channels, encoder_name)
        feat_dim = self.temporal_encoder.feature_dim

        self.conv_lstm = ConvLSTMCell(feat_dim, lstm_hidden)

        # Full U-Net for final segmentation (uses last LSTM hidden state as bottleneck)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=n_classes,
        )
        # Replace bottleneck with LSTM output projection
        self.bottleneck_proj = nn.Conv2d(lstm_hidden, feat_dim, kernel_size=1)
        self.head = nn.Conv2d(16, n_classes, kernel_size=1)  # final logits

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, C, H, W)
        returns: (B, n_classes, H, W)  — raw logits
        """
        B, T, C, H, W = x_seq.shape
        enc_seq = self.temporal_encoder(x_seq)       # (B, T, D, h, w)
        _, _, D, sh, sw = enc_seq.shape

        h, c = self.conv_lstm.init_hidden(B, (sh, sw), x_seq.device)
        for t in range(T):
            h, c = self.conv_lstm(enc_seq[:, t], h, c)

        # Decode using last time-step image + LSTM context
        out = self.unet(x_seq[:, -1])                # (B, n_classes, H, W)
        return out


def build_model(cfg: dict) -> CropClassifier:
    return CropClassifier(
        in_channels=cfg.get("input_channels", 12),
        n_classes=cfg.get("n_classes", 8),
        encoder_name=cfg.get("backbone", "resnet34"),
        sequence_length=cfg.get("sequence_length", 6),
    )
