"""
models/stress_detector/fusion_cnn.py
Dual-stream CNN fusing Sentinel-2 optical indices with Sentinel-1 SAR
backscatter to detect crop moisture stress levels.

Stress classes:
  0: no_stress   1: mild_stress   2: severe_stress

Architecture:
  Optical stream (NDVI, EVI, NDWI, NDRE, SAVI, LSWI + S2 bands)
     → ResNet-style conv blocks → feature map
  SAR stream     (VV, VH, VV/VH ratio)
     → Lightweight conv blocks → feature map
  Fusion          (concat → attention gate → conv head)
     → per-pixel stress classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnRelu(ch, ch), ConvBnRelu(ch, ch)
        )

    def forward(self, x):
        return x + self.body(x)


class OpticalStream(nn.Module):
    """Processes stacked S2 bands + spectral indices."""

    def __init__(self, in_channels: int = 16):   # 10 S2 bands + 6 indices
        super().__init__()
        self.stem = ConvBnRelu(in_channels, 64, kernel=7)
        self.layers = nn.Sequential(
            ConvBnRelu(64, 128, stride=2),
            ResBlock(128),
            ConvBnRelu(128, 256, stride=2),
            ResBlock(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x))   # (B, 256, H/4, W/4)


class SARStream(nn.Module):
    """Lightweight stream for SAR backscatter channels."""

    def __init__(self, in_channels: int = 3):   # VV, VH, ratio
        super().__init__()
        self.layers = nn.Sequential(
            ConvBnRelu(in_channels, 32),
            ConvBnRelu(32, 64, stride=2),
            ResBlock(64),
            ConvBnRelu(64, 128, stride=2),
            ResBlock(128),
            ConvBnRelu(128, 256, kernel=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)             # (B, 256, H/4, W/4)


class ChannelAttention(nn.Module):
    """Squeeze-Excitation block to weight fused feature channels."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class StressDetector(nn.Module):
    """
    Dual-stream fusion CNN for pixel-wise moisture stress detection.
    Input:  optical (B, C_opt, H, W) + sar (B, C_sar, H, W)
    Output: (B, 3, H, W)  raw logits per stress class
    """

    def __init__(
        self,
        optical_channels: int = 16,
        sar_channels: int = 3,
        n_classes: int = 3,
    ):
        super().__init__()
        self.optical_stream = OpticalStream(optical_channels)
        self.sar_stream      = SARStream(sar_channels)
        self.attention       = ChannelAttention(512)

        self.fusion_head = nn.Sequential(
            ConvBnRelu(512, 256),
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(128, 64),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> torch.Tensor:
        opt_feat = self.optical_stream(optical)      # (B, 256, h, w)
        sar_feat = self.sar_stream(sar)              # (B, 256, h, w)
        fused    = torch.cat([opt_feat, sar_feat], dim=1)  # (B, 512, h, w)
        fused    = self.attention(fused)
        return self.fusion_head(fused)               # (B, n_classes, H, W)


def build_model(cfg: dict) -> StressDetector:
    return StressDetector(
        optical_channels=cfg.get("input_channels", 16) - 3,
        sar_channels=3,
        n_classes=cfg.get("n_classes", 3),
    )
