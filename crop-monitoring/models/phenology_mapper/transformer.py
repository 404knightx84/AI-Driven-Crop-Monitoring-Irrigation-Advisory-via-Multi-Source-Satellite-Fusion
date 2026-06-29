"""
models/phenology_mapper/transformer.py
Transformer encoder over a MODIS NDVI/EVI time-series to classify
the current phenological growth stage of each pixel.

Stages (default 6):
  0: pre-sowing    1: germination   2: vegetative
  3: reproductive  4: maturation    5: harvest/fallow
"""

import math
import torch
import torch.nn as nn
from typing import Optional


PHENOLOGY_STAGES = [
    "pre_sowing", "germination", "vegetative",
    "reproductive", "maturation", "harvest_fallow",
]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time steps."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class PhenologyTransformer(nn.Module):
    """
    Input:  (B, T, 2)  — T time steps of (NDVI, EVI) per pixel
    Output: (B, n_stages)  — stage logits

    For spatial maps, flatten (H*W) into the batch dimension.
    """

    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        n_stages: int = 6,
        sequence_length: int = 23,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=sequence_length + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS token for sequence-level classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_stages),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, T, 2)
        mask: (B, T) — True for padded positions
        returns: (B, n_stages)
        """
        B = x.size(0)
        x = self.input_proj(x)                             # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)
        x   = self.pos_enc(x)

        if src_key_padding_mask is not None:
            # Prepend False (not masked) for the CLS token
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = x[:, 0]                                  # (B, d_model)
        return self.classifier(cls_out)                    # (B, n_stages)


def build_model(cfg: dict) -> PhenologyTransformer:
    return PhenologyTransformer(
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 4),
        n_stages=cfg.get("n_stages", 6),
        sequence_length=cfg.get("sequence_length", 23),
    )
