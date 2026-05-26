"""
Visual encoder registry.

Supported encoders:
  swin_base   microsoft/swin-base-patch4-window7-224   out_dim=1024  patches=49
  swin_large  microsoft/swin-large-patch4-window7-224  out_dim=1536  patches=49
  vit_base    google/vit-base-patch16-224              out_dim=768   patches=196
  vit_large   google/vit-large-patch16-224             out_dim=1024  patches=196
  dinov2_base facebook/dinov2-base                     out_dim=768   patches=256
  dinov2_large facebook/dinov2-large                   out_dim=1024  patches=256

All encoders return (B, N_patches, out_dim).
CLS tokens (ViT, DINOv2) are stripped before returning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENCODER_REGISTRY: dict[str, dict] = {
    "swin_base": {
        "hf_name": "microsoft/swin-base-patch4-window7-224",
        "family":  "swin",
        "out_dim": 1024,
        "skip_cls": False,
    },
    "swin_large": {
        "hf_name": "microsoft/swin-large-patch4-window7-224",
        "family":  "swin",
        "out_dim": 1536,
        "skip_cls": False,
    },
    "vit_base": {
        "hf_name": "google/vit-base-patch16-224",
        "family":  "vit",
        "out_dim": 768,
        "skip_cls": True,
    },
    "vit_large": {
        "hf_name": "google/vit-large-patch16-224",
        "family":  "vit",
        "out_dim": 1024,
        "skip_cls": True,
    },
    "dinov2_base": {
        "hf_name": "facebook/dinov2-base",
        "family":  "dinov2",
        "out_dim": 768,
        "skip_cls": True,
    },
    "dinov2_large": {
        "hf_name": "facebook/dinov2-large",
        "family":  "dinov2",
        "out_dim": 1024,
        "skip_cls": True,
    },
}


def list_encoders() -> list[str]:
    return list(ENCODER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Wraps any registered HF vision backbone; normalises output to (B, N, out_dim)."""

    def __init__(self, encoder_type: str):
        super().__init__()
        if encoder_type not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{encoder_type}'. "
                f"Choose from: {list_encoders()}"
            )
        spec = ENCODER_REGISTRY[encoder_type]
        self.out_dim   = spec["out_dim"]
        self._skip_cls = spec["skip_cls"]

        family  = spec["family"]
        hf_name = spec["hf_name"]

        if family == "swin":
            from transformers import SwinModel
            self.backbone = SwinModel.from_pretrained(hf_name)
        elif family == "vit":
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(hf_name)
        elif family == "dinov2":
            from transformers import Dinov2Model
            self.backbone = Dinov2Model.from_pretrained(hf_name)
        else:
            raise ValueError(f"Unsupported encoder family: {family}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch features: (B, N_patches, out_dim)."""
        feats = self.backbone(pixel_values).last_hidden_state
        if self._skip_cls:
            feats = feats[:, 1:, :]  # drop CLS at index 0
        return feats

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
