"""
ImageCaptioner: generic encoder-decoder model for image captioning.

Encoder and decoder types are selected from their registries.
The visual projector bridges the encoder's output dim to the decoder's hidden dim.

Compatible with any (encoder_type, decoder_type) combination from:
  src.model.encoders.ENCODER_REGISTRY
  src.model.decoders.DECODER_REGISTRY
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.decoders import DECODER_REGISTRY, TextDecoder
from src.model.encoders import ENCODER_REGISTRY, VisualEncoder


class ImageCaptioner(nn.Module):
    def __init__(
        self,
        encoder_type: str = "swin_base",
        decoder_type: str = "bert",
        num_decoder_layers: int = 2,
        freeze_vision: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.encoder = VisualEncoder(encoder_type)
        self.decoder = TextDecoder(decoder_type, num_layers=num_decoder_layers)

        # Project encoder patch features to decoder hidden dim
        enc_dim = ENCODER_REGISTRY[encoder_type]["out_dim"]
        dec_dim = DECODER_REGISTRY[decoder_type]["hidden_dim"]
        self.visual_proj = nn.Linear(enc_dim, dec_dim)

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)
        elif freeze_vision:
            self.encoder.freeze()

    # ---- LoRA --------------------------------------------------------------

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to encoder backbone only. Decoder trains fully (random init cross-attn)."""
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],  # Swin uses q_proj/v_proj
            bias="none",
        )
        self.encoder.backbone = get_peft_model(self.encoder.backbone, lora_cfg)
        enc_trainable = sum(p.numel() for p in self.encoder.backbone.parameters() if p.requires_grad)
        print(f"  LoRA encoder: r={r}  alpha={alpha}  trainable={enc_trainable/1e6:.2f}M")

    # ---- helpers -----------------------------------------------------------

    def freeze_vision_encoder(self):
        self.encoder.freeze()

    def unfreeze_vision_encoder(self):
        self.encoder.unfreeze()

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected patch features: (B, N_patches, decoder_hidden_dim)."""
        return self.visual_proj(self.encoder(pixel_values))

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        visual_feats = self.encode_image(pixel_values)
        return self.decoder(input_ids, attention_mask, visual_feats, labels)

    # ---- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
        device: torch.device | None = None,
    ) -> list[str]:
        if device is not None:
            pixel_values = pixel_values.to(device)
        visual_feats = self.encode_image(pixel_values)
        return self.decoder.generate(visual_feats, tokenizer, max_new_tokens)

    # ---- param count -------------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
