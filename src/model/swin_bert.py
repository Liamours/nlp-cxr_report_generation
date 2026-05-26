"""
SwinBERT: Swin Transformer encoder + 2-layer BERT decoder for CXR report generation.

Architecture:
  Image (3x224x224)
    -> SwinModel [microsoft/swin-base-patch4-window7-224]  -> (B, 49, 1024)
    -> nn.Linear(1024, 768)                                -> (B, 49, 768)  [visual_proj]
  Text (input_ids, attention_mask)
    -> BertModel [2 layers, is_decoder=True, cross_attn]   -> (B, L, 768)
    -> nn.Linear(768, 30522)                               -> (B, L, vocab) [lm_head]

Training: teacher forcing, cross-entropy loss on shifted tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel, BertConfig, BertModel


class SwinBertReportGenerator(nn.Module):
    def __init__(
        self,
        swin_name: str = "microsoft/swin-base-patch4-window7-224",
        bert_name: str = "bert-base-uncased",
        num_decoder_layers: int = 2,
        freeze_vision: bool = True,
    ):
        super().__init__()

        # --- Vision encoder (Swin Transformer) ---
        self.vision_encoder = SwinModel.from_pretrained(swin_name)
        swin_hidden = self.vision_encoder.config.hidden_size  # 1024 for swin-base

        # --- Text decoder (2-layer BERT, decoder mode with cross-attention) ---
        decoder_cfg = BertConfig.from_pretrained(bert_name)
        decoder_cfg.num_hidden_layers = num_decoder_layers
        decoder_cfg.is_decoder = True
        decoder_cfg.add_cross_attention = True

        bert_hidden = decoder_cfg.hidden_size  # 768

        # Project Swin patch features to BERT dim
        self.visual_proj = nn.Linear(swin_hidden, bert_hidden)

        # 2-layer BERT decoder (random init — cross-attn layers don't exist in pretrained BERT)
        self.text_decoder = BertModel(decoder_cfg)

        # Language model head
        self.lm_head = nn.Linear(bert_hidden, decoder_cfg.vocab_size, bias=False)

        self._vocab_size = decoder_cfg.vocab_size

        if freeze_vision:
            self.freeze_vision_encoder()

    # ---- helpers -------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def freeze_vision_encoder(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def unfreeze_vision_encoder(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = True

    # ---- forward -------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected visual patch features: (B, 49, 768)."""
        out = self.vision_encoder(pixel_values)
        return self.visual_proj(out.last_hidden_state)

    def forward(
        self,
        pixel_values: torch.Tensor,       # (B, 3, 224, 224)
        input_ids: torch.Tensor,           # (B, L)
        attention_mask: torch.Tensor,      # (B, L)
        labels: torch.Tensor | None = None,  # (B, L), -100 for ignored
    ) -> dict:
        visual_feats = self.encode_image(pixel_values)  # (B, 49, 768)

        decoder_out = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_feats,
        )

        logits = self.lm_head(decoder_out.last_hidden_state)  # (B, L, vocab)

        result = {"logits": logits}

        if labels is not None:
            # Shift right: position i predicts token i+1
            shift_logits = logits[:, :-1].contiguous()          # (B, L-1, vocab)
            shift_labels = labels[:, 1:].contiguous()           # (B, L-1)
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> list[str]:
        """Greedy decode captions for a batch of images."""
        self.eval()
        B = pixel_values.size(0)
        visual_feats = self.encode_image(pixel_values.to(device))

        # Start with [CLS] token
        input_ids = torch.full((B, 1), tokenizer.cls_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            attn = torch.ones_like(input_ids)
            dec_out = self.text_decoder(
                input_ids=input_ids,
                attention_mask=attn,
                encoder_hidden_states=visual_feats,
            )
            logits = self.lm_head(dec_out.last_hidden_state[:, -1, :])  # (B, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)             # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if all sequences have produced [SEP]
            if (next_token == tokenizer.sep_token_id).all():
                break

        sequences = input_ids[:, 1:].tolist()  # strip [CLS]
        return [
            tokenizer.decode(seq, skip_special_tokens=True)
            for seq in sequences
        ]
