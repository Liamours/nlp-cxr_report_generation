"""
Text decoder registry.

Supported decoders:
  bert          bert-base-uncased           hidden=768   vocab=30522   cross-attn
  clinicalbert  emilyalsentzer/Bio_ClinicalBERT  hidden=768  vocab=28996  cross-attn
  biobert       dmis-lab/biobert-base-cased-v1.2 hidden=768  vocab=28996  cross-attn
  gpt2          gpt2                        hidden=768   vocab=50257   cross-attn

All decoders use HuggingFace cross-attention decoder mode
(is_decoder=True / add_cross_attention=True).
num_layers overrides the pretrained layer count when provided.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DECODER_REGISTRY: dict[str, dict] = {
    "bert": {
        "hf_name":      "bert-base-uncased",
        "family":       "bert",
        "hidden_dim":   768,
        "vocab_size":   30522,
        "tokenizer":    "bert-base-uncased",
        "bos_token_id": 101,   # [CLS]
        "eos_token_id": 102,   # [SEP]
        "pad_token_id": 0,
    },
    "clinicalbert": {
        "hf_name":      "emilyalsentzer/Bio_ClinicalBERT",
        "family":       "bert",
        "hidden_dim":   768,
        "vocab_size":   28996,
        "tokenizer":    "emilyalsentzer/Bio_ClinicalBERT",
        "bos_token_id": 101,
        "eos_token_id": 102,
        "pad_token_id": 0,
    },
    "biobert": {
        "hf_name":      "dmis-lab/biobert-base-cased-v1.2",
        "family":       "bert",
        "hidden_dim":   768,
        "vocab_size":   28996,
        "tokenizer":    "dmis-lab/biobert-base-cased-v1.2",
        "bos_token_id": 101,
        "eos_token_id": 102,
        "pad_token_id": 0,
    },
    "gpt2": {
        "hf_name":      "gpt2",
        "family":       "gpt2",
        "hidden_dim":   768,
        "vocab_size":   50257,
        "tokenizer":    "gpt2",
        "bos_token_id": 50256,   # <|endoftext|>
        "eos_token_id": 50256,
        "pad_token_id": 50256,
    },
    "gpt2_medium": {
        "hf_name":      "gpt2-medium",
        "family":       "gpt2",
        "hidden_dim":   1024,
        "vocab_size":   50257,
        "tokenizer":    "gpt2-medium",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
    },
}


def list_decoders() -> list[str]:
    return list(DECODER_REGISTRY.keys())


def build_tokenizer(decoder_type: str):
    """Return the appropriate HF tokenizer for a decoder type."""
    spec = DECODER_REGISTRY[decoder_type]
    family = spec["family"]
    hf_name = spec["tokenizer"]

    if family == "bert":
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(hf_name)
    elif family == "gpt2":
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained(hf_name)
        tok.pad_token = tok.eos_token  # GPT-2 has no pad token by default
    else:
        raise ValueError(f"No tokenizer mapping for family: {family}")
    return tok


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TextDecoder(nn.Module):
    """Cross-attention text decoder. Works with any registered decoder type."""

    def __init__(self, decoder_type: str, num_layers: int | None = None):
        super().__init__()
        if decoder_type not in DECODER_REGISTRY:
            raise ValueError(
                f"Unknown decoder '{decoder_type}'. "
                f"Choose from: {list_decoders()}"
            )
        spec = DECODER_REGISTRY[decoder_type]
        self.hidden_dim    = spec["hidden_dim"]
        self.vocab_size    = spec["vocab_size"]
        self.bos_token_id  = spec["bos_token_id"]
        self.eos_token_id  = spec["eos_token_id"]
        self.pad_token_id  = spec["pad_token_id"]
        self._family       = spec["family"]

        hf_name = spec["hf_name"]

        if spec["family"] == "bert":
            from transformers import BertConfig, BertModel
            cfg = BertConfig.from_pretrained(hf_name)
            if num_layers is not None:
                cfg.num_hidden_layers = num_layers
            cfg.is_decoder          = True
            cfg.add_cross_attention = True
            # Load pretrained self-attention weights; cross-attention is randomly init
            self.model = BertModel.from_pretrained(
                hf_name, config=cfg, ignore_mismatched_sizes=True
            )

        elif spec["family"] == "gpt2":
            from transformers import GPT2Config, GPT2Model
            cfg = GPT2Config.from_pretrained(hf_name)
            if num_layers is not None:
                cfg.n_layer = num_layers
            cfg.add_cross_attention = True
            self.model = GPT2Model(cfg)   # random init (cross-attn not in base GPT-2)

        else:
            raise ValueError(f"Unsupported decoder family: {spec['family']}")

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        logits = self.lm_head(out.last_hidden_state)   # (B, L, vocab)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return result

    # ---- greedy generation -------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
    ) -> list[str]:
        device = encoder_hidden_states.device
        B = encoder_hidden_states.size(0)

        input_ids = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_new_tokens):
            attn = torch.ones_like(input_ids)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                encoder_hidden_states=encoder_hidden_states,
            )
            next_token = self.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1, keepdim=True)
            input_ids  = torch.cat([input_ids, next_token], dim=1)
            if (next_token == self.eos_token_id).all():
                break

        sequences = input_ids[:, 1:].tolist()   # strip BOS
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
