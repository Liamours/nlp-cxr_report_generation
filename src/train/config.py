"""Training configuration — supports both dataclass defaults and YAML file loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Section → field mapping for YAML loading
# keys: (yaml_section, yaml_key) -> dataclass field name
_YAML_MAP: dict[tuple[str, str], str] = {
    ("model", "encoder"):               "encoder_type",
    ("model", "decoder"):               "decoder_type",
    ("model", "num_decoder_layers"):    "num_decoder_layers",
    ("model", "freeze_vision_epochs"):  "freeze_vision_epochs",
    ("dataset", "use"):                 "dataset",
    ("dataset", "max_seq_len"):         "max_seq_len",
    ("dataset", "image_size"):          "image_size",
    ("dataset", "num_workers"):         "num_workers",
    ("training", "epochs"):             "epochs",
    ("training", "batch_size"):         "batch_size",
    ("training", "grad_accum_steps"):   "grad_accum_steps",
    ("training", "lr"):                 "lr",
    ("training", "vision_lr"):          "vision_lr",
    ("training", "weight_decay"):       "weight_decay",
    ("training", "warmup_steps"):       "warmup_steps",
    ("training", "max_grad_norm"):      "max_grad_norm",
    ("training", "bf16"):               "bf16",
    ("paths", "output_dir"):            "output_dir",
    ("paths", "log_dir"):               "log_dir",
    ("checkpointing", "save_every"):    "save_every",
    ("checkpointing", "keep_last"):     "keep_last",
}


@dataclass
class TrainConfig:
    # --- model ---
    encoder_type: str = "swin_base"      # see src/model/encoders.ENCODER_REGISTRY
    decoder_type: str = "bert"           # see src/model/decoders.DECODER_REGISTRY
    num_decoder_layers: int = 2
    freeze_vision_epochs: int = 3        # freeze encoder for first N epochs

    # --- dataset ---
    dataset: str = "mimiccxr"           # "mimiccxr" | "chexpertplus" | "combined"
    max_seq_len: int = 128
    image_size: int = 224
    num_workers: int = 0                 # 0 = main process only (safer on Windows)

    # --- training ---
    epochs: int = 20
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 1e-4
    vision_lr: float = 1e-5             # lower LR for encoder after unfreeze
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    bf16: bool = True

    # --- paths ---
    output_dir: Path = field(
        default_factory=lambda: Path(r"C:\Users\lulay\Desktop\nlp-image_captioning\result\model_finetuned")
    )
    log_dir: Path = field(
        default_factory=lambda: Path(r"C:\Users\lulay\Desktop\nlp-image_captioning\result\log")
    )

    # --- checkpointing ---
    save_every: int = 1
    keep_last: int = 3

    # ---- lifecycle ---------------------------------------------------------

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.log_dir    = Path(self.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---- YAML I/O ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "TrainConfig":
        """Load config from a YAML file (nested sections), then apply any overrides."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Install pyyaml: uv add pyyaml") from e

        with open(path) as f:
            raw: dict = yaml.safe_load(f) or {}

        flat: dict[str, Any] = {}
        for section, entries in raw.items():
            if not isinstance(entries, dict):
                continue
            for key, value in entries.items():
                field_name = _YAML_MAP.get((section, key))
                if field_name is None:
                    raise ValueError(
                        f"Unknown config key '{section}.{key}' in {path}. "
                        f"Valid keys: {sorted(_YAML_MAP.keys())}"
                    )
                flat[field_name] = value

        if overrides:
            flat.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**flat)

    def to_yaml(self, path: str | Path) -> None:
        """Save config back to a YAML file with nested sections."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Install pyyaml: uv add pyyaml") from e

        # Invert _YAML_MAP: field_name -> (section, yaml_key)
        field_to_yaml = {v: k for k, v in _YAML_MAP.items()}

        nested: dict[str, dict] = {}
        for f in dataclasses.fields(self):
            if f.name not in field_to_yaml:
                continue
            section, key = field_to_yaml[f.name]
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            nested.setdefault(section, {})[key] = value

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(nested, fh, sort_keys=False, allow_unicode=True)

    # ---- display -----------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"encoder={self.encoder_type}  decoder={self.decoder_type}  layers={self.num_decoder_layers}",
            f"dataset={self.dataset}  seq_len={self.max_seq_len}  img={self.image_size}",
            f"epochs={self.epochs}  bs={self.batch_size}  accum={self.grad_accum_steps}  bf16={self.bf16}",
            f"lr={self.lr}  vision_lr={self.vision_lr}  freeze_epochs={self.freeze_vision_epochs}",
            f"out={self.output_dir}",
        ]
        return "\n".join(lines)
