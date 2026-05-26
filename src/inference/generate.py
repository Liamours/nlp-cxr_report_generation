"""
Inference: generate a CXR report from one or more images.

Usage:
    from src.inference.generate import ReportGenerator
    gen = ReportGenerator(
        checkpoint="result/model_finetuned/epoch_000/model.pt",
        config="configs/default.yaml",   # or TrainConfig instance
    )
    report = gen.generate("path/to/image.jpg")
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model.captioner import ImageCaptioner
from src.model.decoders import build_tokenizer
from src.train.config import TrainConfig

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


class ReportGenerator:
    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path | TrainConfig | None = None,
        device: str | torch.device | None = None,
    ):
        # --- config ---
        if isinstance(config, TrainConfig):
            cfg = config
        elif config is not None:
            cfg = TrainConfig.from_yaml(config)
        else:
            cfg = TrainConfig()

        self.cfg = cfg
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- tokenizer (decoder-dependent) ---
        self.tokenizer = build_tokenizer(cfg.decoder_type)

        # --- model ---
        self.model = ImageCaptioner(
            encoder_type=cfg.encoder_type,
            decoder_type=cfg.decoder_type,
            num_decoder_layers=cfg.num_decoder_layers,
            freeze_vision=False,
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    def _load_image(self, path: str | Path) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate(self, image_path: str | Path, max_new_tokens: int = 128) -> str:
        pixel_values = self._load_image(image_path)
        reports = self.model.generate(
            pixel_values, self.tokenizer, max_new_tokens=max_new_tokens
        )
        return reports[0]

    @torch.no_grad()
    def generate_batch(
        self,
        image_paths: list[str | Path],
        max_new_tokens: int = 128,
    ) -> list[str]:
        return [self.generate(p, max_new_tokens=max_new_tokens) for p in image_paths]
