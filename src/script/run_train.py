"""
Training orchestrator.

Usage (config file — recommended):
    uv run python src/script/run_train.py --config configs/default.yaml

Usage (CLI only — uses dataclass defaults + overrides):
    uv run python src/script/run_train.py \
        --encoder swin_base --decoder bert \
        --dataset mimiccxr --epochs 20 --batch-size 8

CLI flags override YAML values when both are provided.

Other flags:
    --resume   path/to/epoch_XXX/model.pt   resume from checkpoint
    --no-bf16  force fp32
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chexpertplus.dataset import ChexpertPlusDataset
from src.data.mimiccxr.dataset import MimicCxrDataset
from src.model.captioner import ImageCaptioner
from src.model.decoders import build_tokenizer
from src.train.config import TrainConfig
from src.train.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    # --- config file ---
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/default.yaml)")

    # --- model overrides ---
    p.add_argument("--encoder", default=None,
                   help="Encoder type (overrides config model.encoder)")
    p.add_argument("--decoder", default=None,
                   help="Decoder type (overrides config model.decoder)")
    p.add_argument("--num-decoder-layers", type=int, default=None)
    p.add_argument("--freeze-vision-epochs", type=int, default=None)

    # --- dataset overrides ---
    p.add_argument("--dataset", default=None,
                   choices=["mimiccxr", "chexpertplus", "combined"])

    # --- training overrides ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-bf16", action="store_true")

    # --- misc ---
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--save-config", default=None,
                   help="Save resolved config to this YAML path")
    return p.parse_args()


def build_overrides(args) -> dict:
    """Collect non-None CLI args into a flat override dict (matching TrainConfig fields)."""
    overrides = {}
    if args.encoder:              overrides["encoder_type"]       = args.encoder
    if args.decoder:              overrides["decoder_type"]       = args.decoder
    if args.num_decoder_layers:   overrides["num_decoder_layers"] = args.num_decoder_layers
    if args.freeze_vision_epochs: overrides["freeze_vision_epochs"] = args.freeze_vision_epochs
    if args.dataset:              overrides["dataset"]            = args.dataset
    if args.epochs:               overrides["epochs"]             = args.epochs
    if args.batch_size:           overrides["batch_size"]         = args.batch_size
    if args.lr:                   overrides["lr"]                 = args.lr
    if args.no_bf16:              overrides["bf16"]               = False
    return overrides


def build_datasets(cfg: TrainConfig, tokenizer):
    kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        image_size=cfg.image_size,
    )
    if cfg.dataset == "mimiccxr":
        train_ds = MimicCxrDataset(split="train",    augment=True,  **kwargs)
        val_ds   = MimicCxrDataset(split="validate", augment=False, **kwargs)

    elif cfg.dataset == "chexpertplus":
        train_ds = ChexpertPlusDataset(split="train", augment=True,  **kwargs)
        val_ds   = ChexpertPlusDataset(split="valid", augment=False, **kwargs)

    elif cfg.dataset == "combined":
        train_ds = ConcatDataset([
            MimicCxrDataset(split="train",    augment=True, **kwargs),
            ChexpertPlusDataset(split="train", augment=True, **kwargs),
        ])
        val_ds = MimicCxrDataset(split="validate", augment=False, **kwargs)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    return train_ds, val_ds


def main():
    args = parse_args()
    overrides = build_overrides(args)

    # --- build config ---
    if args.config:
        cfg = TrainConfig.from_yaml(args.config, overrides=overrides)
    else:
        cfg = TrainConfig(**overrides) if overrides else TrainConfig()

    print("Config:\n" + cfg.summary())

    if args.save_config:
        cfg.to_yaml(args.save_config)
        print(f"Config saved -> {args.save_config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- tokenizer (determined by decoder type) ---
    tokenizer = build_tokenizer(cfg.decoder_type)

    # --- datasets ---
    print("Building datasets...")
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # --- model ---
    print("Building model...")
    model = ImageCaptioner(
        encoder_type=cfg.encoder_type,
        decoder_type=cfg.decoder_type,
        num_decoder_layers=cfg.num_decoder_layers,
        freeze_vision=True,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Resumed from {args.resume}")

    counts = model.param_counts()
    print(f"  Total params: {counts['total']/1e6:.1f}M | Trainable: {counts['trainable']/1e6:.1f}M")

    # --- train ---
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
