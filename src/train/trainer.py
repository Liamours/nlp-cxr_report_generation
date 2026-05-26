"""
Training loop for SwinBERT CXR report generation.

Features:
  - Mixed precision (bf16) via torch.amp
  - Gradient accumulation
  - Swin encoder freeze/unfreeze schedule
  - Checkpoint save/load
  - CSV training log
"""

import csv
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.config import TrainConfig


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.bf16)

        self._build_optimizer()
        self._init_log()

    # ---- setup ---------------------------------------------------------

    def _build_optimizer(self):
        cfg = self.cfg
        # Separate param groups: vision encoder gets lower LR after unfreeze
        decoder_params = [
            p for n, p in self.model.named_parameters()
            if "vision_encoder" not in n and p.requires_grad
        ]
        self.optimizer = optim.AdamW(
            decoder_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total_steps = (
            len(self.train_loader) // cfg.grad_accum_steps * cfg.epochs
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_steps / max(total_steps, 1),
            anneal_strategy="cos",
        )
        self.vision_optimizer = None  # created when Swin is unfrozen

    def _unfreeze_vision(self):
        self.model.unfreeze_vision_encoder()
        vision_params = [
            p for p in self.model.vision_encoder.parameters()
        ]
        self.vision_optimizer = optim.AdamW(
            vision_params, lr=self.cfg.vision_lr,
            weight_decay=self.cfg.weight_decay,
        )
        print("  [trainer] Swin encoder unfrozen.")

    def _init_log(self):
        self.log_path = self.cfg.log_dir / "train_log.csv"
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_s"])

    # ---- train/eval loops ----------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        if self.vision_optimizer:
            self.vision_optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False)
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            dtype = torch.bfloat16 if self.cfg.bf16 else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.bf16):
                out = self.model(pixel_values, input_ids, attention_mask, labels)
                loss = out["loss"] / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                if self.vision_optimizer:
                    self.scaler.step(self.vision_optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.vision_optimizer:
                    self.vision_optimizer.zero_grad()

            total_loss += loss.item() * self.cfg.grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * self.cfg.grad_accum_steps:.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            dtype = torch.bfloat16 if self.cfg.bf16 else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.cfg.bf16):
                out = self.model(pixel_values, input_ids, attention_mask, labels)
            total_loss += out["loss"].item()

        return total_loss / len(self.val_loader)

    # ---- checkpoint ----------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float):
        ckpt_dir = self.cfg.output_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        meta = {"epoch": epoch, "val_loss": val_loss}
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Prune old checkpoints
        all_ckpts = sorted(self.cfg.output_dir.glob("epoch_*"))
        for old in all_ckpts[: -self.cfg.keep_last]:
            for fp in old.iterdir():
                fp.unlink()
            old.rmdir()

    # ---- main entry ----------------------------------------------------

    def train(self):
        cfg = self.cfg
        best_val = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # Unfreeze Swin after freeze_vision_epochs
            if epoch == cfg.freeze_vision_epochs + 1:
                self._unfreeze_vision()

            train_loss = self._train_epoch(epoch)
            val_loss = self._eval_epoch()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"{elapsed:.0f}s"
            )

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, f"{elapsed:.1f}"])

            if epoch % cfg.save_every == 0:
                self._save_checkpoint(epoch, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(0, val_loss)  # epoch 000 = best

        print(f"\nTraining done. Best val loss: {best_val:.4f}")
