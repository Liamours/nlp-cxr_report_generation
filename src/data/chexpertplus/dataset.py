"""
CheXpert+ Dataset for CXR report generation.

Reads from preprocessed/metadata_{split}.csv.
Images: PNG 224x224 grayscale -> convert RGB, normalize.
Caption: impression section (cleaned).
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizerBase

import os
DATASET_ROOT = Path(os.environ.get("CHEXPERT_ROOT", "data/chexpertplus"))
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _clean_caption(text) -> str | None:
    if text is None or isinstance(text, float):
        return None
    text = str(text).strip()
    return text if len(text) >= 10 else None


def build_transform(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ]
    return transforms.Compose(ops)


def _select_primary_image(paths: list, views: list) -> str:
    view_priority = {"Frontal": 0, "Lateral": 1}
    best = None
    best_rank = 999
    for p, v in zip(paths, views):
        rank = view_priority.get(v, 2)
        if rank < best_rank:
            best_rank = rank
            best = p
    return best or paths[0]


class ChexpertPlusDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_seq_len: int = 128,
        image_size: int = 224,
        augment: bool = False,
        dataset_root: Path = DATASET_ROOT,
    ):
        self.dataset_root = dataset_root
        if tokenizer is None:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = build_transform(image_size, augment)

        csv = dataset_root / "preprocessed" / f"metadata_{split}.csv"
        df = pd.read_csv(csv, low_memory=False)

        self.records = []
        for _, row in df.iterrows():
            cap = _clean_caption(row.get("caption"))
            if cap is None:
                continue
            paths = json.loads(row["image_paths"])
            views = json.loads(row["views"])
            img_rel = _select_primary_image(paths, views)
            # Normalize path separators
            img_path = dataset_root / Path(img_rel.replace("\\", "/"))
            if not img_path.exists():
                continue
            self.records.append({"image_path": img_path, "caption": cap})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # CheXpert+ PNGs are grayscale ('L') — convert to RGB for Swin
        img = Image.open(rec["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        enc = self.tokenizer(
            rec["caption"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
