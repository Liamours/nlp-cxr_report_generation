"""
MIMIC-CXR Dataset for CXR report generation.

Reads from preprocessed/metadata_{split}.csv.
Selects primary frontal image per study (PA > AP > first).
Images: JPEG 512x512 RGB -> resize 224x224, normalize.
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
DATASET_ROOT = Path(os.environ.get("MIMIC_ROOT", "data/mimiccxr"))
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _select_primary_image(paths: list, views: list) -> str:
    view_priority = {"PA": 0, "AP": 1}
    best = None
    best_rank = 999
    for p, v in zip(paths, views):
        rank = view_priority.get(v, 2)
        if rank < best_rank:
            best_rank = rank
            best = p
    return best or paths[0]


from src.util.text_norm import normalize_report_text as _clean_caption


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


class MimicCxrDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_seq_len: int = 128,
        image_size: int = 224,
        augment: bool = False,
        dataset_root: Path = DATASET_ROOT,
    ):
        self.raw_root = dataset_root / "raw"
        if tokenizer is None:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = build_transform(image_size, augment)

        # normalise split alias: "validate" -> "valid"
        split = "valid" if split == "validate" else split
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
            # Strip 'files/' prefix — raw folder doesn't have that subdir
            img_rel = img_rel.replace("files/", "", 1)
            img_path = self.raw_root / img_rel
            if not img_path.exists():
                continue
            self.records.append({"image_path": img_path, "caption": cap})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

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
