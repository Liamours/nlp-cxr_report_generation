"""
MIMIC-CXR preprocessor.

Input:
  label/mimic_cxr_aug_train.csv    — per-patient rows (64,586 rows)
  label/mimic_cxr_aug_validate.csv — per-patient rows (500 rows)

Output:
  preprocessed/metadata_train.csv
  preprocessed/metadata_valid.csv

Schema matches CheXpert+ output (one row per study):
  dataset, patient_id, study_id, split, image_paths, views,
  caption, findings, has_image, <14 label cols>

Label values: NaN (not available in aug CSV — no CheXpert labels provided)
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

import os
DATASET_ROOT = Path(os.environ.get("MIMIC_ROOT", "data/mimiccxr"))
OUT_ROOT = DATASET_ROOT / "preprocessed"


def _study_id_from_path(img_path: str) -> str:
    # files/p10/p10003502/s50084553/hash.jpg  →  s50084553
    return img_path.split("/")[3]


def _build_view_map(ap: list, pa: list, lateral: list) -> dict:
    view_map = {}
    for p in ap:
        view_map[p] = "AP"
    for p in pa:
        view_map[p] = "PA"
    for p in lateral:
        view_map[p] = "Lateral"
    return view_map


def _parse_caption_findings(text: str):
    text = text.strip()
    imp_match = re.search(r"(?i)impression\s*:", text)
    fin_match = re.search(r"(?i)findings?\s*:", text)

    if imp_match:
        caption = text[imp_match.end():].strip()
        findings = text[fin_match.end(): imp_match.start()].strip() if fin_match else None
    elif fin_match:
        caption = None
        findings = text[fin_match.end():].strip()
    else:
        caption = text if text else None
        findings = None

    return caption or None, findings or None


def _safe_eval(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def _process_csv(csv_path: Path, split: str) -> list:
    df = pd.read_csv(csv_path, low_memory=False)
    records = []

    for _, row in df.iterrows():
        patient_id = f"p{row['subject_id']}"
        all_images = _safe_eval(row["image"])
        ap_imgs = _safe_eval(row["AP"])
        pa_imgs = _safe_eval(row["PA"])
        lat_imgs = _safe_eval(row["Lateral"])
        texts = _safe_eval(row["text"])

        if not all_images:
            continue

        view_map = _build_view_map(ap_imgs, pa_imgs, lat_imgs)

        # Group images by study_id, preserve insertion order
        study_to_images: dict = {}
        for img in all_images:
            sid = _study_id_from_path(img)
            study_to_images.setdefault(sid, []).append(img)

        ordered_studies = sorted(study_to_images.keys())

        for i, study_id in enumerate(ordered_studies):
            imgs = study_to_images[study_id]
            views = [view_map.get(img, "Unknown") for img in imgs]
            text = texts[i] if i < len(texts) else ""
            caption, findings = _parse_caption_findings(text)

            record = {
                "dataset": "mimiccxr",
                "patient_id": patient_id,
                "study_id": study_id,
                "split": split,
                "image_paths": json.dumps(imgs),
                "views": json.dumps(views),
                "caption": caption,
                "findings": findings,
                "has_image": True,
            }
            for col in LABEL_COLS:
                record[col] = float("nan")

            records.append(record)

    return records


def run(dataset_root: Path = DATASET_ROOT, out_root: Path = OUT_ROOT) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {}

    splits = [
        (dataset_root / "label" / "mimic_cxr_aug_train.csv", "train"),
        (dataset_root / "label" / "mimic_cxr_aug_validate.csv", "valid"),
    ]

    for csv_path, split_name in splits:
        print(f"Processing {split_name}...")
        records = _process_csv(csv_path, split_name)
        out_df = pd.DataFrame(records)

        out_path = out_root / f"metadata_{split_name}.csv"
        out_df.to_csv(out_path, index=False)

        stats[split_name] = {
            "studies": len(out_df),
            "patients": out_df["patient_id"].nunique(),
            "images": sum(len(json.loads(p)) for p in out_df["image_paths"]),
        }
        print(f"  [{split_name}] studies={stats[split_name]['studies']:,}  "
              f"patients={stats[split_name]['patients']:,}  "
              f"images={stats[split_name]['images']:,}  -> {out_path.name}")

    return stats


if __name__ == "__main__":
    run()
