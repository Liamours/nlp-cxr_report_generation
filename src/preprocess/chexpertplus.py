"""
CheXpert+ preprocessor.

Input:
  label/df_chexpert_plus_240401.csv  — per-image, 27 cols
  label/impression_fixed.json        — JSONL, per-image, 14 pathology labels

Output:
  preprocessed/metadata_train.csv
  preprocessed/metadata_valid.csv

Schema (one row per study):
  dataset, patient_id, study_id, split, image_paths, views,
  caption, findings, has_image, <14 label cols>

Label values: 1.0=positive, 0.0=negative, -1.0=uncertain, NaN=not mentioned
"""

import json
import ast
from pathlib import Path

import pandas as pd

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

import os
DATASET_ROOT = Path(os.environ.get("CHEXPERT_ROOT", "data/chexpertplus"))
OUT_ROOT = DATASET_ROOT / "preprocessed"


def _load_labels(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.rename(columns={"path_to_image": "path_to_image_label"})
    return df


def _extract_ids(path_to_image: str):
    parts = path_to_image.split("/")
    # e.g. train/patient00003/study1/view1_frontal.jpg
    patient_id = parts[1]
    study_id = f"{parts[1]}_{parts[2]}"
    return patient_id, study_id


def _png_path(path_to_image: str) -> str:
    p = Path(path_to_image)
    return str(Path("preprocessed") / "PNG" / p.parent / (p.stem + ".png"))


def run(dataset_root: Path = DATASET_ROOT, out_root: Path = OUT_ROOT) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading main CSV...")
    df = pd.read_csv(dataset_root / "label" / "df_chexpert_plus_240401.csv", low_memory=False)

    print("Loading impression labels (JSONL)...")
    labels_df = _load_labels(dataset_root / "label" / "impression_fixed.json")
    labels_df["path_to_image"] = labels_df["path_to_image_label"]

    print("Merging labels...")
    df = df.merge(
        labels_df[["path_to_image"] + LABEL_COLS],
        on="path_to_image",
        how="left",
    )

    df[["patient_id", "study_id"]] = df["path_to_image"].apply(
        lambda p: pd.Series(_extract_ids(p))
    )
    df["png_path"] = df["path_to_image"].map(_png_path)

    print("Grouping by study...")
    records = []
    for (patient_id, study_id, split), grp in df.groupby(
        ["patient_id", "study_id", "split"], sort=False
    ):
        image_paths = json.dumps(grp["png_path"].tolist())
        views = json.dumps(grp["frontal_lateral"].tolist())
        caption = grp["section_impression"].dropna().iloc[0] if grp["section_impression"].notna().any() else None
        findings = grp["section_findings"].dropna().iloc[0] if grp["section_findings"].notna().any() else None

        row = {
            "dataset": "chexpertplus",
            "patient_id": patient_id,
            "study_id": study_id,
            "split": split,
            "image_paths": image_paths,
            "views": views,
            "caption": caption,
            "findings": findings,
            "has_image": True,
        }
        for col in LABEL_COLS:
            vals = grp[col].dropna()
            row[col] = vals.iloc[0] if not vals.empty else float("nan")

        records.append(row)

    out_df = pd.DataFrame(records)

    stats = {}
    for split_name, split_df in out_df.groupby("split"):
        out_path = out_root / f"metadata_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        stats[split_name] = {
            "studies": len(split_df),
            "patients": split_df["patient_id"].nunique(),
            "images": sum(len(json.loads(p)) for p in split_df["image_paths"]),
        }
        print(f"  [{split_name}] studies={stats[split_name]['studies']:,}  "
              f"patients={stats[split_name]['patients']:,}  "
              f"images={stats[split_name]['images']:,}  -> {out_path.name}")

    return stats


if __name__ == "__main__":
    run()
