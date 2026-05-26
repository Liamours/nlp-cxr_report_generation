"""
Count images, studies, patients, and label distributions from preprocessed metadata CSVs.
Prints results alongside paper-reported expected values for verification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# Expected values from papers for cross-check
EXPECTED = {
    "chexpertplus": {
        # 223,462 = total images (Table 1 in paper); 223,228 = train-only image-report pairs
        "images": 223462, "studies": 187711, "patients": 64725,
        "images_train": 223228, "studies_train": 187511, "patients_train": 64525,
        "images_valid": 234,    "studies_valid": 200,    "patients_valid": 200,
    },
    "mimiccxr": {
        "images_train": 368960, "studies_train": 222758, "patients_train": 64586,
        "images_valid": 2991,   "studies_valid": 1808,   "patients_valid": 500,
    },
}


def count_structural(csv_paths: list[Path]) -> dict:
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    n_images = sum(len(json.loads(p)) for p in df["image_paths"])
    n_studies = len(df)
    n_patients = df["patient_id"].nunique()
    return {"images": n_images, "studies": n_studies, "patients": n_patients}


def count_labels(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        vals = df[col]
        rows.append({
            "label": col,
            "positive": int((vals == 1.0).sum()),
            "negative": int((vals == 0.0).sum()),
            "uncertain": int((vals == -1.0).sum()),
            "not_mentioned": int(vals.isna().sum()),
        })
    return pd.DataFrame(rows)


def print_report(dataset: str, preprocessed_root: Path):
    csvs = sorted(preprocessed_root.glob("metadata_*.csv"))
    if not csvs:
        print(f"  No preprocessed CSVs found in {preprocessed_root}")
        return

    struct = count_structural(csvs)
    label_df = count_labels(csvs)

    exp = EXPECTED.get(dataset, {})
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Actual':>12} {'Expected (paper)':>18}")
    print(f"  {'-'*52}")

    def row(name, actual, expected=None):
        exp_str = f"{expected:,}" if expected else "N/A"
        flag = " OK" if expected and actual == expected else (" MISMATCH" if expected else "")
        print(f"  {name:<20} {actual:>12,} {exp_str:>18}{flag}")

    row("Images (combined)", struct["images"], exp.get("images"))
    row("Studies (combined)", struct["studies"], exp.get("studies"))
    row("Patients (combined)", struct["patients"], exp.get("patients"))

    # per-split breakdown
    for csv_path in csvs:
        df = pd.read_csv(csv_path, low_memory=False)
        split_name = csv_path.stem.replace("metadata_", "")
        n_img = sum(len(json.loads(p)) for p in df["image_paths"])
        exp_img = exp.get(f"images_{split_name}")
        exp_std = exp.get(f"studies_{split_name}")
        exp_pat = exp.get(f"patients_{split_name}")
        row(f"  [{split_name}] images", n_img, exp_img)
        row(f"  [{split_name}] studies", len(df), exp_std)
        row(f"  [{split_name}] patients", df["patient_id"].nunique(), exp_pat)

    if not label_df.empty and label_df["positive"].sum() > 0:
        print(f"\n  Label distribution (per study, from preprocessed CSV):")
        print(f"  {'Label':<30} {'Pos':>8} {'Neg':>8} {'Unc':>8} {'NaN':>8}")
        print(f"  {'-'*58}")
        for _, r in label_df.iterrows():
            print(f"  {r['label']:<30} {r['positive']:>8,} {r['negative']:>8,} "
                  f"{r['uncertain']:>8,} {r['not_mentioned']:>8,}")
    else:
        print(f"\n  Labels: not available in preprocessed CSV (MIMIC aug format has no label cols)")


def run_all():
    roots = {
        "chexpertplus": Path(r"C:\Users\lulay\Desktop\research-vlm_cxr\dataset\chexpertplus\preprocessed"),
        "mimiccxr":     Path(r"C:\Users\lulay\Desktop\research-vlm_cxr\dataset\mimiccxr\preprocessed"),
    }
    for dataset, root in roots.items():
        print_report(dataset, root)


if __name__ == "__main__":
    run_all()
