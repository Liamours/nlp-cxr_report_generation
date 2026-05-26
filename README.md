# CXR Report Generation

Chest X-ray radiology report generation using a vision encoder + text decoder architecture.
Trained on CheXpert+ and evaluated on MIMIC-CXR for cross-dataset generalization.

---

## Implementations

### Implementation 1 — Swin-Base + BERT

```
Image (224x224)
    └─ Swin-Base Encoder  →  (B, 49, 1024) patch features
              │
       Linear Projection  →  (B, 49, 768)
              │
       BERT Decoder (cross-attention layers)
              │
       Generated Report
```

Encoder frozen for the first N epochs, then unfrozen at a lower learning rate.
Config: `configs/swin_bert_chexpertplus-2605261127.yaml`

---

### Implementation 2 — _(planned)_

> ...

---

### Implementation 3 — _(planned)_

> ...

---

### Implementation 4 — _(planned)_

> ...

---

## Evaluation

Models are evaluated on three metric groups:

- **NLG** — BLEU-1/2/4, ROUGE-L, METEOR, CIDEr
- **CheXbert** — rule-based F1 across 14 CheXpert conditions (micro, macro, per-label)
- **BERTScore** — contextual embedding similarity (P/R/F1)

Default groups used during evaluation: `nlg,chexbert`.

Cross-dataset generalization: train on CheXpert+, evaluate on MIMIC-CXR validate (1,095 studies).

```powershell
uv run python src/script/run_evaluate.py `
  --checkpoint result/swin_bert_chexpertplus-2605261127/epoch_000/model.pt `
  --config     configs/swin_bert_chexpertplus-2605261127.yaml `
  --dataset    mimiccxr `
  --metrics    nlg,chexbert `
  --out        result/eval_swin_bert_chexpertplus_to_mimic-2605261127.json
```

---

## Setup

**Requirements:** Python 3.11+, CUDA GPU, [Rust](https://rustup.rs), [uv](https://github.com/astral-sh/uv)

```powershell
git clone https://github.com/Liamours/nlp-cxr_report_generation.git
cd nlp-cxr_report_generation

uv sync
uv run maturin develop --release --manifest-path nlp_cxr_rs/Cargo.toml
```

---

## Datasets

Not included in this repo. Point to your local copies via environment variables:

```powershell
$env:CHEXPERT_ROOT = "D:\datasets\chexpertplus"
$env:MIMIC_ROOT    = "D:\datasets\mimiccxr"
```

Falls back to `data/chexpertplus` and `data/mimiccxr` (relative to project root) if not set.

Run preprocessing once if `preprocessed/` does not exist:
```powershell
uv run python src/script/run_preprocess.py
```

---

## Training

```powershell
uv run python src/script/run_train.py --config configs/swin_bert_chexpertplus-2605261127.yaml

# resume
uv run python src/script/run_train.py --config configs/... --resume result/.../epoch_001/model.pt
```

See `configs/default.yaml` for all available options.
