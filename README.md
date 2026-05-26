# CXR Report Generation

Chest X-ray radiology report generation using a vision encoder + text decoder architecture.
Trained on CheXpert+ and evaluated on MIMIC-CXR for cross-dataset generalization.

---

## Architecture

**Current model: Swin-Base + BERT**

```
Image (224x224)
    └─ Vision Encoder  →  (B, 49, 1024) patch features
           │
    Linear Projection  →  (B, 49, 768)
           │
    Text Decoder (BERT cross-attention layers)
           │
    Generated Report
```

The architecture is pluggable — encoder and decoder are selected via config:

| Role    | Current        | Other available                                      |
|---------|----------------|------------------------------------------------------|
| Encoder | `swin_base`    | `swin_large`, `vit_base`, `vit_large`, `dinov2_base`, `dinov2_large`, ... |
| Decoder | `bert`         | `clinicalbert`, `biobert`, `gpt2`, `gpt2_medium`, ... |

Switch by editing the config file:
```yaml
model:
  encoder: swin_base
  decoder: bert
```

---

## Project Structure

```
src/
  data/
    chexpertplus/dataset.py   CheXpert+ dataset loader
    mimiccxr/dataset.py       MIMIC-CXR dataset loader
  model/
    encoders.py               Vision encoder registry (Swin, ViT, DINOv2)
    decoders.py               Text decoder registry + tokenizer factory
    captioner.py              ImageCaptioner — unified encoder-decoder model
  train/
    config.py                 TrainConfig dataclass, YAML loader
    trainer.py                Training loop (bf16, grad accum, freeze schedule)
  evaluate/
    metrics.py                BLEU, ROUGE-L, CheXbert F1, BERTScore
    chexpert_labeler.py       Pure-Python CheXbert fallback
  preprocess/
    chexpertplus.py           CheXpert+ CSV → preprocessed metadata
    mimiccxr.py               MIMIC-CXR CSV → preprocessed metadata
  inference/
    generate.py               ReportGenerator — load checkpoint, run greedy decode
  script/
    run_train.py              Training entry point
    run_evaluate.py           Evaluation entry point
    run_preprocess.py         Preprocessing entry point
    run_analysis.py           Error analysis entry point
  util/
    count.py                  Dataset statistics and label distribution

configs/                      YAML experiment configs
nlp_cxr_rs/                   Rust metrics extension (PyO3)
result/                       Training outputs (gitignored)
```

---

## Evaluation — Rust-Accelerated Metrics

NLG and clinical metrics are implemented in Rust via [PyO3](https://pyo3.rs) for ~10x speed improvement over pure Python.

| Metric group | Metrics | Backend |
|---|---|---|
| `nlg` | BLEU-1/2/4, ROUGE-L, METEOR, CIDEr | Rust (BLEU, ROUGE-L) + Python (METEOR, CIDEr) |
| `chexbert` | Micro F1, Macro F1, per-label F1 (14 labels) | Rust |
| `bertscore` | Precision, Recall, F1 | Python (bert-score package) |

Python fallbacks activate automatically if the Rust module is not built.

The Rust module lives in `nlp_cxr_rs/` and must be compiled once (see setup below).

---

## Datasets

Datasets are **not included** in this repository. Set environment variables pointing to your local copies:

```powershell
# Windows PowerShell
$env:CHEXPERT_ROOT = "D:\datasets\chexpertplus"
$env:MIMIC_ROOT    = "D:\datasets\mimiccxr"
```

```bash
# Linux / macOS
export CHEXPERT_ROOT="/data/chexpertplus"
export MIMIC_ROOT="/data/mimiccxr"
```

If not set, the code falls back to `data/chexpertplus` and `data/mimiccxr` relative to the project root.

Expected directory layout:
```
<CHEXPERT_ROOT>/
  label/
    df_chexpert_plus_240401.csv
    impression_fixed.json
  preprocessed/
    metadata_train.csv
    metadata_valid.csv
  CheXpert-v1.0/          (images)

<MIMIC_ROOT>/
  label/
    mimic_cxr_aug_train.csv
    mimic_cxr_aug_validate.csv
  preprocessed/
    metadata_train.csv
    metadata_valid.csv
  files/                  (images, p10/p10.../s.../...)
```

Run preprocessing once if `preprocessed/` does not exist:
```powershell
uv run python src/script/run_preprocess.py
```

---

## Setup

### Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX 4050, CUDA 12.4)
- [Rust](https://rustup.rs) (for the metrics extension)
- [uv](https://github.com/astral-sh/uv) package manager

### Install

```powershell
git clone https://github.com/Liamours/nlp-cxr_report_generation.git
cd nlp-cxr_report_generation

# Install Python dependencies (includes PyTorch cu124)
uv sync

# Build Rust metrics module
uv run maturin develop --release --manifest-path nlp_cxr_rs/Cargo.toml
```

---

## Training

Training is controlled entirely by a YAML config file. CLI flags override individual values.

```powershell
# Train with a config
uv run python src/script/run_train.py --config configs/swin_bert_chexpertplus-2605261127.yaml

# Override specific values
uv run python src/script/run_train.py --config configs/swin_bert_chexpertplus-2605261127.yaml --epochs 5 --batch-size 4

# Resume from checkpoint
uv run python src/script/run_train.py --config configs/swin_bert_chexpertplus-2605261127.yaml --resume result/swin_bert_chexpertplus-2605261127/epoch_001/model.pt
```

Key config options:

```yaml
model:
  encoder: swin_base
  decoder: bert
  num_decoder_layers: 2
  freeze_vision_epochs: 1     # freeze encoder for N epochs, then unfreeze

dataset:
  use: chexpertplus           # chexpertplus | mimiccxr | combined

training:
  epochs: 2
  batch_size: 8
  grad_accum_steps: 4         # effective batch = batch_size * grad_accum_steps
  bf16: true
```

---

## Evaluation

Evaluate a checkpoint against a dataset split. Supports cross-dataset generalization:
train on CheXpert+, test on MIMIC-CXR validate.

```powershell
uv run python src/script/run_evaluate.py `
  --checkpoint result/swin_bert_chexpertplus-2605261127/epoch_000/model.pt `
  --config     configs/swin_bert_chexpertplus-2605261127.yaml `
  --dataset    mimiccxr `
  --metrics    nlg,chexbert `
  --out        result/eval_swin_bert_chexpertplus_to_mimic-2605261127.json
```

Output example:
```
  [NLG metrics]
    bleu_1                :  32.50
    bleu_4                :   8.12
    rouge_l               :  28.74
    meteor                :  19.30

  [CheXbert F1]
    chexbert_micro_f1     :  61.20
    chexbert_macro_f1     :  44.87

  [CheXbert per-label F1]
    pleural_effusion      :  78.2  ###############
    atelectasis           :  65.1  #############
    ...
```

---

## Configs

| File | Description |
|------|-------------|
| `configs/default.yaml` | Template with all options documented |
| `configs/swin_bert_chexpertplus-2605261127.yaml` | 2-epoch CheXpert+ training, cross-dataset eval on MIMIC |
| `configs/swin_clinicalbert_combined.yaml` | Swin + ClinicalBERT, combined dataset |
| `configs/vit_gpt2_mimic.yaml` | ViT + GPT-2, MIMIC-CXR only |
