"""
Evaluate a trained checkpoint on the validation set.

Usage:
    uv run python src/script/run_evaluate.py \
        --checkpoint result/swin_bert_chexpertplus-2605261127/epoch_002/model.pt \
        --config configs/swin_bert_chexpertplus-2605261127.yaml \
        --dataset mimiccxr \
        [--metrics nlg,chexbert,bertscore] \
        [--max-samples 1000] \
        [--out result/eval_swin_bert_chexpertplus_to_mimic-2605261127.json]

Metric groups:
    nlg        BLEU-1/2/4, ROUGE-L, METEOR, CIDEr   (Rust-accelerated; Python fallback)
    chexbert   CheXbert-approx micro/macro F1        (Rust-accelerated; Python fallback)
    bertscore  BERTScore P/R/F1                      (slow, downloads BERT)

Default: nlg,chexbert

Cross-dataset generalization:
    Train on CheXpert+ → evaluate on MIMIC-CXR validate via --dataset mimiccxr.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.chexpertplus.dataset import ChexpertPlusDataset
from src.data.mimiccxr.dataset import MimicCxrDataset
from src.evaluate.metrics import compute_metrics
from src.model.captioner import ImageCaptioner
from src.model.decoders import build_tokenizer
from src.train.config import TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default=None,
                   help="YAML config used for training")
    p.add_argument("--dataset",    default=None,
                   choices=["mimiccxr", "chexpertplus"],
                   help="Override dataset from config")
    p.add_argument("--metrics",    default="nlg,chexbert",
                   help="Comma-separated metric groups: nlg,chexbert,bertscore")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap number of val samples (useful for quick checks)")
    p.add_argument("--out",        default="result/eval_results.json")
    return p.parse_args()


def _print_results(scores: dict) -> None:
    """Pretty-print metric results grouped by type."""
    nlg_keys      = {"bleu_1", "bleu_2", "bleu_4", "rouge_l", "meteor", "cider"}
    bertscore_keys = {"bertscore_p", "bertscore_r", "bertscore_f1"}
    chexbert_keys  = {"chexbert_micro_f1", "chexbert_macro_f1"}

    def _section(title: str, keys: set[str]) -> None:
        present = {k: v for k, v in scores.items() if k in keys}
        if not present:
            return
        print(f"\n  [{title}]")
        for k, v in present.items():
            if isinstance(v, float):
                print(f"    {k:<22}: {v:6.2f}")

    _section("NLG metrics",  nlg_keys)
    _section("BERTScore",    bertscore_keys)
    _section("CheXbert F1",  chexbert_keys)

    # Per-label CheXbert breakdown
    per_label = scores.get("chexbert_per_label")
    if per_label:
        print("\n  [CheXbert per-label F1]")
        rows = sorted(per_label.items(), key=lambda x: -x[1])
        for label, f1 in rows:
            bar = "#" * int(f1 / 5)
            print(f"    {label:<30}: {f1:5.1f}  {bar}")


def main():
    args = parse_args()
    groups = frozenset(g.strip() for g in args.metrics.split(",") if g.strip())

    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    if args.dataset:
        cfg.dataset = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | encoder={cfg.encoder_type}  decoder={cfg.decoder_type}")
    print(f"Metric groups: {sorted(groups)}")

    tokenizer = build_tokenizer(cfg.decoder_type)

    # --- val dataset ---
    ds_kwargs = dict(tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, image_size=cfg.image_size)
    if cfg.dataset in ("mimiccxr", "combined"):
        val_ds = MimicCxrDataset(split="validate", **ds_kwargs)
    else:
        val_ds = ChexpertPlusDataset(split="valid", **ds_kwargs)

    if args.max_samples:
        val_ds = Subset(val_ds, list(range(min(args.max_samples, len(val_ds)))))
    print(f"Val samples: {len(val_ds)}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- model ---
    model = ImageCaptioner(
        encoder_type=cfg.encoder_type,
        decoder_type=cfg.decoder_type,
        num_decoder_layers=cfg.num_decoder_layers,
        freeze_vision=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # --- generate ---
    hypotheses: list[str] = []
    references: list[str] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating"):
            pixel_values = batch["pixel_values"].to(device)
            for seq in batch["labels"]:
                ids = [t for t in seq.tolist() if t != -100]
                references.append(tokenizer.decode(ids, skip_special_tokens=True))
            preds = model.generate(pixel_values, tokenizer, max_new_tokens=128)
            hypotheses.extend(preds)

    # --- compute metrics ---
    print("\nComputing metrics...")
    scores = compute_metrics(hypotheses, references, groups=groups)

    print("\n=== Evaluation Results ===")
    _print_results(scores)

    # --- save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten per-label into top-level for JSON readability
    flat_scores = {k: v for k, v in scores.items() if not isinstance(v, dict)}
    per_label   = scores.get("chexbert_per_label", {})
    for lbl, f1 in per_label.items():
        flat_scores[f"chexbert_{lbl}"] = f1

    payload = {
        "checkpoint": str(args.checkpoint),
        "config":     str(args.config),
        "dataset":    cfg.dataset,
        "encoder":    cfg.encoder_type,
        "decoder":    cfg.decoder_type,
        "n_samples":  len(hypotheses),
        "metrics":    flat_scores,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
