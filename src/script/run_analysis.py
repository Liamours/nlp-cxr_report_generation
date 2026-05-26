"""
Error analysis on the validation set.

Usage:
    uv run python src/script/run_analysis.py \
        --checkpoint result/model_finetuned/epoch_000/model.pt \
        --config configs/default.yaml \
        [--max-samples 500] [--out result/analysis.json]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.analysis.error_analysis import run_full_analysis
from src.data.chexpertplus.dataset import ChexpertPlusDataset
from src.data.mimiccxr.dataset import MimicCxrDataset
from src.model.captioner import ImageCaptioner
from src.model.decoders import build_tokenizer
from src.train.config import TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--out", default="result/analysis.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(cfg.decoder_type)

    ds_kwargs = dict(tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, image_size=cfg.image_size)
    if cfg.dataset in ("mimiccxr", "combined"):
        ds = MimicCxrDataset(split="validate", **ds_kwargs)
    else:
        ds = ChexpertPlusDataset(split="valid", **ds_kwargs)

    if args.max_samples:
        ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = ImageCaptioner(
        encoder_type=cfg.encoder_type,
        decoder_type=cfg.decoder_type,
        num_decoder_layers=cfg.num_decoder_layers,
        freeze_vision=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    hypotheses, references = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            pixel_values = batch["pixel_values"].to(device)
            for seq in batch["labels"]:
                ids = [t for t in seq.tolist() if t != -100]
                references.append(tokenizer.decode(ids, skip_special_tokens=True))
            preds = model.generate(pixel_values, tokenizer, max_new_tokens=128)
            hypotheses.extend(preds)

    report = run_full_analysis(hypotheses, references)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved -> {out_path}")

    dist = report["distribution"]
    print(f"\nROUGE-L  mean={dist['mean']}  median={dist['median']}  std={dist['std']}")
    print(f"Length   hyp={report['length']['hyp_mean']}  ref={report['length']['ref_mean']}")
    print("\nTop keyword recalls:")
    for kw, v in sorted(report["keywords"].items(), key=lambda x: -x[1]["recall"])[:5]:
        print(f"  {kw:<20} {v['recall']}% ({v['both_count']}/{v['ref_count']})")


if __name__ == "__main__":
    main()
