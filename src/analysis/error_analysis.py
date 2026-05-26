"""
Error analysis for CXR report generation.

Functions:
  - length_analysis: compare hyp vs ref token lengths
  - score_distribution: per-sample metric histogram
  - worst_best_samples: return N worst / best by ROUGE-L
  - keyword_recall: recall of key clinical terms
"""

from __future__ import annotations

import statistics
from collections import Counter

from src.evaluate.metrics import rouge_l as _rouge_l_single, _tokenize


CLINICAL_KEYWORDS = [
    "pneumonia", "effusion", "cardiomegaly", "edema", "consolidation",
    "atelectasis", "pneumothorax", "opacity", "infiltrate", "normal",
    "no acute", "clear", "stable", "worsening", "improving",
]


def length_analysis(hypotheses: list[str], references: list[str]) -> dict:
    hyp_lens = [len(_tokenize(h)) for h in hypotheses]
    ref_lens = [len(_tokenize(r)) for r in references]
    return {
        "hyp_mean": round(statistics.mean(hyp_lens), 1),
        "hyp_std":  round(statistics.stdev(hyp_lens) if len(hyp_lens) > 1 else 0.0, 1),
        "ref_mean": round(statistics.mean(ref_lens), 1),
        "ref_std":  round(statistics.stdev(ref_lens) if len(ref_lens) > 1 else 0.0, 1),
        "length_ratio": round(statistics.mean(hyp_lens) / max(statistics.mean(ref_lens), 1), 3),
    }


def score_distribution(
    hypotheses: list[str],
    references: list[str],
    n_bins: int = 10,
) -> dict:
    scores = []
    for h, r in zip(hypotheses, references):
        scores.append(_rouge_l_single([h], [r]) * 100)

    lo, hi = min(scores), max(scores)
    step = (hi - lo) / n_bins or 1.0
    bins = [0] * n_bins
    for s in scores:
        idx = min(int((s - lo) / step), n_bins - 1)
        bins[idx] += 1

    return {
        "mean":    round(statistics.mean(scores), 2),
        "median":  round(statistics.median(scores), 2),
        "std":     round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 2),
        "min":     round(lo, 2),
        "max":     round(hi, 2),
        "bins":    bins,
        "bin_edges": [round(lo + i * step, 1) for i in range(n_bins + 1)],
    }


def worst_best_samples(
    hypotheses: list[str],
    references: list[str],
    n: int = 10,
) -> dict:
    scored = [
        {"score": _rouge_l_single([h], [r]) * 100, "hyp": h, "ref": r}
        for h, r in zip(hypotheses, references)
    ]
    scored.sort(key=lambda x: x["score"])
    return {
        "worst": scored[:n],
        "best":  scored[-n:][::-1],
    }


def keyword_recall(
    hypotheses: list[str],
    references: list[str],
    keywords: list[str] | None = None,
) -> dict[str, dict]:
    kws = keywords or CLINICAL_KEYWORDS
    results = {}
    for kw in kws:
        in_ref   = sum(1 for r in references  if kw in r.lower())
        in_both  = sum(1 for h, r in zip(hypotheses, references)
                       if kw in r.lower() and kw in h.lower())
        recall = in_both / max(in_ref, 1)
        results[kw] = {
            "ref_count":  in_ref,
            "both_count": in_both,
            "recall":     round(recall * 100, 1),
        }
    return results


def run_full_analysis(
    hypotheses: list[str],
    references: list[str],
    n_examples: int = 5,
) -> dict:
    return {
        "length":       length_analysis(hypotheses, references),
        "distribution": score_distribution(hypotheses, references),
        "examples":     worst_best_samples(hypotheses, references, n=n_examples),
        "keywords":     keyword_recall(hypotheses, references),
    }
