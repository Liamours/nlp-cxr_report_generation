"""
Evaluation metrics for CXR report generation.

Metric groups
─────────────
nlg        : BLEU-1/2/4, ROUGE-L, METEOR, CIDEr   (pure Python, fast)
bertscore  : BERTScore P/R/F1                       (needs bert-base-uncased)
chexbert   : CheXbert-approx micro/macro F1         (rule-based, fast)

Entry points
────────────
compute_metrics(hyps, refs, groups={"nlg"})        → flat dict of scores
compute_all_metrics(hyps, refs)                    → all three groups
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


# ---- tokenization ----------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ---- BLEU ------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(hyp: list[str], refs: list[list[str]], n: int) -> tuple[int, int]:
    hyp_ng = _ngrams(hyp, n)
    max_ref_ng: Counter = Counter()
    for ref in refs:
        for ng, cnt in _ngrams(ref, n).items():
            max_ref_ng[ng] = max(max_ref_ng[ng], cnt)
    clipped = sum(min(cnt, max_ref_ng[ng]) for ng, cnt in hyp_ng.items())
    return clipped, max(len(hyp) - n + 1, 0)


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
    max_n: int = 4,
    weights: tuple[float, ...] | None = None,
) -> dict[str, float]:
    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))

    clipped_counts = [0] * max_n
    total_counts   = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp_str, ref_input in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        refs = [_tokenize(r) for r in ([ref_input] if isinstance(ref_input, str) else ref_input)]

        hyp_len += len(hyp)
        closest = min(refs, key=lambda r: (abs(len(r) - len(hyp)), len(r)))
        ref_len += len(closest)

        for n in range(1, max_n + 1):
            c, t = _clipped_precision(hyp, refs, n)
            clipped_counts[n - 1] += c
            total_counts[n - 1] += t

    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    log_avg = 0.0
    bleu_n: dict[str, float] = {}
    for n in range(1, max_n + 1):
        p = clipped_counts[n - 1] / max(total_counts[n - 1], 1)
        bleu_n[f"bleu_{n}"] = bp * math.exp(math.log(p) if p > 0 else float("-inf"))
        if p > 0:
            log_avg += weights[n - 1] * math.log(p)
        else:
            log_avg = float("-inf")
            break

    bleu_n["bleu"] = bp * math.exp(log_avg) if log_avg != float("-inf") else 0.0
    return bleu_n


# ---- ROUGE-L ---------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            tmp = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else max(dp[j + 1], dp[j])
            prev = tmp
    return dp[n]


def rouge_l(hypotheses: Sequence[str], references: Sequence[str | Sequence[str]]) -> float:
    scores = []
    for hyp_str, ref_input in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        refs = [_tokenize(r) for r in ([ref_input] if isinstance(ref_input, str) else ref_input)]
        best = max(_lcs_length(hyp, ref) for ref in refs)
        prec = best / max(len(hyp), 1)
        rec  = best / max(max(len(r) for r in refs), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        scores.append(f1)
    return sum(scores) / max(len(scores), 1)


# ---- METEOR ----------------------------------------------------------------

def _stemmer(word: str) -> str:
    # Porter-lite: strip common suffixes for English
    suffixes = ["ing", "tion", "ed", "ly", "er", "est", "s"]
    for sfx in suffixes:
        if word.endswith(sfx) and len(word) - len(sfx) >= 3:
            return word[: -len(sfx)]
    return word


def meteor(hypotheses: Sequence[str], references: Sequence[str | Sequence[str]]) -> float:
    alpha, beta, gamma = 0.9, 3.0, 0.5

    scores = []
    for hyp_str, ref_input in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        refs = [_tokenize(r) for r in ([ref_input] if isinstance(ref_input, str) else ref_input)]

        best_f = 0.0
        for ref in refs:
            hyp_stems = [_stemmer(w) for w in hyp]
            ref_stems = [_stemmer(w) for w in ref]

            # Exact + stem match
            matched_h = set()
            matched_r = set()
            for i, hw in enumerate(hyp):
                for j, rw in enumerate(ref):
                    if j in matched_r:
                        continue
                    if hw == rw or hyp_stems[i] == ref_stems[j]:
                        matched_h.add(i)
                        matched_r.add(j)
                        break

            m = len(matched_h)
            if m == 0:
                continue
            prec = m / len(hyp)
            rec  = m / len(ref)
            fmean = prec * rec / max(alpha * prec + (1 - alpha) * rec, 1e-9)

            # Chunk penalty
            chunks = 1
            prev_h = -2
            for i in sorted(matched_h):
                if i != prev_h + 1:
                    chunks += 1
                prev_h = i
            pen = gamma * (chunks / max(m, 1)) ** beta
            best_f = max(best_f, fmean * (1 - pen))

        scores.append(best_f)
    return sum(scores) / max(len(scores), 1)


# ---- CIDEr -----------------------------------------------------------------

def _idf_weights(
    all_refs: list[list[list[str]]], n: int
) -> dict[tuple, float]:
    df: Counter = Counter()
    num_docs = sum(len(refs) for refs in all_refs)
    for refs in all_refs:
        seen = set()
        for ref in refs:
            for ng in _ngrams(ref, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    return {ng: math.log((num_docs + 1) / (cnt + 1)) for ng, cnt in df.items()}


def _cider_n(
    hypotheses: list[list[str]],
    all_refs: list[list[list[str]]],
    idf: dict[tuple, float],
    n: int,
) -> float:
    scores = []
    for hyp, refs in zip(hypotheses, all_refs):
        hyp_ng = _ngrams(hyp, n)
        hyp_vec = {ng: cnt * idf.get(ng, 0.0) for ng, cnt in hyp_ng.items()}
        hyp_norm = math.sqrt(sum(v ** 2 for v in hyp_vec.values())) or 1.0

        ref_scores = []
        for ref in refs:
            ref_ng = _ngrams(ref, n)
            ref_vec = {ng: cnt * idf.get(ng, 0.0) for ng, cnt in ref_ng.items()}
            ref_norm = math.sqrt(sum(v ** 2 for v in ref_vec.values())) or 1.0
            dot = sum(hyp_vec.get(ng, 0.0) * v for ng, v in ref_vec.items())
            ref_scores.append(dot / (hyp_norm * ref_norm))
        scores.append(sum(ref_scores) / max(len(ref_scores), 1))
    return sum(scores) / max(len(scores), 1)


def cider(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
    n_max: int = 4,
) -> float:
    hyps = [_tokenize(h) for h in hypotheses]
    all_refs = [
        [_tokenize(r) for r in ([ref] if isinstance(ref, str) else ref)]
        for ref in references
    ]
    score = 0.0
    for n in range(1, n_max + 1):
        idf = _idf_weights(all_refs, n)
        score += _cider_n(hyps, all_refs, idf, n)
    return score / n_max


# ---- NLG metrics (pure Python) --------------------------------------------

def nlg_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
) -> dict[str, float]:
    bleu = corpus_bleu(hypotheses, references)
    return {
        "bleu_1":  round(bleu["bleu_1"] * 100, 2),
        "bleu_2":  round(bleu["bleu_2"] * 100, 2),
        "bleu_4":  round(bleu["bleu"]   * 100, 2),
        "rouge_l": round(rouge_l(hypotheses, references) * 100, 2),
        "meteor":  round(meteor(hypotheses, references)  * 100, 2),
        "cider":   round(cider(hypotheses, references)   * 10,  4),
    }


# ---- BERTScore -------------------------------------------------------------

def bertscore(
    hypotheses: Sequence[str],
    references: Sequence[str],
    model_type: str = "bert-base-uncased",
) -> dict[str, float]:
    """
    BERTScore using contextual BERT embeddings.
    Uses bert-base-uncased by default (already downloaded for the model).
    """
    import evaluate as hf_evaluate
    metric = hf_evaluate.load("bertscore")
    results = metric.compute(
        predictions=list(hypotheses),
        references=list(references),
        model_type=model_type,
        lang="en",
        verbose=False,
    )
    mean = lambda lst: sum(lst) / max(len(lst), 1)
    return {
        "bertscore_p":  round(mean(results["precision"]) * 100, 2),
        "bertscore_r":  round(mean(results["recall"])    * 100, 2),
        "bertscore_f1": round(mean(results["f1"])        * 100, 2),
    }


# ---- CheXbert (rule-based) -------------------------------------------------

def chexbert_f1(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """
    Rule-based CheXbert-approximation F1 across 14 CheXpert conditions.
    Returns micro_f1, macro_f1, and per-label F1.
    """
    from src.evaluate.chexpert_labeler import chexbert_f1 as _chexbert_f1
    return _chexbert_f1(list(hypotheses), list(references))


# ---- unified entry points --------------------------------------------------

_VALID_GROUPS = {"nlg", "bertscore", "chexbert"}


def compute_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
    groups: set[str] | frozenset[str] = frozenset({"nlg"}),
) -> dict[str, float | dict]:
    """
    Compute metrics for the requested groups.

    groups: any subset of {"nlg", "bertscore", "chexbert"}

    Returns a flat dict of scalar scores, except chexbert_per_label which
    is a nested dict (label → f1).
    """
    unknown = set(groups) - _VALID_GROUPS
    if unknown:
        raise ValueError(f"Unknown metric groups: {unknown}. Valid: {_VALID_GROUPS}")

    results: dict[str, float | dict] = {}

    if "nlg" in groups:
        results.update(nlg_metrics(hypotheses, references))

    if "bertscore" in groups:
        # BERTScore only accepts single (not multi) references
        flat_refs = [
            r if isinstance(r, str) else r[0]
            for r in references
        ]
        results.update(bertscore(hypotheses, flat_refs))

    if "chexbert" in groups:
        flat_refs = [
            r if isinstance(r, str) else r[0]
            for r in references
        ]
        results.update(chexbert_f1(hypotheses, flat_refs))

    return results


def compute_all_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
) -> dict[str, float | dict]:
    """Compute all metric groups. Equivalent to compute_metrics(..., groups=all)."""
    return compute_metrics(hypotheses, references, groups=frozenset(_VALID_GROUPS))
