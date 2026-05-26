"""
Rule-based CheXpert labeler for 14 radiology conditions.

Implements sentence-level negation detection (NegBio-style) to assign
binary labels to free-text radiology reports.

Output: {label: 0 | 1} for each of the 14 CheXpert conditions.

Reference:
  Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty
  Labels and Expert Comparison", AAAI 2019.

Note: This is the rule-based approach. The neural CheXbert model (fine-tuned
BERT) improves upon this, but requires the Stanford model weights. The rule-
based version is used here as a proxy metric labeled "chexbert_approx_f1".
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

LABEL_NAMES: list[str] = [
    "no_finding",
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_opacity",
    "lung_lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
]

# Keywords that indicate a condition is PRESENT (before negation check)
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "no_finding": [
        "no acute", "no significant", "no active disease",
        "unremarkable", "within normal limits", "no evidence of acute",
        "no acute cardiopulmonary", "normal chest",
    ],
    "enlarged_cardiomediastinum": [
        "cardiomediastinal silhouette", "mediastinal widening",
        "widened mediastin", "mediastinal enlarg", "prominent mediastin",
    ],
    "cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged cardiac",
        "enlarged heart", "cardiac silhouette is enlarged",
        "increased cardiac", "cardiac size is enlarged",
    ],
    "lung_opacity": [
        "opacity", "opacit", "haziness", "hazy", "infiltrat",
        "airspace", "air space", "interstitial markings",
    ],
    "lung_lesion": [
        "nodule", " mass ", "lung lesion", "pulmonary lesion",
        "pulmonary nodule", "lung mass", "mass lesion",
    ],
    "edema": [
        "pulmonary edema", "edema", "vascular congestion",
        "interstitial edema", "alveolar edema",
        "pulmonary vascular congestion", "fluid overload",
        "interstitial prominence",
    ],
    "consolidation": [
        "consolidat", "lobar consolidat", "segmental consolidat",
        "airspace consolidat",
    ],
    "pneumonia": [
        "pneumonia", "pneumonic", "infectious", "infection",
        "bacterial pneumonia", "viral pneumonia",
    ],
    "atelectasis": [
        "atelectasis", "atelectat", "subsegmental atelectasis",
        "bibasilar atelectasis", "basilar atelectasis", "discoid atelectasis",
        "plate-like atelectasis", "linear atelectasis",
    ],
    "pneumothorax": [
        "pneumothorax",
    ],
    "pleural_effusion": [
        "pleural effusion", "effusion", "hydrothorax",
        "bilateral effusion", "left effusion", "right effusion",
    ],
    "pleural_other": [
        "pleural thickening", "pleural plaque", "pleural calcifi",
        "blunting of the costophrenic", "blunted costophrenic",
        "pleural disease",
    ],
    "fracture": [
        "fracture", "rib fracture", "osseous fracture",
        "compression fracture",
    ],
    "support_devices": [
        "endotracheal tube", "et tube", "nasogastric tube", "ng tube",
        "enteric tube", "pacemaker", "icd ", "defibrillator",
        "central venous catheter", "picc", "chest tube", "drain",
        "tracheostomy", "internal jugular", "subclavian line",
        "right-sided line", "left-sided line", "cardiac device",
    ],
}

# Negation terms — checked in a window *before* the keyword
_NEGATIONS: tuple[str, ...] = (
    "no ",
    "without ",
    "absent ",
    "not ",
    "no evidence of ",
    "no evidence for ",
    "negative for ",
    "rules out ",
    "ruled out ",
    "no new ",
    "no acute ",
    "no significant ",
    "no definite ",
    "free of ",
    "resolved ",
    "resolution of ",
    "cleared ",
    "improved ",
    "improving ",
    "previously seen ",
    "no longer ",
    "cannot exclude",   # uncertainty → treat as negated for binary
    "cannot be excluded",
    "may represent",
    "possible ",
    "possibly ",
    "questionable ",
    "suspected ",
)

# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"[.;!\n]")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.lower()) if s.strip()]


def _keyword_in_sentence(sent: str, keywords: list[str]) -> list[int]:
    """Return character positions where any keyword starts in sent."""
    positions = []
    for kw in keywords:
        idx = 0
        while True:
            pos = sent.find(kw, idx)
            if pos == -1:
                break
            positions.append(pos)
            idx = pos + 1
    return positions


def _is_negated(sent: str, keyword_pos: int, window: int = 50) -> bool:
    """Check if a negation term appears in the window before keyword_pos."""
    prefix = sent[max(0, keyword_pos - window): keyword_pos]
    return any(neg in prefix for neg in _NEGATIONS)


def label_report(text: str) -> dict[str, int]:
    """Return {label: 0|1} for one report. 1 = condition affirmed."""
    if not text or not text.strip():
        return {lbl: 0 for lbl in LABEL_NAMES}

    labels: dict[str, int] = {}

    for label, keywords in _LABEL_KEYWORDS.items():
        found_positive = False
        for sent in _sentences(text):
            positions = _keyword_in_sentence(sent, keywords)
            for pos in positions:
                if not _is_negated(sent, pos):
                    found_positive = True
                    break
            if found_positive:
                break
        labels[label] = int(found_positive)

    # "No Finding": only 1 if explicitly stated AND no other condition is 1
    if labels.get("no_finding") == 1:
        other_positive = any(
            v == 1 for k, v in labels.items() if k != "no_finding"
        )
        if other_positive:
            labels["no_finding"] = 0

    return labels


def label_reports(texts: list[str]) -> list[dict[str, int]]:
    return [label_report(t) for t in texts]


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def chexbert_f1(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute CheXbert-style F1 between generated and reference reports.

    Returns:
      micro_f1   : micro-averaged F1 across all 14 labels
      macro_f1   : macro-averaged F1 (average of per-label F1)
      per_label  : {label: f1} for each of the 14 conditions
    """
    hyp_labels = label_reports(hypotheses)
    ref_labels = label_reports(references)

    per_label_f1: dict[str, float] = {}
    total_tp = total_fp = total_fn = 0

    for label in LABEL_NAMES:
        tp = fp = fn = 0
        for h, r in zip(hyp_labels, ref_labels):
            hv, rv = h[label], r[label]
            if hv == 1 and rv == 1:
                tp += 1
            elif hv == 1 and rv == 0:
                fp += 1
            elif hv == 0 and rv == 1:
                fn += 1

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_label_f1[label] = round(f1 * 100, 2)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec  = total_tp / max(total_tp + total_fn, 1)
    micro_f1   = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)
    macro_f1   = sum(per_label_f1.values()) / len(LABEL_NAMES)

    return {
        "chexbert_micro_f1": round(micro_f1 * 100, 2),
        "chexbert_macro_f1": round(macro_f1, 2),
        "chexbert_per_label": per_label_f1,
    }
