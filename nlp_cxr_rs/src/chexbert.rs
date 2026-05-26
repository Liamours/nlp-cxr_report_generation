use std::collections::HashMap;

const LABEL_NAMES: &[&str] = &[
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
];

const NEGATIONS: &[&str] = &[
    "no ", "without ", "absent ", "not ", "no evidence of ",
    "no evidence for ", "negative for ", "rules out ", "ruled out ",
    "no new ", "no acute ", "no significant ", "no definite ",
    "free of ", "resolved ", "resolution of ", "cleared ",
    "improved ", "improving ", "no longer ",
    "cannot exclude", "may represent",
    "possible ", "possibly ", "questionable ", "suspected ",
];

fn keywords_for(label: &str) -> &'static [&'static str] {
    match label {
        "no_finding" => &[
            "no acute", "no significant", "no active disease",
            "unremarkable", "within normal limits",
            "no evidence of acute", "no acute cardiopulmonary", "normal chest",
        ],
        "enlarged_cardiomediastinum" => &[
            "cardiomediastinal silhouette", "mediastinal widening",
            "widened mediastin", "mediastinal enlarg", "prominent mediastin",
        ],
        "cardiomegaly" => &[
            "cardiomegaly", "cardiac enlargement", "enlarged cardiac",
            "enlarged heart", "cardiac silhouette is enlarged",
            "increased cardiac", "cardiac size is enlarged",
        ],
        "lung_opacity" => &[
            "opacity", "opacit", "haziness", "hazy", "infiltrat",
            "airspace", "air space", "interstitial markings",
        ],
        "lung_lesion" => &[
            "nodule", " mass ", "lung lesion", "pulmonary lesion",
            "pulmonary nodule", "lung mass",
        ],
        "edema" => &[
            "pulmonary edema", "edema", "vascular congestion",
            "interstitial edema", "alveolar edema",
            "pulmonary vascular congestion", "fluid overload",
            "interstitial prominence",
        ],
        "consolidation" => &[
            "consolidat", "lobar consolidat", "segmental consolidat",
            "airspace consolidat",
        ],
        "pneumonia" => &[
            "pneumonia", "pneumonic", "infection",
            "bacterial pneumonia", "viral pneumonia",
        ],
        "atelectasis" => &[
            "atelectasis", "atelectat", "subsegmental atelectasis",
            "bibasilar atelectasis", "basilar atelectasis",
            "discoid atelectasis", "plate-like atelectasis", "linear atelectasis",
        ],
        "pneumothorax"    => &["pneumothorax"],
        "pleural_effusion" => &[
            "pleural effusion", "effusion", "hydrothorax",
            "bilateral effusion", "left effusion", "right effusion",
        ],
        "pleural_other" => &[
            "pleural thickening", "pleural plaque", "pleural calcifi",
            "blunting of the costophrenic", "blunted costophrenic",
            "pleural disease",
        ],
        "fracture" => &[
            "fracture", "rib fracture", "osseous fracture", "compression fracture",
        ],
        "support_devices" => &[
            "endotracheal tube", "et tube", "nasogastric tube", "ng tube",
            "enteric tube", "pacemaker", "icd ", "defibrillator",
            "central venous catheter", "picc", "chest tube", "drain",
            "tracheostomy", "subclavian line", "cardiac device",
        ],
        _ => &[],
    }
}

fn sentences(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c| matches!(c, '.' | ';' | '!' | '\n'))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn is_negated(sent: &str, kw_pos: usize) -> bool {
    let start = kw_pos.saturating_sub(50);
    // safe byte-boundary slice
    let prefix = &sent[start..kw_pos];
    NEGATIONS.iter().any(|neg| prefix.contains(neg))
}

fn label_report(text: &str) -> HashMap<&'static str, u8> {
    let sents = sentences(text);
    let mut labels: HashMap<&'static str, u8> =
        LABEL_NAMES.iter().map(|&n| (n, 0u8)).collect();

    'label: for &label in LABEL_NAMES {
        for kw in keywords_for(label) {
            for sent in &sents {
                if let Some(pos) = sent.find(kw) {
                    if !is_negated(sent, pos) {
                        labels.insert(label, 1);
                        continue 'label;
                    }
                }
            }
        }
    }

    // "No Finding" is only valid when no other condition is positive
    if labels.get("no_finding") == Some(&1) {
        let other_positive = LABEL_NAMES.iter()
            .filter(|&&n| n != "no_finding")
            .any(|n| labels.get(n) == Some(&1));
        if other_positive {
            labels.insert("no_finding", 0);
        }
    }
    labels
}

/// CheXbert-style F1 (rule-based, 14 labels).
/// Returns {"chexbert_micro_f1", "chexbert_macro_f1"} plus per-label keys.
pub fn chexbert_f1(
    hypotheses: &[String],
    references: &[String],
) -> HashMap<String, f64> {
    let hyp_labels: Vec<_> = hypotheses.iter().map(|t| label_report(t)).collect();
    let ref_labels: Vec<_> = references.iter().map(|t| label_report(t)).collect();

    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;
    let mut per_label: HashMap<String, f64> = HashMap::new();

    for &label in LABEL_NAMES {
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        for (h, r) in hyp_labels.iter().zip(ref_labels.iter()) {
            match (h[label], r[label]) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_ += 1,
                _      => {}
            }
        }
        let prec = tp as f64 / (tp + fp).max(1) as f64;
        let rec  = tp as f64 / (tp + fn_).max(1) as f64;
        let f1   = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        per_label.insert(label.to_string(), f1 * 100.0);
        total_tp += tp;
        total_fp += fp;
        total_fn += fn_;
    }

    let micro_prec = total_tp as f64 / (total_tp + total_fp).max(1) as f64;
    let micro_rec  = total_tp as f64 / (total_tp + total_fn).max(1) as f64;
    let micro_f1   = if micro_prec + micro_rec > 0.0 {
        2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
    } else { 0.0 };
    let macro_f1: f64 = per_label.values().sum::<f64>() / LABEL_NAMES.len() as f64;

    let mut result = HashMap::new();
    result.insert("chexbert_micro_f1".to_string(), micro_f1 * 100.0);
    result.insert("chexbert_macro_f1".to_string(), macro_f1);
    for (k, v) in per_label {
        result.insert(format!("chexbert_label_{k}"), v);
    }
    result
}
