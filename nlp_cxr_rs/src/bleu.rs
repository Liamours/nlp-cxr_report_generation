use std::collections::HashMap;

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

type Ngram = Vec<String>;

fn count_ngrams(tokens: &[String], n: usize) -> HashMap<Ngram, usize> {
    let mut map: HashMap<Ngram, usize> = HashMap::new();
    if tokens.len() < n {
        return map;
    }
    for i in 0..=(tokens.len() - n) {
        *map.entry(tokens[i..i + n].to_vec()).or_insert(0) += 1;
    }
    map
}

fn clipped_precision(hyp: &[String], ref_: &[String], n: usize) -> (usize, usize) {
    let hyp_ng = count_ngrams(hyp, n);
    let ref_ng = count_ngrams(ref_, n);
    let clipped: usize = hyp_ng
        .iter()
        .map(|(ng, &c)| c.min(*ref_ng.get(ng).unwrap_or(&0)))
        .sum();
    let total = hyp.len().saturating_sub(n - 1);
    (clipped, total)
}

/// Corpus BLEU — returns {"bleu_1", "bleu_2", "bleu_4", "bleu"} as percentages.
pub fn corpus_bleu(
    hypotheses: &[String],
    references: &[String],
    max_n: usize,
) -> HashMap<String, f64> {
    let weight = 1.0_f64 / max_n as f64;
    let mut clipped_counts = vec![0usize; max_n];
    let mut total_counts   = vec![0usize; max_n];
    let mut hyp_len = 0usize;
    let mut ref_len = 0usize;

    for (h, r) in hypotheses.iter().zip(references.iter()) {
        let hyp  = tokenize(h);
        let ref_ = tokenize(r);
        hyp_len += hyp.len();
        ref_len += ref_.len();
        for n in 1..=max_n {
            let (c, t) = clipped_precision(&hyp, &ref_, n);
            clipped_counts[n - 1] += c;
            total_counts[n - 1]   += t;
        }
    }

    let bp = if hyp_len == 0 {
        0.0
    } else if hyp_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    };

    let mut result = HashMap::new();
    let mut log_avg = 0.0_f64;
    let mut bleu_valid = true;

    for n in 1..=max_n {
        let p = clipped_counts[n - 1] as f64 / total_counts[n - 1].max(1) as f64;
        result.insert(format!("bleu_{n}"), bp * p * 100.0);
        if p > 0.0 && bleu_valid {
            log_avg += weight * p.ln();
        } else {
            bleu_valid = false;
        }
    }
    result.insert(
        "bleu".to_string(),
        if bleu_valid { bp * log_avg.exp() * 100.0 } else { 0.0 },
    );
    result
}
