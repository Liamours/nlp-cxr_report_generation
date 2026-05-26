use crate::bleu::tokenize;

fn lcs_length(a: &[String], b: &[String]) -> usize {
    let n = b.len();
    let mut dp = vec![0usize; n + 1];
    for tok_a in a {
        let mut prev = 0usize;
        for j in 0..n {
            let tmp = dp[j + 1];
            dp[j + 1] = if tok_a == &b[j] {
                prev + 1
            } else {
                dp[j + 1].max(dp[j])
            };
            prev = tmp;
        }
    }
    dp[n]
}

/// Corpus ROUGE-L F1 — returns a percentage.
pub fn rouge_l(hypotheses: &[String], references: &[String]) -> f64 {
    let total: f64 = hypotheses
        .iter()
        .zip(references.iter())
        .map(|(h, r)| {
            let hyp  = tokenize(h);
            let ref_ = tokenize(r);
            let lcs  = lcs_length(&hyp, &ref_) as f64;
            let prec = lcs / hyp.len().max(1) as f64;
            let rec  = lcs / ref_.len().max(1) as f64;
            let denom = prec + rec;
            if denom > 0.0 { 2.0 * prec * rec / denom } else { 0.0 }
        })
        .sum();

    if hypotheses.is_empty() {
        0.0
    } else {
        total / hypotheses.len() as f64 * 100.0
    }
}
