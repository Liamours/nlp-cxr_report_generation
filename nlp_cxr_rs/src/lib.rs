use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

mod bleu;
mod chexbert;
mod rouge;

/// corpus_bleu(hypotheses, references, max_n=4) -> dict
///
/// Returns BLEU-1/2/4 and BLEU (geometric mean) as percentages.
#[pyfunction]
#[pyo3(signature = (hypotheses, references, max_n = 4))]
fn corpus_bleu(
    hypotheses: Vec<String>,
    references: Vec<String>,
    max_n: usize,
) -> HashMap<String, f64> {
    bleu::corpus_bleu(&hypotheses, &references, max_n)
}

/// rouge_l(hypotheses, references) -> float
///
/// Returns corpus ROUGE-L F1 as a percentage.
#[pyfunction]
fn rouge_l(hypotheses: Vec<String>, references: Vec<String>) -> f64 {
    rouge::rouge_l(&hypotheses, &references)
}

/// chexbert_f1(hypotheses, references) -> dict
///
/// Rule-based CheXbert F1 across 14 CheXpert conditions.
/// Returns {"chexbert_micro_f1", "chexbert_macro_f1",
///          "chexbert_per_label": {label: f1, ...}}.
#[pyfunction]
fn chexbert_f1(
    py: Python<'_>,
    hypotheses: Vec<String>,
    references: Vec<String>,
) -> PyResult<PyObject> {
    let flat = chexbert::chexbert_f1(&hypotheses, &references);

    let result    = PyDict::new_bound(py);
    let per_label = PyDict::new_bound(py);

    for (k, v) in flat {
        if let Some(label) = k.strip_prefix("chexbert_label_") {
            per_label.set_item(label, v)?;
        } else {
            result.set_item(k, v)?;
        }
    }
    result.set_item("chexbert_per_label", per_label)?;
    Ok(result.into())
}

#[pymodule]
fn nlp_cxr_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(corpus_bleu, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_l, m)?)?;
    m.add_function(wrap_pyfunction!(chexbert_f1, m)?)?;
    Ok(())
}
