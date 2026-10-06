//! Side-by-side comparison of FinBERT precisions against the f32 reference.

use std::fmt::Write as _;
use std::time::Instant;

use anyhow::{Result, ensure};
use serde::Serialize;

use super::{Probs, SentimentModel};
use crate::research::stats;

#[derive(Debug, Clone, Serialize)]
pub struct VariantResult {
    pub name: String,
    pub file_mb: f64,
    pub per_sec: f64,
    /// Share of headlines with the same top label as the reference.
    pub label_agreement: f64,
    /// Differences in polarity score (p⁺ − p⁻) versus the reference.
    pub mean_abs_score_diff: f64,
    pub p99_abs_score_diff: f64,
    pub max_abs_score_diff: f64,
    pub score_correlation: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelComparison {
    pub headlines: usize,
    pub variants: Vec<VariantResult>,
}

pub struct Variant<'a> {
    pub name: &'a str,
    pub model: &'a dyn SentimentModel,
    pub file_mb: f64,
}

fn label(p: &Probs) -> usize {
    let v = [p.positive, p.negative, p.neutral];
    (0..3).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(2)
}

/// Score `texts` with every variant; the first is the reference.
pub fn compare(variants: &[Variant], texts: &[String]) -> Result<ModelComparison> {
    ensure!(!texts.is_empty(), "no headlines to compare");
    ensure!(!variants.is_empty(), "nothing to compare");
    let mut runs = Vec::new();
    for v in variants {
        let t = Instant::now();
        let probs = v.model.predict(texts)?;
        runs.push((probs, t.elapsed().as_secs_f64()));
    }
    let reference = &runs[0].0;
    let ref_scores: Vec<f64> = reference.iter().map(Probs::score).collect();
    let n = texts.len() as f64;

    let results = variants
        .iter()
        .zip(&runs)
        .map(|(v, (probs, secs))| {
            let mut diffs: Vec<f64> = reference
                .iter()
                .zip(probs)
                .map(|(a, b)| (a.score() - b.score()).abs())
                .collect();
            let agree = reference
                .iter()
                .zip(probs)
                .filter(|(a, b)| label(a) == label(b))
                .count();
            let scores: Vec<f64> = probs.iter().map(Probs::score).collect();
            let mean = stats::mean(&diffs);
            diffs.sort_by(f64::total_cmp);
            VariantResult {
                name: v.name.to_string(),
                file_mb: v.file_mb,
                per_sec: n / secs,
                label_agreement: agree as f64 / n,
                mean_abs_score_diff: mean,
                p99_abs_score_diff: diffs[((diffs.len() - 1) as f64 * 0.99) as usize],
                max_abs_score_diff: *diffs.last().unwrap_or(&0.0),
                score_correlation: stats::pearson(&ref_scores, &scores),
            }
        })
        .collect();
    Ok(ModelComparison {
        headlines: texts.len(),
        variants: results,
    })
}

impl ModelComparison {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "\nFinBERT precision comparison on {} headlines (reference: first row)",
            self.headlines
        );
        let _ = writeln!(
            s,
            "{:<12} {:>9} {:>12} {:>12} {:>10} {:>14} {:>10}",
            "Precision", "File", "Headlines/s", "Same label", "Score r", "|Δ| mean/p99", "|Δ| max"
        );
        let _ = writeln!(s, "{}", "-".repeat(85));
        for v in &self.variants {
            let _ = writeln!(
                s,
                "{:<12} {:>6.0} MB {:>12.1} {:>11.2}% {:>10.5} {:>7.4}/{:<6.4} {:>10.4}",
                v.name,
                v.file_mb,
                v.per_sec,
                v.label_agreement * 100.0,
                v.score_correlation,
                v.mean_abs_score_diff,
                v.p99_abs_score_diff,
                v.max_abs_score_diff
            );
        }
        s
    }
}

/// Deterministic pseudo-random sample: order by an FNV-1a hash of the text.
pub fn sample(mut texts: Vec<String>, limit: usize) -> Vec<String> {
    let fnv = |t: &str| {
        t.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    };
    texts.sort_by_key(|t| fnv(t));
    texts.truncate(limit);
    texts
}
