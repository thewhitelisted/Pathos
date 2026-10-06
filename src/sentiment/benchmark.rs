//! Side-by-side comparison of the f32 and int8 FinBERT models.

use std::fmt::Write as _;
use std::time::Instant;

use anyhow::{Result, ensure};
use serde::Serialize;

use super::{Probs, SentimentModel};
use crate::research::stats;

#[derive(Debug, Clone, Serialize)]
pub struct ModelComparison {
    pub headlines: usize,
    pub f32_file_mb: f64,
    pub q8_file_mb: f64,
    pub f32_per_sec: f64,
    pub q8_per_sec: f64,
    /// Share of headlines where both models pick the same top label.
    pub label_agreement: f64,
    /// Polarity score (p⁺ − p⁻) differences.
    pub mean_abs_score_diff: f64,
    pub p99_abs_score_diff: f64,
    pub max_abs_score_diff: f64,
    pub score_correlation: f64,
}

fn label(p: &Probs) -> usize {
    let v = [p.positive, p.negative, p.neutral];
    (0..3).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(2)
}

pub fn compare(
    f32_model: &dyn SentimentModel,
    q8_model: &dyn SentimentModel,
    texts: &[String],
    file_mb: (f64, f64),
) -> Result<ModelComparison> {
    ensure!(!texts.is_empty(), "no headlines to compare");
    let t0 = Instant::now();
    let a = f32_model.predict(texts)?;
    let f32_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let b = q8_model.predict(texts)?;
    let q8_secs = t1.elapsed().as_secs_f64();

    let mut diffs: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x.score() - y.score()).abs())
        .collect();
    let agree = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| label(x) == label(y))
        .count();
    let sa: Vec<f64> = a.iter().map(Probs::score).collect();
    let sb: Vec<f64> = b.iter().map(Probs::score).collect();
    let mean = stats::mean(&diffs);
    diffs.sort_by(f64::total_cmp);
    let n = texts.len() as f64;
    Ok(ModelComparison {
        headlines: texts.len(),
        f32_file_mb: file_mb.0,
        q8_file_mb: file_mb.1,
        f32_per_sec: n / f32_secs,
        q8_per_sec: n / q8_secs,
        label_agreement: agree as f64 / n,
        mean_abs_score_diff: mean,
        p99_abs_score_diff: diffs[((diffs.len() - 1) as f64 * 0.99) as usize],
        max_abs_score_diff: *diffs.last().unwrap_or(&0.0),
        score_correlation: stats::pearson(&sa, &sb),
    })
}

impl ModelComparison {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "\nFinBERT f32 vs int8 (Q8_0) on {} headlines",
            self.headlines
        );
        let _ = writeln!(s, "  {:<28} {:>10} {:>10}", "", "f32", "int8");
        let _ = writeln!(
            s,
            "  {:<28} {:>7.0} MB {:>7.0} MB",
            "Model file", self.f32_file_mb, self.q8_file_mb
        );
        let _ = writeln!(
            s,
            "  {:<28} {:>8.1}/s {:>8.1}/s",
            "Throughput", self.f32_per_sec, self.q8_per_sec
        );
        let _ = writeln!(
            s,
            "  Size reduction            {:.2}x",
            self.f32_file_mb / self.q8_file_mb
        );
        let _ = writeln!(
            s,
            "  Speed-up                  {:.2}x",
            self.q8_per_sec / self.f32_per_sec
        );
        let _ = writeln!(
            s,
            "  Same top label            {:.2}%",
            self.label_agreement * 100.0
        );
        let _ = writeln!(
            s,
            "  Score correlation         {:.5}",
            self.score_correlation
        );
        let _ = writeln!(
            s,
            "  |Δ score| mean / p99 / max {:.4} / {:.4} / {:.4}",
            self.mean_abs_score_diff, self.p99_abs_score_diff, self.max_abs_score_diff
        );
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
