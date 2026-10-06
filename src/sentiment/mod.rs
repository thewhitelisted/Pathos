//! Headline sentiment scoring and per-ticker aggregation.

pub mod download;
pub mod finbert;
pub mod store;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::data::news::Headline;

/// Class probabilities for one piece of text.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Probs {
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
}

impl Probs {
    /// Polarity in [-1, 1].
    pub fn score(&self) -> f64 {
        self.positive - self.negative
    }
}

/// Anything that can turn headlines into class probabilities. FinBERT in
/// production; a deterministic stub in tests.
pub trait SentimentModel: Send + Sync {
    fn predict(&self, texts: &[String]) -> Result<Vec<Probs>>;
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredHeadline {
    #[serde(flatten)]
    pub headline: Headline,
    pub probs: Probs,
    pub score: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SentimentSummary {
    /// Recency-weighted mean polarity in [-1, 1].
    pub score: f64,
    /// How much to trust `score`, in [0, 1). Grows with the (effective)
    /// number of headlines and shrinks when they disagree.
    pub confidence: f64,
    pub headlines: usize,
    /// Recency-weighted standard deviation of headline scores.
    pub dispersion: f64,
    /// Kish effective number of headlines after recency weighting.
    pub effective_count: f64,
    pub positive: usize,
    pub negative: usize,
    pub neutral: usize,
}

impl SentimentSummary {
    /// Sampling variance of `score` as an estimate of the underlying mean
    /// sentiment: σ²/n_eff, with a floor on σ so one or two headlines that
    /// happen to agree are not treated as certain.
    pub fn score_variance(&self) -> f64 {
        if self.effective_count <= 0.0 {
            return 0.0;
        }
        self.dispersion.max(0.3).powi(2) / self.effective_count
    }
}

/// Half-life for headline recency weighting.
const HALF_LIFE_DAYS: f64 = 3.0;
/// Effective headline count at which confidence from volume reaches 50%.
const CONFIDENCE_HALF_COUNT: f64 = 6.0;

pub fn summarize(scored: &[ScoredHeadline], now: DateTime<Utc>) -> SentimentSummary {
    let aged: Vec<(f64, Probs)> = scored
        .iter()
        .map(|s| {
            let age_days = (now - s.headline.published).num_seconds().max(0) as f64 / 86_400.0;
            (age_days, s.probs)
        })
        .collect();
    summarize_aged(&aged)
}

/// Aggregate `(age in days, probabilities)` pairs. Shared by the live
/// analyzer and historical evaluation so both compute exactly the same signal.
pub fn summarize_aged(items: &[(f64, Probs)]) -> SentimentSummary {
    if items.is_empty() {
        return SentimentSummary::default();
    }
    let weights: Vec<f64> = items
        .iter()
        .map(|(age, _)| 0.5f64.powf(age / HALF_LIFE_DAYS))
        .collect();
    let w_sum: f64 = weights.iter().sum();
    let w_sq: f64 = weights.iter().map(|w| w * w).sum();
    let mean = items
        .iter()
        .zip(&weights)
        .map(|((_, p), w)| p.score() * w)
        .sum::<f64>()
        / w_sum;
    let var = items
        .iter()
        .zip(&weights)
        .map(|((_, p), w)| w * (p.score() - mean).powi(2))
        .sum::<f64>()
        / w_sum;

    // Kish effective sample size: many stale headlines count for less.
    let n_eff = w_sum * w_sum / w_sq;
    let volume = n_eff / (n_eff + CONFIDENCE_HALF_COUNT);
    // Scores live in [-1, 1] so the std dev is at most 1.
    let agreement = 1.0 - 0.5 * var.sqrt().min(1.0);

    let label = |p: &Probs| {
        if p.positive >= p.negative && p.positive >= p.neutral {
            0
        } else if p.negative >= p.neutral {
            1
        } else {
            2
        }
    };
    let count = |l| items.iter().filter(|(_, p)| label(p) == l).count();

    SentimentSummary {
        score: mean,
        confidence: volume * agreement,
        headlines: items.len(),
        dispersion: var.sqrt(),
        effective_count: n_eff,
        positive: count(0),
        negative: count(1),
        neutral: count(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(score: f64, hours_ago: i64, now: DateTime<Utc>) -> ScoredHeadline {
        let (positive, negative) = if score >= 0.0 {
            (score, 0.0)
        } else {
            (0.0, -score)
        };
        ScoredHeadline {
            headline: Headline {
                title: "t".into(),
                url: String::new(),
                source: String::new(),
                published: now - chrono::Duration::hours(hours_ago),
            },
            probs: Probs {
                positive,
                negative,
                neutral: 1.0 - positive - negative,
            },
            score,
        }
    }

    #[test]
    fn empty_has_no_confidence() {
        let s = summarize(&[], Utc::now());
        assert_eq!(s.confidence, 0.0);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn recent_headlines_dominate() {
        let now = Utc::now();
        let s = summarize(&[scored(0.9, 1, now), scored(-0.9, 24 * 14, now)], now);
        assert!(s.score > 0.7, "score {}", s.score);
    }

    #[test]
    fn confidence_grows_with_volume_and_agreement() {
        let now = Utc::now();
        let few = summarize(&vec![scored(0.5, 1, now); 2], now);
        let many = summarize(&vec![scored(0.5, 1, now); 20], now);
        let mixed: Vec<_> = (0..20)
            .map(|i| scored(if i % 2 == 0 { 0.9 } else { -0.9 }, 1, now))
            .collect();
        let mixed = summarize(&mixed, now);
        assert!(many.confidence > few.confidence);
        assert!(many.confidence > mixed.confidence);
        assert!(many.confidence < 1.0);
    }
}
