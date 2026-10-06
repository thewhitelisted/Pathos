//! Turning sentiment and fundamentals into Black-Litterman views.

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

use super::black_litterman::Views;
use crate::data::sec::Fundamentals;
use crate::sentiment::SentimentSummary;

/// Confidence assigned to a full set of fundamentals. Deliberately modest:
/// annual figures say more about quality than about the next few months.
const FUNDAMENTALS_CONFIDENCE: f64 = 0.35;
/// Views below this confidence are dropped rather than entered with a huge Ω.
const MIN_VIEW_CONFIDENCE: f64 = 0.02;
const MAX_VIEW_CONFIDENCE: f64 = 0.95;

/// Quality score in [-1, 1] from growth, profitability and leverage, plus the
/// fraction of those three metrics that were available.
pub fn fundamental_score(f: &Fundamentals) -> Option<(f64, f64)> {
    let parts: Vec<f64> = [
        f.revenue_growth.map(|g| (g / 0.15).tanh()),
        f.net_margin.map(|m| ((m - 0.08) / 0.10).tanh()),
        f.debt_to_equity.map(|de| {
            if de.is_finite() {
                ((1.0 - de) / 1.0).tanh()
            } else {
                -1.0
            }
        }),
    ]
    .into_iter()
    .flatten()
    .collect();
    if parts.is_empty() {
        return None;
    }
    let coverage = parts.len() as f64 / 3.0;
    Some((parts.iter().sum::<f64>() / parts.len() as f64, coverage))
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Signal {
    /// Combined directional signal in [-1, 1].
    pub value: f64,
    /// Confidence in [0, 1).
    pub confidence: f64,
    pub fundamental_score: Option<f64>,
}

/// Confidence-weighted blend of the sentiment and fundamentals signals.
pub fn combine(
    sentiment: &SentimentSummary,
    fundamentals: Option<&Fundamentals>,
    sentiment_weight: f64,
) -> Signal {
    combine_scores(
        sentiment,
        fundamentals.and_then(fundamental_score),
        sentiment_weight,
    )
}

/// [`combine`] given a precomputed `(fundamental score, coverage)`.
pub fn combine_scores(
    sentiment: &SentimentSummary,
    fund: Option<(f64, f64)>,
    sentiment_weight: f64,
) -> Signal {
    let sw = sentiment_weight.clamp(0.0, 1.0);
    let ws = sw * sentiment.confidence;
    let (fs, fval) = match fund {
        Some((score, coverage)) => ((1.0 - sw) * FUNDAMENTALS_CONFIDENCE * coverage, score),
        None => (0.0, 0.0),
    };
    let total = ws + fs;
    Signal {
        value: if total > 0.0 {
            (ws * sentiment.score + fs * fval) / total
        } else {
            0.0
        },
        confidence: total.min(MAX_VIEW_CONFIDENCE),
        fundamental_score: fund.map(|(s, _)| s),
    }
}

/// One absolute view per asset with a meaningful signal:
///
/// ```text
/// Qᵢ = πᵢ + κ · signalᵢ · σᵢ          (tilt by a fraction of the asset's vol)
/// Ωᵢᵢ = τ σᵢ² · (1 − cᵢ) / cᵢ           (Idzorek-style confidence mapping)
/// ```
///
/// With `c = 0.5`, Ω equals the prior's uncertainty `τσ²` and the posterior
/// lands halfway between π and Q; as `c → 1` it moves all the way to Q.
pub fn build_views(
    pi: &DVector<f64>,
    vols: &[f64],
    signals: &[Signal],
    tau: f64,
    view_scale: f64,
) -> (Views, Vec<Option<f64>>) {
    let n = pi.len();
    let mut rows = Vec::new();
    let mut q = Vec::new();
    let mut omega = Vec::new();
    let mut per_asset = vec![None; n];

    for (i, s) in signals.iter().enumerate() {
        if s.confidence < MIN_VIEW_CONFIDENCE {
            continue;
        }
        let c = s.confidence.clamp(MIN_VIEW_CONFIDENCE, MAX_VIEW_CONFIDENCE);
        let view = pi[i] + view_scale * s.value * vols[i];
        rows.push(i);
        q.push(view);
        omega.push(tau * vols[i].powi(2) * (1.0 - c) / c);
        per_asset[i] = Some(view);
    }

    let mut p = DMatrix::zeros(rows.len(), n);
    for (k, &i) in rows.iter().enumerate() {
        p[(k, i)] = 1.0;
    }
    (
        Views {
            p,
            q: DVector::from_vec(q),
            omega: DVector::from_vec(omega),
        },
        per_asset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentiment(score: f64, confidence: f64) -> SentimentSummary {
        SentimentSummary {
            score,
            confidence,
            ..Default::default()
        }
    }

    #[test]
    fn healthy_company_scores_positive() {
        let f = Fundamentals {
            revenue_growth: Some(0.2),
            net_margin: Some(0.25),
            debt_to_equity: Some(0.3),
            ..Default::default()
        };
        let (score, coverage) = fundamental_score(&f).unwrap();
        assert!(score > 0.5);
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn negative_equity_is_penalized() {
        let f = Fundamentals {
            debt_to_equity: Some(f64::INFINITY),
            ..Default::default()
        };
        assert_eq!(fundamental_score(&f).unwrap().0, -1.0);
    }

    #[test]
    fn no_information_means_no_view() {
        let s = combine(&sentiment(0.0, 0.0), None, 0.7);
        assert_eq!(s.confidence, 0.0);
        let pi = DVector::from_vec(vec![0.05]);
        let (views, per_asset) = build_views(&pi, &[0.3], &[s], 0.05, 0.25);
        assert_eq!(views.p.nrows(), 0);
        assert!(per_asset[0].is_none());
    }

    #[test]
    fn views_tilt_in_signal_direction() {
        let pi = DVector::from_vec(vec![0.05, 0.05]);
        let signals = [
            combine(&sentiment(0.8, 0.6), None, 0.7),
            combine(&sentiment(-0.8, 0.6), None, 0.7),
        ];
        let (views, _) = build_views(&pi, &[0.3, 0.3], &signals, 0.05, 0.25);
        assert_eq!(views.p.nrows(), 2);
        assert!(views.q[0] > 0.05 && views.q[1] < 0.05);
        assert!(views.omega.iter().all(|&o| o > 0.0));
    }
}
