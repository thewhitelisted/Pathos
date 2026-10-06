//! Data-driven Black-Litterman views.
//!
//! `pathos evaluate` estimates, on historical point-in-time data, how much
//! a unit of each signal has actually moved forward market-excess returns:
//!
//! ```text
//! (r_{i,t→t+h} · 252/h) / σ_i  =  κ · x_{i,t} + ε     (x demeaned per date)
//! ```
//!
//! a pooled regression with Driscoll-Kraay standard errors. This is Grinold's
//! "alpha = IC × volatility × score" written as a regression, so κ is in units
//! of annualized return per unit of volatility per unit of signal.
//!
//! The live model then uses, per asset with a signal,
//!
//! ```text
//! Qᵢ = πᵢ + σᵢ (κ_s x_s,i + κ_f x_f,i)
//! Ωᵢᵢ = σᵢ² (SE(κ_s)² x_s,i² + SE(κ_f)² x_f,i² + κ_s² Var(ŝᵢ))
//! ```
//!
//! i.e. Ω is the delta-method variance of the view from (1) estimation error
//! in κ and (2) sampling noise in a ticker's average headline score. A signal
//! with no demonstrated predictive power gets κ ≈ 0 and the result collapses
//! to the market prior, which is the honest outcome.

use chrono::{DateTime, NaiveDate, Utc};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use super::black_litterman::Views;
use crate::sentiment::SentimentSummary;

/// Numerical floor on view variance (annualized return², i.e. 0.1% vol).
const OMEGA_FLOOR: f64 = 1e-6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Coefficient {
    pub kappa: f64,
    pub se: f64,
    pub t: f64,
    pub dates: usize,
    pub observations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub generated_at: DateTime<Utc>,
    pub horizon_days: usize,
    pub sample_start: NaiveDate,
    pub sample_end: NaiveDate,
    pub universe: Vec<String>,
    /// Minimum headlines in the window for a ticker to have a sentiment signal.
    pub min_headlines: usize,
    pub sentiment: Option<Coefficient>,
    pub fundamentals: Option<Coefficient>,
}

impl Calibration {
    pub fn default_path() -> std::path::PathBuf {
        crate::http::cache_root().join("calibration.json")
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn summary(&self) -> String {
        let fmt = |name: &str, c: &Option<Coefficient>| match c {
            Some(c) => format!("{name} κ = {:+.3} (t = {:+.2})", c.kappa, c.t),
            None => format!("{name}: not estimated"),
        };
        format!(
            "{}; {} — {}-day horizon, {} to {}",
            fmt("sentiment", &self.sentiment),
            fmt("fundamentals", &self.fundamentals),
            self.horizon_days,
            self.sample_start,
            self.sample_end
        )
    }
}

fn demean(xs: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = xs.iter().flatten().copied().collect();
    if present.is_empty() {
        return xs.to_vec();
    }
    let m = present.iter().sum::<f64>() / present.len() as f64;
    xs.iter().map(|x| x.map(|v| v - m)).collect()
}

/// Build calibrated views. `sentiment[i]` is `None` when the ticker had too few
/// headlines; `fundamental[i]` when no fundamentals were available.
pub fn calibrated_views(
    pi: &DVector<f64>,
    vols: &[f64],
    sentiment: &[Option<&SentimentSummary>],
    fundamental: &[Option<f64>],
    cal: &Calibration,
) -> (Views, Vec<Option<f64>>) {
    let n = pi.len();
    let xs = demean(
        &sentiment
            .iter()
            .map(|s| s.map(|s| s.score))
            .collect::<Vec<_>>(),
    );
    let xf = demean(fundamental);

    let mut rows = Vec::new();
    let mut q = Vec::new();
    let mut omega = Vec::new();
    let mut per_asset = vec![None; n];
    for i in 0..n {
        let mut alpha = 0.0;
        let mut var = 0.0;
        let mut used = false;
        if let (Some(c), Some(x), Some(s)) = (&cal.sentiment, xs[i], sentiment[i]) {
            alpha += c.kappa * x;
            var += (c.se * x).powi(2) + c.kappa.powi(2) * s.score_variance();
            used = true;
        }
        if let (Some(c), Some(x)) = (&cal.fundamentals, xf[i]) {
            alpha += c.kappa * x;
            var += (c.se * x).powi(2);
            used = true;
        }
        if !used {
            continue;
        }
        let view = pi[i] + vols[i] * alpha;
        rows.push(i);
        q.push(view);
        omega.push(vols[i].powi(2) * var + OMEGA_FLOOR);
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

    fn cal(kappa: f64, se: f64) -> Calibration {
        Calibration {
            generated_at: Utc::now(),
            horizon_days: 21,
            sample_start: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            sample_end: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            universe: vec![],
            min_headlines: 3,
            sentiment: Some(Coefficient {
                kappa,
                se,
                t: kappa / se,
                dates: 100,
                observations: 1000,
            }),
            fundamentals: None,
        }
    }

    fn summary(score: f64) -> SentimentSummary {
        SentimentSummary {
            score,
            dispersion: 0.4,
            effective_count: 10.0,
            headlines: 10,
            ..Default::default()
        }
    }

    #[test]
    fn tilts_relative_to_cross_section() {
        let pi = DVector::from_vec(vec![0.05, 0.05, 0.05]);
        let (a, b, c) = (summary(0.3), summary(0.1), summary(-0.1));
        let (views, q) = calibrated_views(
            &pi,
            &[0.3; 3],
            &[Some(&a), Some(&b), Some(&c)],
            &[None; 3],
            &cal(0.5, 0.2),
        );
        assert_eq!(views.p.nrows(), 3);
        // Cross-sectional mean is 0.1, so the middle name gets no tilt.
        assert!((q[1].unwrap() - 0.05).abs() < 1e-12);
        assert!(q[0].unwrap() > 0.05 && q[2].unwrap() < 0.05);
        // α = σ κ x = 0.3 · 0.5 · 0.2
        assert!((q[0].unwrap() - 0.05 - 0.03).abs() < 1e-12);
    }

    #[test]
    fn zero_kappa_means_no_tilt() {
        let pi = DVector::from_vec(vec![0.04, 0.06]);
        let (a, b) = (summary(0.8), summary(-0.8));
        let (_, q) = calibrated_views(
            &pi,
            &[0.3; 2],
            &[Some(&a), Some(&b)],
            &[None; 2],
            &cal(0.0, 0.1),
        );
        assert_eq!(q, vec![Some(0.04), Some(0.06)]);
    }

    #[test]
    fn noisier_estimates_get_wider_omega() {
        let pi = DVector::from_vec(vec![0.05, 0.05]);
        let (a, b) = (summary(0.5), summary(-0.5));
        let s = [Some(&a), Some(&b)];
        let (precise, _) = calibrated_views(&pi, &[0.3; 2], &s, &[None; 2], &cal(0.5, 0.05));
        let (noisy, _) = calibrated_views(&pi, &[0.3; 2], &s, &[None; 2], &cal(0.5, 0.5));
        assert!(noisy.omega[0] > 10.0 * precise.omega[0]);
    }

    #[test]
    fn names_without_signals_get_no_view() {
        let pi = DVector::from_vec(vec![0.05, 0.05, 0.05]);
        let (a, b) = (summary(0.5), summary(-0.5));
        let (views, q) = calibrated_views(
            &pi,
            &[0.3; 3],
            &[Some(&a), None, Some(&b)],
            &[None; 3],
            &cal(0.5, 0.1),
        );
        assert_eq!(views.p.nrows(), 2);
        assert!(q[1].is_none());
    }
}
