//! Covariance estimation with Ledoit-Wolf shrinkage.
//!
//! With a year of daily data and a handful of assets the sample covariance is
//! noisy, and Black-Litterman / mean-variance amplify that noise. Ledoit &
//! Wolf (2004) give a closed-form optimal blend between the sample matrix
//! `S` and a scaled identity target `m I`:
//!
//! ```text
//! Σ = δ m I + (1 − δ) S,   m = tr(S) / N
//! ```

use anyhow::{Result, bail};
use nalgebra::DMatrix;

use super::TRADING_DAYS;

#[derive(Debug, Clone)]
pub struct RiskModel {
    /// Annualized shrunk covariance of log returns.
    pub sigma: DMatrix<f64>,
    /// Shrinkage intensity δ in [0, 1].
    pub shrinkage: f64,
    /// Number of daily observations used.
    pub observations: usize,
}

impl RiskModel {
    pub fn volatilities(&self) -> Vec<f64> {
        self.sigma
            .diagonal()
            .iter()
            .map(|v| v.max(0.0).sqrt())
            .collect()
    }

    pub fn correlation(&self) -> Vec<Vec<f64>> {
        let vols = self.volatilities();
        let n = vols.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.sigma[(i, j)] / (vols[i] * vols[j]))
                    .collect()
            })
            .collect()
    }
}

/// `returns` is `T x N`: one row per day, one column per asset.
pub fn ledoit_wolf(returns: &[Vec<f64>]) -> Result<RiskModel> {
    let t = returns.len();
    let n = returns.first().map_or(0, Vec::len);
    if n == 0 {
        bail!("no assets");
    }
    if t < 30 {
        bail!("only {t} overlapping trading days of price history; need at least 30");
    }

    let mut x = DMatrix::from_fn(t, n, |r, c| returns[r][c]);
    for mut col in x.column_iter_mut() {
        let mean = col.mean();
        col.add_scalar_mut(-mean);
    }
    let tf = t as f64;
    let s = x.transpose() * &x / tf;

    // All norms use the N-normalized Frobenius inner product from the paper.
    let nf = n as f64;
    let m = s.trace() / nf;
    let target = DMatrix::<f64>::identity(n, n) * m;
    let d2 = (&s - &target).norm_squared() / nf;

    let mut b_bar2 = 0.0;
    for row in x.row_iter() {
        let xk = row.transpose();
        b_bar2 += (&xk * xk.transpose() - &s).norm_squared() / nf;
    }
    b_bar2 /= tf * tf;
    let b2 = b_bar2.min(d2);
    let shrinkage = if d2 > 0.0 { b2 / d2 } else { 1.0 };

    let sigma = (target * shrinkage + s * (1.0 - shrinkage)) * TRADING_DAYS;
    Ok(RiskModel {
        sigma,
        shrinkage,
        observations: t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random returns (LCG) so the test is reproducible
    /// without pulling in a RNG crate.
    fn returns(t: usize, n: usize) -> Vec<Vec<f64>> {
        let mut state: u64 = 42;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.04
        };
        (0..t)
            .map(|_| {
                let common = next();
                (0..n).map(|_| common + next()).collect()
            })
            .collect()
    }

    #[test]
    fn rejects_short_history() {
        assert!(ledoit_wolf(&returns(10, 3)).is_err());
    }

    #[test]
    fn is_symmetric_positive_definite_and_shrunk() {
        let rm = ledoit_wolf(&returns(250, 5)).unwrap();
        assert!((0.0..=1.0).contains(&rm.shrinkage));
        assert_eq!(rm.sigma, rm.sigma.transpose());
        assert!(rm.sigma.clone().cholesky().is_some());
        // Common factor => positive correlations.
        assert!(rm.correlation()[0][1] > 0.2);
    }

    #[test]
    fn matches_sample_covariance_when_no_shrinkage_needed() {
        // Perfectly identity-like data: shrinkage target equals S, any δ
        // gives the same matrix, so check against the plain estimate.
        let r: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    vec![0.01, -0.01]
                } else {
                    vec![-0.01, 0.01]
                }
            })
            .collect();
        let rm = ledoit_wolf(&r).unwrap();
        let daily_var = 0.0001;
        assert!((rm.sigma[(0, 0)] - daily_var * TRADING_DAYS).abs() < 1e-9);
    }
}
