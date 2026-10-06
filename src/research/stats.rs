//! Small, dependency-free statistics toolkit for signal evaluation.
//!
//! Financial panels have overlapping forward returns and persistent signals,
//! so naive i.i.d. standard errors overstate significance badly. Everything
//! here that reports a standard error is autocorrelation-robust
//! (Newey-West / Driscoll-Kraay).

use serde::Serialize;

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (n − 1).
pub fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return f64::NAN;
    }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// Ranks starting at 1, ties get their average rank.
pub fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            out[k] = avg;
        }
        i = j + 1;
    }
    out
}

pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

/// Newey-West (Bartlett kernel) long-run variance of a series about its mean.
pub fn long_run_variance(xs: &[f64], lags: usize) -> f64 {
    let n = xs.len();
    let m = mean(xs);
    let d: Vec<f64> = xs.iter().map(|x| x - m).collect();
    let gamma = |l: usize| {
        d[l..]
            .iter()
            .zip(&d[..n - l])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n as f64
    };
    let mut v = gamma(0);
    for l in 1..=lags.min(n.saturating_sub(1)) {
        v += 2.0 * (1.0 - l as f64 / (lags + 1) as f64) * gamma(l);
    }
    v.max(0.0)
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MeanEstimate {
    pub mean: f64,
    /// Newey-West standard error of the mean.
    pub se: f64,
    pub t: f64,
    pub n: usize,
}

pub fn mean_with_nw(xs: &[f64], lags: usize) -> MeanEstimate {
    let n = xs.len();
    let m = mean(xs);
    let se = (long_run_variance(xs, lags) / n as f64).sqrt();
    MeanEstimate {
        mean: m,
        se,
        t: if se > 0.0 { m / se } else { 0.0 },
        n,
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SlopeEstimate {
    pub beta: f64,
    /// Driscoll-Kraay standard error: robust to cross-sectional correlation
    /// within a date and serial correlation across dates.
    pub se: f64,
    pub t: f64,
    pub dates: usize,
    pub observations: usize,
}

/// Pooled no-intercept regression `y = βx + ε` over a panel given as one
/// `(x, y)` cross-section per date (callers demean within each date).
pub fn pooled_slope(cross_sections: &[(Vec<f64>, Vec<f64>)], lags: usize) -> Option<SlopeEstimate> {
    let sxx: f64 = cross_sections
        .iter()
        .flat_map(|(x, _)| x)
        .map(|x| x * x)
        .sum();
    let sxy: f64 = cross_sections
        .iter()
        .flat_map(|(x, y)| x.iter().zip(y))
        .map(|(x, y)| x * y)
        .sum();
    if sxx <= 0.0 || cross_sections.len() < 2 {
        return None;
    }
    let beta = sxy / sxx;
    // Per-date score contributions u_t = Σ_i x_it ε_it; Var(β̂) = LRV(Σ u_t)/(Σx²)².
    let u: Vec<f64> = cross_sections
        .iter()
        .map(|(x, y)| x.iter().zip(y).map(|(x, y)| x * (y - beta * x)).sum())
        .collect();
    let t_len = u.len() as f64;
    let se = (long_run_variance(&u, lags) * t_len).sqrt() / sxx;
    Some(SlopeEstimate {
        beta,
        se,
        t: if se > 0.0 { beta / se } else { 0.0 },
        dates: cross_sections.len(),
        observations: cross_sections.iter().map(|(x, _)| x.len()).sum(),
    })
}

pub fn max_drawdown(daily_returns: &[f64]) -> f64 {
    let (mut peak, mut equity, mut worst) = (1.0f64, 1.0f64, 0.0f64);
    for r in daily_returns {
        equity *= 1.0 + r;
        peak = peak.max(equity);
        worst = worst.min(equity / peak - 1.0);
    }
    worst
}

/// Deterministic xorshift64* generator so bootstrap results are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n.max(1)
    }
}

/// Politis-Romano stationary bootstrap: resample index paths made of blocks
/// with geometric lengths (mean `mean_block`), preserving autocorrelation.
/// Returns the (2.5%, 97.5%) percentiles of `stat` over `reps` resamples.
pub fn stationary_bootstrap_ci(
    n: usize,
    mean_block: f64,
    reps: usize,
    seed: u64,
    stat: impl Fn(&[usize]) -> f64,
) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let p = 1.0 / mean_block.max(1.0);
    let mut draws: Vec<f64> = (0..reps)
        .map(|_| {
            let mut idx = Vec::with_capacity(n);
            let mut cur = rng.below(n);
            for _ in 0..n {
                idx.push(cur);
                cur = if rng.uniform() < p {
                    rng.below(n)
                } else {
                    (cur + 1) % n
                };
            }
            stat(&idx)
        })
        .filter(|v| v.is_finite())
        .collect();
    if draws.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    draws.sort_by(f64::total_cmp);
    let q = |p: f64| draws[((draws.len() - 1) as f64 * p).round() as usize];
    (q(0.025), q(0.975))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_handle_ties() {
        assert_eq!(ranks(&[10.0, 30.0, 20.0, 30.0]), vec![1.0, 3.5, 2.0, 3.5]);
    }

    #[test]
    fn spearman_is_rank_based() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 1000.0]; // monotone, non-linear
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-12);
        let rev: Vec<f64> = y.iter().rev().copied().collect();
        assert!((spearman(&x, &rev) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn newey_west_reduces_to_iid_without_lags() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e = mean_with_nw(&xs, 0);
        // Population variance / n (NW uses 1/n normalisation).
        let pop_var = xs.iter().map(|x| (x - 3.5f64).powi(2)).sum::<f64>() / 6.0;
        assert!((e.se - (pop_var / 6.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn newey_west_widens_for_persistent_series() {
        // A slowly varying series: positive autocorrelation inflates the SE.
        let xs: Vec<f64> = (0..200).map(|i| ((i / 20) % 2) as f64).collect();
        assert!(mean_with_nw(&xs, 10).se > 2.0 * mean_with_nw(&xs, 0).se);
    }

    #[test]
    fn pooled_slope_recovers_beta() {
        let mut rng = Rng::new(7);
        let sections: Vec<(Vec<f64>, Vec<f64>)> = (0..300)
            .map(|_| {
                let x: Vec<f64> = (0..10).map(|_| rng.uniform() - 0.5).collect();
                let y = x
                    .iter()
                    .map(|x| 0.8 * x + 0.05 * (rng.uniform() - 0.5))
                    .collect();
                (x, y)
            })
            .collect();
        let est = pooled_slope(&sections, 5).unwrap();
        assert!((est.beta - 0.8).abs() < 0.01, "beta {}", est.beta);
        assert!(est.t > 50.0);
        assert_eq!(est.observations, 3000);
    }

    #[test]
    fn drawdown() {
        assert!((max_drawdown(&[0.1, -0.5, 0.2]) + 0.5).abs() < 1e-12);
        assert_eq!(max_drawdown(&[0.01, 0.02]), 0.0);
    }

    #[test]
    fn bootstrap_ci_covers_mean() {
        let mut rng = Rng::new(3);
        let xs: Vec<f64> = (0..500).map(|_| rng.uniform()).collect();
        let (lo, hi) = stationary_bootstrap_ci(xs.len(), 10.0, 500, 1, |idx| {
            idx.iter().map(|&i| xs[i]).sum::<f64>() / idx.len() as f64
        });
        assert!(lo < 0.5 && 0.5 < hi && hi - lo < 0.1, "({lo}, {hi})");
    }
}
