//! Long-only, position-capped mean-variance optimization.
//!
//! ```text
//! maximize   μᵀw − (δ/2) wᵀΣw
//! subject to Σ wᵢ = 1,  0 ≤ wᵢ ≤ u
//! ```
//!
//! Solved with FISTA (accelerated projected gradient ascent). The objective is
//! smooth and concave with Lipschitz gradient `L = δ λ_max(Σ)`, and the
//! feasible set is a "capped simplex" whose Euclidean projection we compute
//! exactly by bisection — so this converges to the global optimum without a
//! general-purpose QP solver.

use anyhow::{Result, ensure};
use nalgebra::{DMatrix, DVector};

const MAX_ITERS: usize = 20_000;
const TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct Solution {
    pub weights: DVector<f64>,
    pub iterations: usize,
    pub converged: bool,
}

pub fn optimize(
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
    risk_aversion: f64,
    max_weight: f64,
) -> Result<Solution> {
    let n = mu.len();
    ensure!(n > 0, "no assets to optimize");
    ensure!(
        sigma.nrows() == n && sigma.ncols() == n,
        "Σ must be {n}x{n}"
    );
    ensure!(risk_aversion > 0.0, "risk aversion must be positive");
    ensure!(
        max_weight * n as f64 >= 1.0 - 1e-12,
        "max weight {max_weight} is infeasible for {n} assets (needs ≥ {:.4})",
        1.0 / n as f64
    );

    let lambda_max = sigma.clone().symmetric_eigenvalues().max().max(1e-12);
    let step = 1.0 / (risk_aversion * lambda_max);
    let grad = |w: &DVector<f64>| mu - sigma * w * risk_aversion;

    let mut w = project_capped_simplex(&DVector::from_element(n, 1.0 / n as f64), max_weight);
    let mut y = w.clone();
    let mut t = 1.0f64;
    for iter in 1..=MAX_ITERS {
        let w_next = project_capped_simplex(&(&y + grad(&y) * step), max_weight);
        let t_next = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        y = &w_next + (&w_next - &w) * ((t - 1.0) / t_next);
        let delta = (&w_next - &w).norm();
        w = w_next;
        t = t_next;
        if delta < TOLERANCE {
            return Ok(Solution {
                weights: w,
                iterations: iter,
                converged: true,
            });
        }
    }
    Ok(Solution {
        weights: w,
        iterations: MAX_ITERS,
        converged: false,
    })
}

/// Euclidean projection of `v` onto `{w : Σw = 1, 0 ≤ w ≤ cap}`.
///
/// The projection is `wᵢ = clamp(vᵢ − θ, 0, cap)` for the unique θ making the
/// weights sum to one; the sum is monotone in θ so bisection finds it.
pub fn project_capped_simplex(v: &DVector<f64>, cap: f64) -> DVector<f64> {
    let sum_at = |theta: f64| v.iter().map(|x| (x - theta).clamp(0.0, cap)).sum::<f64>();
    let mut lo = v.min() - cap; // every weight at cap: sum = n·cap ≥ 1
    let mut hi = v.max(); // every weight at 0: sum = 0
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if sum_at(mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-15 {
            break;
        }
    }
    let theta = 0.5 * (lo + hi);
    v.map(|x| (x - theta).clamp(0.0, cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sigma() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.006, 0.01, 0.006, 0.09, 0.012, 0.01, 0.012, 0.0625],
        )
    }

    #[test]
    fn projection_satisfies_constraints() {
        let v = DVector::from_vec(vec![3.0, -1.0, 0.2, 0.9]);
        let w = project_capped_simplex(&v, 0.4);
        assert_relative_eq!(w.sum(), 1.0, epsilon = 1e-10);
        assert!(w.iter().all(|&x| (-1e-12..=0.4 + 1e-12).contains(&x)));
        assert_relative_eq!(w[0], 0.4, epsilon = 1e-10);
        assert_relative_eq!(w[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn recovers_market_portfolio_from_implied_returns() {
        // Reverse optimization: if μ = δΣw_mkt, the unconstrained optimum
        // (with the budget constraint slack) is exactly w_mkt.
        let s = sigma();
        let w_mkt = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let mu = &s * &w_mkt * 2.5;
        let sol = optimize(&mu, &s, 2.5, 1.0).unwrap();
        assert!(sol.converged);
        assert_relative_eq!(sol.weights, w_mkt, epsilon = 1e-6);
    }

    #[test]
    fn never_shorts_and_respects_cap() {
        let s = sigma();
        // A strongly negative view on asset 1 would make the unconstrained
        // solution short it.
        let mu = DVector::from_vec(vec![0.15, -0.30, 0.05]);
        let sol = optimize(&mu, &s, 2.5, 0.6).unwrap();
        assert_relative_eq!(sol.weights.sum(), 1.0, epsilon = 1e-9);
        assert!(
            sol.weights
                .iter()
                .all(|&w| (-1e-9..=0.6 + 1e-9).contains(&w))
        );
        assert!(sol.weights[1] < 1e-9);
        assert_relative_eq!(sol.weights[0], 0.6, epsilon = 1e-6);
    }

    #[test]
    fn optimum_beats_random_feasible_portfolios() {
        let s = sigma();
        let mu = DVector::from_vec(vec![0.08, 0.12, 0.07]);
        let delta = 3.0;
        let obj = |w: &DVector<f64>| w.dot(&mu) - 0.5 * delta * (w.transpose() * &s * w)[(0, 0)];
        let best = optimize(&mu, &s, delta, 0.5).unwrap().weights;
        for a in 0..=10 {
            for b in 0..=(10 - a) {
                let w = DVector::from_vec(vec![
                    a as f64 / 10.0,
                    b as f64 / 10.0,
                    (10 - a - b) as f64 / 10.0,
                ]);
                if w.iter().all(|&x| x <= 0.5) {
                    assert!(obj(&best) >= obj(&w) - 1e-9);
                }
            }
        }
    }

    #[test]
    fn rejects_infeasible_cap() {
        let s = sigma();
        assert!(optimize(&DVector::zeros(3), &s, 2.5, 0.2).is_err());
    }
}
