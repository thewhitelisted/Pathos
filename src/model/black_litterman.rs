//! Black-Litterman posterior expected returns.
//!
//! Prior: equilibrium excess returns implied by market-cap weights,
//! `π = δ Σ w_mkt`. Views: `P μ = Q + ε`, `ε ~ N(0, Ω)`. Posterior mean
//!
//! ```text
//! μ_BL = π + τΣPᵀ (PτΣPᵀ + Ω)⁻¹ (Q − Pπ)
//! ```
//!
//! which is algebraically identical to the textbook
//! `[(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]` but only inverts a `k x k`
//! SPD matrix (via Cholesky) and stays well-defined as Ω → 0.

use anyhow::{Result, anyhow, ensure};
use nalgebra::{DMatrix, DVector};

pub fn implied_returns(
    sigma: &DMatrix<f64>,
    w_mkt: &DVector<f64>,
    risk_aversion: f64,
) -> DVector<f64> {
    sigma * w_mkt * risk_aversion
}

#[derive(Debug, Clone)]
pub struct Views {
    /// `k x n` pick matrix.
    pub p: DMatrix<f64>,
    /// `k` view returns.
    pub q: DVector<f64>,
    /// Diagonal of Ω (view variances), length `k`.
    pub omega: DVector<f64>,
}

#[derive(Debug, Clone)]
pub struct Posterior {
    pub mu: DVector<f64>,
    /// Predictive covariance `Σ + M`, where `M` is the posterior covariance
    /// of the mean. Accounts for estimation uncertainty when optimizing.
    pub sigma: DMatrix<f64>,
}

pub fn posterior(
    sigma: &DMatrix<f64>,
    pi: &DVector<f64>,
    tau: f64,
    views: &Views,
) -> Result<Posterior> {
    let n = sigma.nrows();
    let k = views.p.nrows();
    ensure!(
        sigma.ncols() == n && pi.len() == n,
        "Σ and π dimensions disagree"
    );
    ensure!(
        views.p.ncols() == n,
        "P has {} columns, expected {n}",
        views.p.ncols()
    );
    ensure!(
        views.q.len() == k && views.omega.len() == k,
        "P, Q and Ω disagree on view count"
    );

    let tau_sigma = sigma * tau;
    if k == 0 {
        return Ok(Posterior {
            mu: pi.clone(),
            sigma: sigma + tau_sigma,
        });
    }

    let ts_pt = &tau_sigma * views.p.transpose(); // n x k
    let a = &views.p * &ts_pt + DMatrix::from_diagonal(&views.omega); // k x k
    let chol = a
        .cholesky()
        .ok_or_else(|| anyhow!("PτΣPᵀ + Ω is not positive definite"))?;

    let innovation = &views.q - &views.p * pi;
    let mu = pi + &ts_pt * chol.solve(&innovation);
    let m = &tau_sigma - &ts_pt * chol.solve(&ts_pt.transpose());
    Ok(Posterior {
        mu,
        sigma: sigma + m,
    })
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
    fn no_views_returns_prior() {
        let s = sigma();
        let pi = implied_returns(&s, &DVector::from_vec(vec![0.5, 0.3, 0.2]), 2.5);
        let views = Views {
            p: DMatrix::zeros(0, 3),
            q: DVector::zeros(0),
            omega: DVector::zeros(0),
        };
        let post = posterior(&s, &pi, 0.05, &views).unwrap();
        assert_relative_eq!(post.mu, pi, epsilon = 1e-12);
    }

    #[test]
    fn matches_textbook_formula() {
        let s = sigma();
        let tau = 0.05;
        let pi = implied_returns(&s, &DVector::from_vec(vec![0.5, 0.3, 0.2]), 2.5);
        let views = Views {
            p: DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, -1.0]),
            q: DVector::from_vec(vec![0.10, 0.02]),
            omega: DVector::from_vec(vec![0.001, 0.002]),
        };
        let post = posterior(&s, &pi, tau, &views).unwrap();

        let ts_inv = (&s * tau).try_inverse().unwrap();
        let o_inv = DMatrix::from_diagonal(&views.omega.map(|x| 1.0 / x));
        let pt = views.p.transpose();
        let precision = &ts_inv + &pt * &o_inv * &views.p;
        let expected = precision.try_inverse().unwrap() * (&ts_inv * &pi + &pt * &o_inv * &views.q);
        assert_relative_eq!(post.mu, expected, epsilon = 1e-10);
    }

    #[test]
    fn certain_view_is_matched_exactly() {
        let s = sigma();
        let pi = implied_returns(&s, &DVector::from_vec(vec![0.5, 0.3, 0.2]), 2.5);
        let views = Views {
            p: DMatrix::from_row_slice(1, 3, &[0.0, 1.0, 0.0]),
            q: DVector::from_vec(vec![0.25]),
            omega: DVector::from_vec(vec![0.0]),
        };
        let post = posterior(&s, &pi, 0.05, &views).unwrap();
        assert_relative_eq!(post.mu[1], 0.25, epsilon = 1e-12);
        // Correlated assets move with the view; the posterior is coherent.
        assert!(post.mu[0] > pi[0] && post.mu[2] > pi[2]);
    }
}
