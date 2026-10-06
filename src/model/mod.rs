//! Portfolio math: risk model, Black-Litterman posterior, optimizer.

pub mod black_litterman;
pub mod calibration;
pub mod covariance;
pub mod optimizer;
pub mod signals;

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

pub const TRADING_DAYS: f64 = 252.0;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PortfolioStats {
    /// Annualized expected excess return under the posterior.
    pub expected_return: f64,
    /// Annualized volatility.
    pub volatility: f64,
    pub sharpe: f64,
}

pub fn portfolio_stats(
    w: &DVector<f64>,
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
) -> PortfolioStats {
    let expected_return = w.dot(mu);
    let volatility = (w.transpose() * sigma * w)[(0, 0)].max(0.0).sqrt();
    PortfolioStats {
        expected_return,
        volatility,
        sharpe: if volatility > 0.0 {
            expected_return / volatility
        } else {
            0.0
        },
    }
}
