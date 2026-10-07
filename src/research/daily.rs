//! Daily long-short test of a short-horizon signal.
//!
//! Each trading day `t` the names with a signal are ranked; the top fraction
//! is bought and the bottom fraction sold short, equal-weighted, $1 per leg.
//! The position is held from the close of `t` to the close of `t + 1`.
//! Because the signal at `t` only uses headlines dated before `t`, there is
//! no look-ahead. Turnover is measured as `Σ|w_t − w_{t−1}|` (both legs, so
//! a full replacement of the book is 4.0), and the break-even cost is the
//! one-way cost per unit traded at which the mean net return reaches zero.

use serde::Serialize;

use super::panel::Panel;
use super::stats::{self, MeanEstimate};
use crate::model::TRADING_DAYS;

#[derive(Debug, Clone, Serialize)]
pub struct NetResult {
    pub cost_bps: f64,
    pub annual_return: f64,
    pub sharpe: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DailyLongShort {
    pub signal: String,
    /// Fraction of ranked names in each leg.
    pub leg_fraction: f64,
    pub days: usize,
    pub avg_names_per_leg: f64,
    /// Mean daily gross return with a Newey-West standard error.
    pub daily_gross: MeanEstimate,
    pub gross_annual_return: f64,
    pub gross_annual_volatility: f64,
    pub gross_sharpe: f64,
    /// Average Σ|Δw| per day, excluding the initial build.
    pub avg_daily_turnover: f64,
    /// One-way cost (bps of value traded) that makes the mean net return zero.
    pub break_even_cost_bps: f64,
    pub net: Vec<NetResult>,
}

pub const COST_LEVELS_BPS: [f64; 3] = [10.0, 25.0, 50.0];

// `t` indexes the price panel and every ticker's signal row at once.
#[allow(clippy::needless_range_loop)]
pub fn run(
    panel: &Panel,
    signal: &[Vec<Option<f64>>],
    name: &str,
    t_start: usize,
    t_end: usize,
    leg_fraction: f64,
) -> Option<DailyLongShort> {
    let n = signal.len();
    let mut prev = vec![0.0; n];
    let mut gross = Vec::new();
    let mut turnover = Vec::new();
    let mut names = 0usize;

    for t in t_start..t_end.min(panel.len().saturating_sub(1)) {
        let mut ranked: Vec<(usize, f64)> = (0..n)
            .filter(|&i| panel.simple_return(i, t + 1).is_some())
            .filter_map(|i| Some((i, signal[i][t]?)))
            .collect();
        let k = ((ranked.len() as f64) * leg_fraction).floor() as usize;
        if k < 2 {
            // Too few names to form both legs: hold nothing today.
            if prev.iter().any(|w| *w != 0.0) {
                turnover.push(prev.iter().map(|w: &f64| w.abs()).sum());
                prev = vec![0.0; n];
            }
            gross.push(0.0);
            continue;
        }
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut w = vec![0.0; n];
        for &(i, _) in &ranked[ranked.len() - k..] {
            w[i] = 1.0 / k as f64;
        }
        for &(i, _) in &ranked[..k] {
            w[i] = -1.0 / k as f64;
        }
        if prev.iter().any(|x| *x != 0.0) {
            turnover.push(w.iter().zip(&prev).map(|(a, b)| (a - b).abs()).sum());
        }
        let r: f64 = (0..n)
            .filter(|&i| w[i] != 0.0)
            .map(|i| w[i] * panel.simple_return(i, t + 1).unwrap_or(0.0))
            .sum();
        gross.push(r);
        names += k;
        prev = w;
    }

    let active_days = gross.iter().filter(|r| **r != 0.0).count();
    if gross.len() < 60 || active_days < 30 {
        return None;
    }
    let daily = stats::mean_with_nw(&gross, 5);
    let vol = stats::std_dev(&gross) * TRADING_DAYS.sqrt();
    let avg_turnover = if turnover.is_empty() {
        0.0
    } else {
        stats::mean(&turnover)
    };
    // Mean cost per day at c bps = c/1e4 · total turnover / days.
    let turnover_per_day = turnover.iter().sum::<f64>() / gross.len() as f64;
    let net = COST_LEVELS_BPS
        .iter()
        .map(|&c| {
            let mean = daily.mean - c / 1e4 * turnover_per_day;
            NetResult {
                cost_bps: c,
                annual_return: mean * TRADING_DAYS,
                sharpe: if vol > 0.0 {
                    mean * TRADING_DAYS / vol
                } else {
                    0.0
                },
            }
        })
        .collect();

    Some(DailyLongShort {
        signal: name.to_string(),
        leg_fraction,
        days: gross.len(),
        avg_names_per_leg: names as f64 / active_days.max(1) as f64,
        gross_annual_return: daily.mean * TRADING_DAYS,
        gross_annual_volatility: vol,
        gross_sharpe: if vol > 0.0 {
            daily.mean * TRADING_DAYS / vol
        } else {
            0.0
        },
        daily_gross: daily,
        avg_daily_turnover: avg_turnover,
        break_even_cost_bps: if turnover_per_day > 0.0 {
            daily.mean / turnover_per_day * 1e4
        } else {
            f64::NAN
        },
        net,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::prices::PriceHistory;
    use chrono::NaiveDate;

    /// Six names; name i returns (i − 2.5)·1% every day, and the signal
    /// ranks them correctly, so the long-short earns the top-minus-bottom
    /// spread daily with zero turnover after the first day.
    #[test]
    fn perfect_signal_earns_spread_without_turnover() {
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let days = 120;
        let make = |ticker: &str, daily: f64| PriceHistory {
            ticker: ticker.into(),
            name: None,
            currency: None,
            last_price: 1.0,
            splits: vec![],
            closes: (0..days)
                .map(|k| {
                    (
                        start + chrono::Duration::days(k),
                        100.0 * (1.0 + daily).powi(k as i32),
                    )
                })
                .collect(),
        };
        let bench = make("SPY", 0.0);
        let hs: Vec<PriceHistory> = (0..6)
            .map(|i| make(&format!("S{i}"), (i as f64 - 2.5) * 0.01))
            .collect();
        let refs: Vec<&PriceHistory> = hs.iter().collect();
        let panel = Panel::new(&bench, &refs);
        let signal: Vec<Vec<Option<f64>>> = (0..6)
            .map(|i| vec![Some(i as f64); days as usize])
            .collect();

        let res = run(&panel, &signal, "perfect", 0, panel.len() - 1, 1.0 / 3.0).unwrap();
        // Long names 4,5 (+1.5%, +2.5%), short 0,1 (−2.5%, −1.5%): +4%/day.
        assert!(
            (res.daily_gross.mean - 0.04).abs() < 1e-9,
            "{}",
            res.daily_gross.mean
        );
        assert_eq!(res.avg_daily_turnover, 0.0);
        assert!((res.avg_names_per_leg - 2.0).abs() < 1e-12);
    }

    #[test]
    fn useless_signal_has_no_edge_and_full_turnover_costs() {
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let days = 200i64;
        let mut rng = stats::Rng::new(11);
        let hs: Vec<PriceHistory> = (0..12)
            .map(|i| {
                let mut px = 100.0;
                PriceHistory {
                    ticker: format!("S{i}"),
                    name: None,
                    currency: None,
                    last_price: 1.0,
                    splits: vec![],
                    closes: (0..days)
                        .map(|k| {
                            px *= 1.0 + (rng.uniform() - 0.5) * 0.04;
                            (start + chrono::Duration::days(k), px)
                        })
                        .collect(),
                }
            })
            .collect();
        let bench = hs[0].clone();
        let refs: Vec<&PriceHistory> = hs.iter().collect();
        let panel = Panel::new(&bench, &refs);
        let signal: Vec<Vec<Option<f64>>> = (0..12)
            .map(|_| (0..days).map(|_| Some(rng.uniform())).collect())
            .collect();
        let res = run(&panel, &signal, "noise", 0, panel.len() - 1, 1.0 / 3.0).unwrap();
        assert!(res.daily_gross.t.abs() < 3.0, "t {}", res.daily_gross.t);
        assert!(res.avg_daily_turnover > 1.5, "random ranks churn the book");
        assert!(res.net[2].annual_return < res.gross_annual_return);
    }
}
