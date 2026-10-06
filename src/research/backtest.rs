//! Walk-forward backtest of the allocation strategies.
//!
//! At each rebalance date `r` every strategy sees only information available
//! at the close of `r`: trailing prices for the covariance, point-in-time
//! fundamentals and share counts, headlines dated before `r`, and — for the
//! calibrated strategy — a κ estimated only from forward returns that had
//! fully *realized* by `r`. Weights then drift with prices until the next
//! rebalance, and turnover is charged a proportional cost.

use nalgebra::DVector;
use serde::Serialize;

use super::panel::{FundamentalPanel, Panel};
use super::stats::{self, SlopeEstimate};
use crate::model::calibration::{Calibration, Coefficient, calibrated_views};
use crate::model::{black_litterman, covariance, optimizer, signals};
use crate::sentiment::SentimentSummary;

/// One date's cross-section for calibration: demeaned signal `x` and
/// vol-scaled, annualized, demeaned forward excess return `y`.
#[derive(Debug, Clone)]
pub struct CrossSection {
    pub t: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Pooled slope using only cross-sections whose forward window had closed by
/// `as_of` (`t + horizon ≤ as_of`).
pub fn estimate(
    sections: &[CrossSection],
    horizon: usize,
    as_of: Option<usize>,
    lags: usize,
) -> Option<SlopeEstimate> {
    let usable: Vec<(Vec<f64>, Vec<f64>)> = sections
        .iter()
        .filter(|s| as_of.is_none_or(|a| s.t + horizon <= a))
        .map(|s| (s.x.clone(), s.y.clone()))
        .collect();
    stats::pooled_slope(&usable, lags)
}

pub fn to_coefficient(e: SlopeEstimate) -> Coefficient {
    Coefficient {
        kappa: e.beta,
        se: e.se,
        t: e.t,
        dates: e.dates,
        observations: e.observations,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BacktestConfig {
    pub rebalance_every: usize,
    pub cov_window: usize,
    pub max_weight: f64,
    pub risk_aversion: f64,
    pub tau: f64,
    pub cost_bps: f64,
    pub horizon: usize,
    pub min_calibration_dates: usize,
    pub calibration_lags: usize,
    pub min_headlines: usize,
    /// Parameters for the hand-tuned (uncalibrated) view mapping.
    pub view_scale: f64,
    pub sentiment_weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StrategyStats {
    pub annual_return: f64,
    pub annual_volatility: f64,
    /// Return / volatility with a zero risk-free rate.
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub avg_turnover: f64,
    pub annual_cost: f64,
    /// Versus the market-cap strategy.
    pub active_return: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    /// 95% stationary-bootstrap interval for the annualized active return.
    pub active_return_ci: (f64, f64),
}

#[derive(Debug, Clone, Serialize)]
pub struct StrategyResult {
    pub name: String,
    pub daily_returns: Vec<f64>,
    pub turnover: Vec<f64>,
    pub stats: Option<StrategyStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BacktestResult {
    pub config: BacktestConfig,
    pub start: chrono::NaiveDate,
    pub end: chrono::NaiveDate,
    pub dates: Vec<chrono::NaiveDate>,
    pub rebalances: usize,
    /// Rebalances where the calibrated strategy had enough history to form
    /// views (before that it equals "BL, no views").
    pub calibrated_rebalances: usize,
    pub market_cap_fallbacks: usize,
    pub strategies: Vec<StrategyResult>,
}

pub const STRATEGIES: [&str; 5] = [
    "Market cap",
    "Equal weight",
    "BL, no views",
    "BL + heuristic views",
    "BL + calibrated views",
];

pub struct Inputs<'a> {
    pub panel: &'a Panel,
    pub sentiment: &'a [Vec<Option<SentimentSummary>>],
    pub fundamentals: &'a FundamentalPanel,
    pub sentiment_sections: &'a [CrossSection],
    pub fundamental_sections: &'a [CrossSection],
}

pub fn run(inp: &Inputs, cfg: &BacktestConfig, start: usize) -> Option<BacktestResult> {
    let panel = inp.panel;
    let n = panel.tickers.len();
    let first = start.max(cfg.cov_window);
    let last = panel.len().checked_sub(1)?;
    if first + cfg.rebalance_every > last {
        return None;
    }

    let mut holdings = vec![vec![0.0; n]; STRATEGIES.len()];
    let mut returns = vec![Vec::new(); STRATEGIES.len()];
    let mut turnovers = vec![Vec::new(); STRATEGIES.len()];
    let mut dates = Vec::new();
    let (mut rebalances, mut calibrated, mut fallbacks) = (0, 0, 0);

    let mut r = first;
    while r < last {
        // ---- rebalance at the close of r ----
        let universe: Vec<usize> = (0..n)
            .filter(|&i| {
                panel
                    .trailing_log_returns(&[i], r, cfg.cov_window)
                    .is_some()
            })
            .collect();
        let mut costs = vec![0.0; STRATEGIES.len()];
        if universe.len() >= 2
            && let Some(targets) = targets(inp, cfg, r, &universe, &mut calibrated, &mut fallbacks)
        {
            rebalances += 1;
            for (s, target) in targets.into_iter().enumerate() {
                let mut full = vec![0.0; n];
                for (k, &i) in universe.iter().enumerate() {
                    full[i] = target[k];
                }
                // The initial build from cash costs every strategy the same,
                // so it is excluded from turnover and costs to keep the
                // comparison about rebalancing behaviour.
                if holdings[s].iter().any(|w| *w != 0.0) {
                    let turnover: f64 = full
                        .iter()
                        .zip(&holdings[s])
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    turnovers[s].push(turnover);
                    costs[s] = turnover * cfg.cost_bps / 1e4;
                }
                holdings[s] = full;
            }
        }

        // ---- hold until the next rebalance, letting weights drift ----
        let next = (r + cfg.rebalance_every).min(last);
        for d in r + 1..=next {
            dates.push(panel.dates[d]);
            for s in 0..STRATEGIES.len() {
                let rets: Vec<f64> = (0..n)
                    .map(|i| panel.simple_return(i, d).unwrap_or(0.0))
                    .collect();
                let gross: f64 = holdings[s].iter().zip(&rets).map(|(w, r)| w * r).sum();
                let cost = if d == r + 1 { costs[s] } else { 0.0 };
                returns[s].push(gross - cost);
                let growth = 1.0 + gross;
                if growth > 0.0 {
                    for (w, r) in holdings[s].iter_mut().zip(&rets) {
                        *w *= (1.0 + r) / growth;
                    }
                }
            }
        }
        r = next;
    }

    let benchmark = returns[0].clone();
    let strategies = STRATEGIES
        .iter()
        .enumerate()
        .map(|(s, name)| StrategyResult {
            name: name.to_string(),
            stats: strategy_stats(&returns[s], &benchmark, &turnovers[s], cfg),
            daily_returns: returns[s].clone(),
            turnover: turnovers[s].clone(),
        })
        .collect();

    Some(BacktestResult {
        config: cfg.clone(),
        start: panel.dates[first],
        end: panel.dates[last],
        dates,
        rebalances,
        calibrated_rebalances: calibrated,
        market_cap_fallbacks: fallbacks,
        strategies,
    })
}

/// Target weights (over `universe`) for every strategy at rebalance `r`.
fn targets(
    inp: &Inputs,
    cfg: &BacktestConfig,
    r: usize,
    universe: &[usize],
    calibrated: &mut usize,
    fallbacks: &mut usize,
) -> Option<Vec<Vec<f64>>> {
    let k = universe.len();
    let returns = inp
        .panel
        .trailing_log_returns(universe, r, cfg.cov_window)?;
    let risk = covariance::ledoit_wolf(&returns).ok()?;
    let vols = risk.volatilities();
    let cap = cfg.max_weight.max(1.0 / k as f64);

    let caps: Option<Vec<f64>> = universe
        .iter()
        .map(|&i| inp.fundamentals.market_cap[i][r])
        .collect();
    let w_mkt = match caps {
        Some(c) => {
            let total: f64 = c.iter().sum();
            DVector::from_iterator(k, c.iter().map(|x| x / total))
        }
        None => {
            *fallbacks += 1;
            DVector::from_element(k, 1.0 / k as f64)
        }
    };
    let pi = black_litterman::implied_returns(&risk.sigma, &w_mkt, cfg.risk_aversion);
    let optimize = |views: &black_litterman::Views| -> Option<Vec<f64>> {
        let post = black_litterman::posterior(&risk.sigma, &pi, cfg.tau, views).ok()?;
        let sol = optimizer::optimize(&post.mu, &post.sigma, cfg.risk_aversion, cap).ok()?;
        Some(sol.weights.iter().copied().collect())
    };
    let no_views = black_litterman::Views {
        p: nalgebra::DMatrix::zeros(0, k),
        q: DVector::zeros(0),
        omega: DVector::zeros(0),
    };

    let sentiment: Vec<Option<&SentimentSummary>> = universe
        .iter()
        .map(|&i| inp.sentiment[i][r].as_ref())
        .collect();
    let fund: Vec<Option<(f64, f64)>> = universe
        .iter()
        .map(|&i| inp.fundamentals.score[i][r])
        .collect();

    // Hand-tuned mapping, exactly as the live analyzer's default.
    let empty = SentimentSummary::default();
    let heuristic: Vec<signals::Signal> = sentiment
        .iter()
        .zip(&fund)
        .map(|(s, f)| signals::combine_scores(s.unwrap_or(&empty), *f, cfg.sentiment_weight))
        .collect();
    let (h_views, _) = signals::build_views(&pi, &vols, &heuristic, cfg.tau, cfg.view_scale);

    // Calibrated mapping, estimated strictly out of sample.
    let est = |sections: &[CrossSection]| {
        estimate(sections, cfg.horizon, Some(r), cfg.calibration_lags)
            .filter(|e| e.dates >= cfg.min_calibration_dates)
            .map(to_coefficient)
    };
    let cal = Calibration {
        generated_at: chrono::Utc::now(),
        horizon_days: cfg.horizon,
        sample_start: inp.panel.dates[0],
        sample_end: inp.panel.dates[r],
        universe: vec![],
        min_headlines: cfg.min_headlines,
        sentiment: est(inp.sentiment_sections),
        fundamentals: est(inp.fundamental_sections),
    };
    let c_views = if cal.sentiment.is_some() || cal.fundamentals.is_some() {
        *calibrated += 1;
        let fscore: Vec<Option<f64>> = fund.iter().map(|f| f.map(|(s, _)| s)).collect();
        calibrated_views(&pi, &vols, &sentiment, &fscore, &cal).0
    } else {
        no_views.clone()
    };

    Some(vec![
        w_mkt.iter().copied().collect(),
        vec![1.0 / k as f64; k],
        optimize(&no_views)?,
        optimize(&h_views)?,
        optimize(&c_views)?,
    ])
}

fn strategy_stats(
    returns: &[f64],
    benchmark: &[f64],
    turnover: &[f64],
    cfg: &BacktestConfig,
) -> Option<StrategyStats> {
    if returns.len() < 20 {
        return None;
    }
    let ann = crate::model::TRADING_DAYS;
    let mean = stats::mean(returns) * ann;
    let vol = stats::std_dev(returns) * ann.sqrt();
    let active: Vec<f64> = returns.iter().zip(benchmark).map(|(a, b)| a - b).collect();
    let active_mean = stats::mean(&active) * ann;
    let te = stats::std_dev(&active) * ann.sqrt();
    let years = returns.len() as f64 / ann;
    let ci = if te > 0.0 {
        stats::stationary_bootstrap_ci(active.len(), 10.0, 2000, 42, |idx| {
            idx.iter().map(|&i| active[i]).sum::<f64>() / idx.len() as f64 * ann
        })
    } else {
        (0.0, 0.0)
    };
    Some(StrategyStats {
        annual_return: mean,
        annual_volatility: vol,
        sharpe: if vol > 0.0 { mean / vol } else { 0.0 },
        max_drawdown: stats::max_drawdown(returns),
        avg_turnover: if turnover.is_empty() {
            0.0
        } else {
            stats::mean(turnover)
        },
        annual_cost: turnover.iter().sum::<f64>() * cfg.cost_bps / 1e4 / years,
        active_return: active_mean,
        tracking_error: te,
        information_ratio: if te > 0.0 { active_mean / te } else { 0.0 },
        active_return_ci: ci,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::prices::PriceHistory;
    use crate::research::panel::{FundamentalPanel, Panel};
    use chrono::NaiveDate;

    fn history(ticker: &str, drift: f64, wobble: f64) -> PriceHistory {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        PriceHistory {
            ticker: ticker.into(),
            name: None,
            currency: None,
            last_price: 1.0,
            closes: (0..400)
                .map(|k| {
                    let px = 100.0 * (drift * k as f64 + wobble * (k as f64 * 0.7).sin()).exp();
                    (start + chrono::Duration::days(k), px)
                })
                .collect(),
        }
    }

    fn config() -> BacktestConfig {
        BacktestConfig {
            rebalance_every: 21,
            cov_window: 252,
            max_weight: 0.6,
            risk_aversion: 2.5,
            tau: 0.05,
            cost_bps: 10.0,
            horizon: 21,
            min_calibration_dates: 126,
            calibration_lags: 25,
            min_headlines: 3,
            view_scale: 0.25,
            sentiment_weight: 0.7,
        }
    }

    #[test]
    fn runs_without_signals_and_keeps_strategies_consistent() {
        let bench = history("SPY", 0.0004, 0.01);
        let (a, b, c) = (
            history("A", 0.0005, 0.02),
            history("B", 0.0002, 0.015),
            history("C", 0.0007, 0.03),
        );
        let panel = Panel::new(&bench, &[&a, &b, &c]);
        let n = 3;
        let sentiment = vec![vec![None; panel.len()]; n];
        let fundamentals = FundamentalPanel {
            score: vec![vec![None; panel.len()]; n],
            market_cap: vec![vec![None; panel.len()]; n],
        };
        let inputs = Inputs {
            panel: &panel,
            sentiment: &sentiment,
            fundamentals: &fundamentals,
            sentiment_sections: &[],
            fundamental_sections: &[],
        };
        let res = run(&inputs, &config(), 0).expect("backtest runs");
        assert_eq!(res.dates.len(), panel.len() - 1 - 252);
        assert_eq!(
            res.market_cap_fallbacks, res.rebalances,
            "no caps -> equal-weight prior"
        );
        assert_eq!(res.calibrated_rebalances, 0);

        let by_name = |name: &str| res.strategies.iter().find(|s| s.name == name).unwrap();
        // Without market caps the prior is equal weight, so the two coincide.
        assert_eq!(
            by_name("Market cap").daily_returns,
            by_name("Equal weight").daily_returns
        );
        // No signals and no calibration: calibrated == heuristic == no views.
        assert_eq!(
            by_name("BL + calibrated views").daily_returns,
            by_name("BL, no views").daily_returns
        );
        assert_eq!(
            by_name("BL + heuristic views").daily_returns,
            by_name("BL, no views").daily_returns
        );
        // The initial build is excluded from turnover.
        assert_eq!(by_name("Equal weight").turnover.len(), res.rebalances - 1);
    }
}
