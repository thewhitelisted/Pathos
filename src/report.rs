//! The analysis result, plus a plain-text rendering for the terminal.

use std::fmt::Write as _;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::data::sec::Fundamentals;
use crate::model::PortfolioStats;
use crate::model::calibration::Calibration;
use crate::model::signals::Signal;
use crate::params::AnalysisParams;
use crate::sentiment::{ScoredHeadline, SentimentSummary};

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub generated_at: DateTime<Utc>,
    pub params: AnalysisParams,
    pub assets: Vec<AssetReport>,
    pub portfolio: PortfolioSummary,
    pub benchmarks: Vec<Benchmark>,
    pub risk: RiskSummary,
    pub warnings: Vec<String>,
    pub timings_ms: Timings,
    /// "calibrated" (views estimated by `pathos evaluate`) or "heuristic".
    pub view_mode: String,
    pub calibration: Option<Calibration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssetReport {
    pub ticker: String,
    pub name: Option<String>,
    pub currency: Option<String>,
    pub price: f64,
    pub market_cap: Option<f64>,
    /// Weight in the market-cap (or equal-weight fallback) prior portfolio.
    pub market_weight: f64,
    /// Annualized volatility.
    pub volatility: f64,
    pub sentiment: SentimentSummary,
    pub fundamentals: Option<Fundamentals>,
    pub signal: Signal,
    /// Equilibrium implied excess return π.
    pub prior_return: f64,
    /// The view Q fed to Black-Litterman, if the signal was strong enough.
    pub view_return: Option<f64>,
    /// Posterior expected excess return μ_BL.
    pub posterior_return: f64,
    /// Optimal portfolio weight.
    pub weight: f64,
    pub target_value: f64,
    pub shares: u64,
    pub headlines: Vec<ScoredHeadline>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PortfolioSummary {
    #[serde(flatten)]
    pub stats: PortfolioStats,
    pub invested: f64,
    pub cash_left: f64,
    pub optimizer_iterations: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Benchmark {
    pub name: String,
    pub weights: Vec<f64>,
    #[serde(flatten)]
    pub stats: PortfolioStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskSummary {
    pub shrinkage: f64,
    pub observations: usize,
    pub correlation: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Timings {
    pub data: u128,
    pub sentiment: u128,
    pub model: u128,
}

fn pct(x: f64) -> String {
    format!("{:+.1}%", x * 100.0)
}

impl Report {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "\n{:<7} {:>9} {:>10} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>6}",
            "Ticker",
            "Price",
            "Sentiment",
            "News",
            "Signal",
            "Prior",
            "Post.",
            "Mkt Wt",
            "Weight",
            "Value",
            "Shares"
        );
        let _ = writeln!(s, "{}", "-".repeat(101));
        let mut assets: Vec<&AssetReport> = self.assets.iter().collect();
        assets.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        for a in assets {
            let _ = writeln!(
                s,
                "{:<7} {:>9.2} {:>10} {:>6} {:>8} {:>8} {:>8} {:>7.1}% {:>7.1}% {:>10.2} {:>6}",
                a.ticker,
                a.price,
                format!("{:+.2}", a.sentiment.score),
                a.sentiment.headlines,
                format!("{:+.2}", a.signal.value),
                pct(a.prior_return),
                pct(a.posterior_return),
                a.market_weight * 100.0,
                a.weight * 100.0,
                a.target_value,
                a.shares,
            );
        }
        let p = &self.portfolio;
        let _ = writeln!(
            s,
            "\nExpected excess return {}  ·  Volatility {:.1}%  ·  Sharpe {:.2}",
            pct(p.stats.expected_return),
            p.stats.volatility * 100.0,
            p.stats.sharpe
        );
        let _ = writeln!(
            s,
            "Invested {:.2}  ·  Cash left {:.2}",
            p.invested, p.cash_left
        );
        let _ = writeln!(s, "\nBenchmarks (under the posterior):");
        for b in &self.benchmarks {
            let _ = writeln!(
                s,
                "  {:<26} return {:>7}  vol {:>5.1}%  Sharpe {:>5.2}",
                b.name,
                pct(b.stats.expected_return),
                b.stats.volatility * 100.0,
                b.stats.sharpe
            );
        }
        match &self.calibration {
            Some(c) if self.view_mode == "calibrated" => {
                let _ = writeln!(s, "\nViews: calibrated — {}", c.summary());
            }
            _ => {
                let _ = writeln!(s, "\nViews: heuristic (run `pathos evaluate` to calibrate)");
            }
        }
        let _ = writeln!(
            s,
            "Risk model: {} daily observations, Ledoit-Wolf shrinkage {:.2}",
            self.risk.observations, self.risk.shrinkage
        );
        if !self.warnings.is_empty() {
            let _ = writeln!(s, "\nWarnings:");
            for w in &self.warnings {
                let _ = writeln!(s, "  - {w}");
            }
        }
        s
    }
}
