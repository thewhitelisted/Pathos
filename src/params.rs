//! User-tunable analysis parameters, shared by the CLI and the HTTP API.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::data::normalize_ticker;

pub const MAX_TICKERS: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisParams {
    pub tickers: Vec<String>,
    /// Cash to allocate, in the tickers' quote currency.
    pub budget: f64,
    /// Maximum weight in any single position.
    pub max_weight: f64,
    /// Calendar days of price history for the risk model.
    pub lookback_days: u32,
    /// Only headlines from the last N days are scored.
    pub news_days: u32,
    /// Maximum headlines scored per ticker.
    pub max_headlines: usize,
    /// Risk aversion δ, used for both the implied prior and the optimizer.
    pub risk_aversion: f64,
    /// Black-Litterman τ: uncertainty in the prior relative to Σ.
    pub tau: f64,
    /// κ: a signal of ±1 tilts the view by κ times the asset's volatility.
    pub view_scale: f64,
    /// Share of the signal from news sentiment (the rest from fundamentals).
    pub sentiment_weight: f64,
    /// Ignore any saved calibration and use the hand-tuned view mapping
    /// (`view_scale`, `sentiment_weight`).
    pub heuristic_views: bool,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            tickers: Vec::new(),
            budget: 10_000.0,
            max_weight: 0.35,
            lookback_days: 365,
            news_days: 7,
            max_headlines: 40,
            risk_aversion: 2.5,
            tau: 0.05,
            view_scale: 0.25,
            sentiment_weight: 0.7,
            heuristic_views: false,
        }
    }
}

impl AnalysisParams {
    /// Normalize tickers (upper-case, de-duplicated) and check ranges.
    pub fn validated(mut self) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        self.tickers = self
            .tickers
            .iter()
            .flat_map(|t| t.split([',', ' ']))
            .map(normalize_ticker)
            .filter(|t| !t.is_empty() && seen.insert(t.clone()))
            .collect();
        ensure!(self.tickers.len() >= 2, "enter at least two tickers");
        ensure!(
            self.tickers.len() <= MAX_TICKERS,
            "at most {MAX_TICKERS} tickers are supported"
        );
        ensure!(
            self.tickers.iter().all(|t| t.len() <= 12
                && t.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '^')),
            "tickers may only contain letters, digits, '-', '.' or '^'"
        );
        ensure!(
            self.budget.is_finite() && self.budget > 0.0,
            "budget must be positive"
        );
        ensure!(
            self.max_weight > 0.0 && self.max_weight <= 1.0,
            "max weight must be in (0, 1]"
        );
        ensure!(
            (60..=1825).contains(&self.lookback_days),
            "lookback must be 60–1825 days"
        );
        ensure!(
            (1..=30).contains(&self.news_days),
            "news window must be 1–30 days"
        );
        ensure!(
            (1..=100).contains(&self.max_headlines),
            "max headlines must be 1–100"
        );
        ensure!(
            self.risk_aversion > 0.0 && self.risk_aversion <= 50.0,
            "risk aversion must be in (0, 50]"
        );
        ensure!(self.tau > 0.0 && self.tau <= 1.0, "tau must be in (0, 1]");
        ensure!(
            (0.0..=2.0).contains(&self.view_scale),
            "view scale must be in [0, 2]"
        );
        ensure!(
            (0.0..=1.0).contains(&self.sentiment_weight),
            "sentiment weight must be in [0, 1]"
        );
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_dedupes() {
        let p = AnalysisParams {
            tickers: vec!["aapl, msft".into(), "AAPL".into(), "brk.b".into()],
            ..Default::default()
        }
        .validated()
        .unwrap();
        assert_eq!(p.tickers, ["AAPL", "MSFT", "BRK-B"]);
    }

    #[test]
    fn rejects_bad_input() {
        let one = AnalysisParams {
            tickers: vec!["AAPL".into()],
            ..Default::default()
        };
        assert!(one.validated().is_err());
        let bad = AnalysisParams {
            tickers: vec!["AAPL".into(), "../etc".into()],
            ..Default::default()
        };
        assert!(bad.validated().is_err());
    }
}
