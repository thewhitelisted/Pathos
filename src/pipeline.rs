//! End-to-end analysis: fetch → score → model → optimize → report.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use futures::future::join_all;
use nalgebra::DVector;

use crate::data::news::{self, Headline};
use crate::data::prices::{self, PriceHistory};
use crate::data::sec::{self, Fundamentals};
use crate::http::Fetcher;
use crate::model::calibration::{Calibration, calibrated_views};
use crate::model::{black_litterman, covariance, optimizer, portfolio_stats, signals};
use crate::params::AnalysisParams;
use crate::report::{AssetReport, Benchmark, PortfolioSummary, Report, RiskSummary, Timings};
use crate::sentiment::store::ScoreStore;
use crate::sentiment::{self, Probs, ScoredHeadline, SentimentModel};

pub struct Analyzer {
    fetcher: Fetcher,
    model: Arc<dyn SentimentModel>,
    /// Headline text -> FinBERT output, persisted across runs. Inference
    /// dominates runtime and the same headlines recur, so repeat analyses are
    /// near-instant.
    scores: Arc<ScoreStore>,
    /// Signal-to-view mapping estimated by `pathos evaluate`, if available.
    calibration: Option<Calibration>,
}

struct TickerData {
    ticker: String,
    prices: PriceHistory,
    fundamentals: Option<Fundamentals>,
    headlines: Vec<Headline>,
}

impl Analyzer {
    pub fn new(fetcher: Fetcher, model: Arc<dyn SentimentModel>, scores: Arc<ScoreStore>) -> Self {
        Self {
            fetcher,
            model,
            scores,
            calibration: None,
        }
    }

    pub fn with_calibration(mut self, calibration: Option<Calibration>) -> Self {
        self.calibration = calibration;
        self
    }

    /// Score texts, running the model only on ones not seen before.
    async fn score(&self, texts: Vec<String>) -> Result<Vec<Probs>> {
        let model = Arc::clone(&self.model);
        let scores = Arc::clone(&self.scores);
        tokio::task::spawn_blocking(move || scores.score(model.as_ref(), &texts, &|_, _| {}))
            .await
            .context("sentiment worker panicked")?
    }

    pub async fn analyze(&self, params: AnalysisParams) -> Result<Report> {
        let params = params.validated()?;
        let mut warnings = Vec::new();

        // ---- 1. Data -------------------------------------------------------
        let t0 = Instant::now();
        let cik_map = match sec::ticker_map(&self.fetcher).await {
            Ok(m) => m,
            Err(e) => {
                warnings.push(format!("fundamentals and market caps unavailable — {e:#}"));
                HashMap::new()
            }
        };
        let fetched = join_all(
            params
                .tickers
                .iter()
                .map(|t| self.fetch_ticker(t, &cik_map, &params)),
        )
        .await;

        let mut data = Vec::new();
        for (ticker, result) in params.tickers.iter().zip(fetched) {
            match result {
                Ok((d, mut w)) => {
                    warnings.append(&mut w);
                    data.push(d);
                }
                Err(e) => warnings.push(format!("{ticker}: dropped — {e:#}")),
            }
        }
        if data.len() < 2 {
            bail!(
                "need price data for at least two tickers; {}",
                warnings.join("; ")
            );
        }
        let data_ms = t0.elapsed().as_millis();

        // ---- 2. Sentiment ---------------------------------------------------
        let t1 = Instant::now();
        let texts: Vec<String> = data
            .iter()
            .flat_map(|d| d.headlines.iter().map(|h| h.title.clone()))
            .collect();
        let mut probs = self.score(texts).await?.into_iter();
        let now = Utc::now();
        let scored: Vec<Vec<ScoredHeadline>> = data
            .iter()
            .map(|d| {
                d.headlines
                    .iter()
                    .zip(probs.by_ref())
                    .map(|(h, p)| ScoredHeadline {
                        headline: h.clone(),
                        probs: p,
                        score: p.score(),
                    })
                    .collect()
            })
            .collect();
        let summaries: Vec<_> = scored
            .iter()
            .map(|s| sentiment::summarize(s, now))
            .collect();
        let sentiment_ms = t1.elapsed().as_millis();

        // ---- 3. Risk model & prior -------------------------------------------
        let t2 = Instant::now();
        let n = data.len();
        let histories: Vec<&PriceHistory> = data.iter().map(|d| &d.prices).collect();
        let returns = prices::aligned_log_returns(&histories);
        let risk = covariance::ledoit_wolf(&returns)?;
        let vols = risk.volatilities();

        let caps: Vec<Option<f64>> = data
            .iter()
            .map(|d| {
                let f = d.fundamentals.as_ref()?;
                f.shares_outstanding.map(|s| {
                    prices::market_cap(d.prices.last_price, s, f.shares_as_of, &d.prices.splits)
                })
            })
            .collect();
        let w_mkt = if caps.iter().all(Option::is_some) {
            let caps: Vec<f64> = caps.iter().flatten().copied().collect();
            let total: f64 = caps.iter().sum();
            DVector::from_iterator(n, caps.iter().map(|c| c / total))
        } else {
            let missing: Vec<&str> = data
                .iter()
                .zip(&caps)
                .filter(|(_, c)| c.is_none())
                .map(|(d, _)| d.ticker.as_str())
                .collect();
            warnings.push(format!(
                "no share count for {}; using an equal-weight prior instead of market caps",
                missing.join(", ")
            ));
            DVector::from_element(n, 1.0 / n as f64)
        };
        let pi = black_litterman::implied_returns(&risk.sigma, &w_mkt, params.risk_aversion);

        // ---- 4. Views & posterior --------------------------------------------
        let mut sigs: Vec<signals::Signal> = data
            .iter()
            .zip(&summaries)
            .map(|(d, s)| signals::combine(s, d.fundamentals.as_ref(), params.sentiment_weight))
            .collect();
        let calibration = self
            .calibration
            .as_ref()
            .filter(|_| !params.heuristic_views);
        let (views, view_returns) = match calibration {
            Some(cal) => {
                let sentiment: Vec<Option<&sentiment::SentimentSummary>> = summaries
                    .iter()
                    .map(|s| (s.headlines >= cal.min_headlines).then_some(s))
                    .collect();
                let fund: Vec<Option<f64>> = sigs.iter().map(|s| s.fundamental_score).collect();
                let (views, q) = calibrated_views(&pi, &vols, &sentiment, &fund, cal);
                // Display the calibrated tilt (alpha per unit of volatility) and
                // the weight Black-Litterman gives the view vs the prior.
                for (k, row) in views.p.row_iter().enumerate() {
                    let i = row.iter().position(|&v| v == 1.0).unwrap_or(0);
                    let prior_var = params.tau * vols[i].powi(2);
                    sigs[i].value = (views.q[k] - pi[i]) / vols[i];
                    sigs[i].confidence = prior_var / (prior_var + views.omega[k]);
                }
                for (i, s) in sigs.iter_mut().enumerate() {
                    if q[i].is_none() {
                        s.value = 0.0;
                        s.confidence = 0.0;
                    }
                }
                (views, q)
            }
            None => signals::build_views(&pi, &vols, &sigs, params.tau, params.view_scale),
        };
        if views.p.nrows() == 0 {
            warnings.push(
                "no ticker had a confident enough signal; the result is the market prior".into(),
            );
        }
        let post = black_litterman::posterior(&risk.sigma, &pi, params.tau, &views)?;

        // ---- 5. Optimize ---------------------------------------------------
        let mut cap = params.max_weight;
        if cap * (n as f64) < 1.0 {
            cap = 1.0 / n as f64;
            warnings.push(format!(
                "max weight {:.0}% is infeasible for {n} assets; raised to {:.1}%",
                params.max_weight * 100.0,
                cap * 100.0
            ));
        }
        let solution = optimizer::optimize(&post.mu, &post.sigma, params.risk_aversion, cap)?;
        if !solution.converged {
            warnings.push(
                "optimizer hit its iteration limit; weights may be slightly suboptimal".into(),
            );
        }
        let w = solution.weights.map(|x| if x < 1e-6 { 0.0 } else { x });
        let w = &w / w.sum();

        let stats = |weights: &DVector<f64>| portfolio_stats(weights, &post.mu, &post.sigma);
        let no_views = optimizer::optimize(&pi, &post.sigma, params.risk_aversion, cap)?.weights;
        let equal = DVector::from_element(n, 1.0 / n as f64);
        let benchmarks = vec![
            ("Pathos (with views)", &w),
            ("Capped optimum, no views", &no_views),
            ("Market-cap prior", &w_mkt),
            ("Equal weight", &equal),
        ]
        .into_iter()
        .map(|(name, weights)| Benchmark {
            name: name.into(),
            weights: weights.iter().copied().collect(),
            stats: stats(weights),
        })
        .collect();
        let model_ms = t2.elapsed().as_millis();

        // ---- 6. Report -----------------------------------------------------
        let price_vec: Vec<f64> = data.iter().map(|d| d.prices.last_price).collect();
        let shares = round_to_shares(params.budget, w.as_slice(), &price_vec);
        let invested: f64 = shares
            .iter()
            .zip(&price_vec)
            .map(|(&s, p)| s as f64 * p)
            .sum();
        let assets: Vec<AssetReport> = data
            .into_iter()
            .zip(scored)
            .zip(summaries)
            .enumerate()
            .map(|(i, ((d, headlines), sentiment))| {
                let price = d.prices.last_price;
                AssetReport {
                    name: d
                        .prices
                        .name
                        .clone()
                        .or_else(|| d.fundamentals.as_ref()?.entity_name.clone()),
                    currency: d.prices.currency.clone(),
                    ticker: d.ticker,
                    price,
                    market_cap: caps[i],
                    market_weight: w_mkt[i],
                    volatility: vols[i],
                    sentiment,
                    fundamentals: d.fundamentals,
                    signal: sigs[i],
                    prior_return: pi[i],
                    view_return: view_returns[i],
                    posterior_return: post.mu[i],
                    weight: w[i],
                    target_value: params.budget * w[i],
                    shares: shares[i],
                    headlines,
                }
            })
            .collect();

        Ok(Report {
            generated_at: now,
            portfolio: PortfolioSummary {
                stats: stats(&w),
                invested,
                cash_left: params.budget - invested,
                optimizer_iterations: solution.iterations,
            },
            params,
            assets,
            benchmarks,
            risk: RiskSummary {
                shrinkage: risk.shrinkage,
                observations: risk.observations,
                correlation: risk.correlation(),
            },
            view_mode: if calibration.is_some() {
                "calibrated"
            } else {
                "heuristic"
            }
            .into(),
            calibration: self.calibration.clone(),
            warnings,
            timings_ms: Timings {
                data: data_ms,
                sentiment: sentiment_ms,
                model: model_ms,
            },
        })
    }

    async fn fetch_ticker(
        &self,
        ticker: &str,
        cik_map: &HashMap<String, String>,
        params: &AnalysisParams,
    ) -> Result<(TickerData, Vec<String>)> {
        let mut warnings = Vec::new();
        let prices = prices::fetch_history(&self.fetcher, ticker, params.lookback_days).await?;

        let fundamentals = async {
            let cik = sec::lookup_cik(cik_map, ticker)?;
            sec::fetch_fundamentals(&self.fetcher, &cik).await
        };
        let headlines = news::fetch_headlines(
            &self.fetcher,
            ticker,
            prices.name.as_deref(),
            params.news_days,
            params.max_headlines,
        );
        let (fundamentals, headlines) = tokio::join!(fundamentals, headlines);

        let fundamentals = match fundamentals {
            Ok(f) => Some(f),
            // SEC as a whole is unavailable; already reported once.
            Err(_) if cik_map.is_empty() => None,
            Err(e) => {
                warnings.push(format!("{ticker}: no fundamentals — {e:#}"));
                None
            }
        };
        if headlines.is_empty() {
            warnings.push(format!(
                "{ticker}: no headlines in the last {} days",
                params.news_days
            ));
        }
        Ok((
            TickerData {
                ticker: ticker.to_string(),
                prices,
                fundamentals,
                headlines,
            },
            warnings,
        ))
    }
}

/// Whole-share quantities closest to the target weights within `budget`.
///
/// Start from the floor of each target, then greedily spend leftover cash on
/// the position furthest below target, as long as one more share brings it
/// closer to its target than leaving it short.
pub fn round_to_shares(budget: f64, weights: &[f64], prices: &[f64]) -> Vec<u64> {
    let targets: Vec<f64> = weights.iter().map(|w| w * budget).collect();
    let mut shares: Vec<u64> = targets
        .iter()
        .zip(prices)
        .map(|(t, p)| (t / p).floor().max(0.0) as u64)
        .collect();
    let mut cash = budget
        - shares
            .iter()
            .zip(prices)
            .map(|(&s, p)| s as f64 * p)
            .sum::<f64>();
    loop {
        let best = (0..prices.len())
            .filter(|&i| prices[i] <= cash)
            .map(|i| (i, targets[i] - shares[i] as f64 * prices[i]))
            .filter(|&(i, shortfall)| shortfall > prices[i] / 2.0)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, _)) => {
                shares[i] += 1;
                cash -= prices[i];
            }
            None => return shares,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_never_overspends_and_tracks_targets() {
        let prices = [333.0, 241.6, 531.9, 380.7];
        let weights = [0.3, 0.3, 0.2, 0.2];
        let shares = round_to_shares(5_000.0, &weights, &prices);
        let spent: f64 = shares.iter().zip(&prices).map(|(&s, p)| s as f64 * p).sum();
        assert!(spent <= 5_000.0);
        for ((s, p), w) in shares.iter().zip(&prices).zip(&weights) {
            assert!(
                (*s as f64 * p - w * 5_000.0).abs() <= *p,
                "{s} shares at {p}"
            );
        }
        // Rounding up beats plain floor here (floor leaves > $500 idle).
        assert!(5_000.0 - spent < 500.0);
    }
}
