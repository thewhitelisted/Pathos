//! `pathos evaluate`: does the signal actually predict returns?

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use futures::future::join_all;
use serde::Serialize;

use super::archive::{self, DatedHeadline};
use super::backtest::{self, BacktestConfig, BacktestResult, CrossSection};
use super::panel::{self, Panel};
use super::stats::{self, MeanEstimate};
use crate::data::prices::{self, PriceHistory};
use crate::data::sec::{self, CompanyFacts};
use crate::http::Fetcher;
use crate::model::calibration::Calibration;
use crate::params::AnalysisParams;
use crate::sentiment::store::ScoreStore;
use crate::sentiment::{Probs, SentimentModel};

/// Liquid large caps across sectors: enough breadth for cross-sectional
/// statistics while keeping collection and scoring time reasonable.
pub const DEFAULT_UNIVERSE: &[&str] = &[
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "NFLX", "JPM", "BAC",
    "GS", "XOM", "CVX", "JNJ", "PFE", "UNH", "PG", "KO", "PEP", "WMT", "COST", "DIS",
];
/// Small and mid caps (roughly $0.2B–$10B) across consumer, technology,
/// health care, clean energy, fintech and industrials, each an SEC registrant
/// with at least 3.3 years of price history as of October 2026.
pub const SMALL_CAP_UNIVERSE: &[&str] = &[
    "CROX", "BOOT", "SHAK", "WING", "ELF", "AEO", "URBN", "PTON", "BYND", "FIGS", "FSLY", "UPST",
    "AI", "PATH", "BB", "RGTI", "AEHR", "SOUN", "HIMS", "NVAX", "RXRX", "TDOC", "GERN", "PLUG",
    "RUN", "FCEL", "LMND", "OPEN", "JOBY", "RIOT", "MARA", "AMC", "FUBO", "GOGO",
];
/// Days of news aggregated into one signal (matches the live default).
const NEWS_WINDOW_DAYS: i64 = 7;
/// Signals persist for roughly this many trading days; used in HAC lags.
const NEWS_PERSISTENCE: usize = 5;
const FUNDAMENTAL_PERSISTENCE: usize = 21;
const VOL_WINDOW: usize = 63;
const MIN_NAMES: usize = 5;

/// `[ticker][date]` signal values.
type SignalGrid = Vec<Vec<Option<f64>>>;

#[derive(Debug, Clone, Serialize)]
pub struct EvalParams {
    pub tickers: Vec<String>,
    /// Market proxy for excess returns (e.g. SPY for large caps, IWM for small).
    pub benchmark: String,
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub per_week: usize,
    pub headlines_csv: Option<PathBuf>,
    pub horizons: Vec<usize>,
    pub calibration_horizon: usize,
    pub min_headlines: usize,
    pub rebalance_every: usize,
    pub cost_bps: f64,
    pub max_weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IcRow {
    pub signal: String,
    pub horizon: usize,
    #[serde(flatten)]
    pub ic: MeanEstimate,
    /// Share of dates with a positive IC.
    pub hit_rate: f64,
    pub avg_names: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Coverage {
    pub headlines: usize,
    pub headlines_per_ticker: Vec<(String, usize)>,
    /// Share of (ticker, date) pairs in the evaluation window with a signal.
    pub signal_coverage: f64,
    pub newly_scored: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    pub generated_at: DateTime<Utc>,
    pub params: EvalParams,
    pub universe: Vec<String>,
    pub coverage: Coverage,
    pub ic: Vec<IcRow>,
    pub calibration: Calibration,
    pub backtest: Option<BacktestResult>,
    pub warnings: Vec<String>,
    pub elapsed_secs: u64,
}

pub async fn evaluate(
    fetcher: &Fetcher,
    model: Arc<dyn SentimentModel>,
    store: Arc<ScoreStore>,
    params: EvalParams,
) -> Result<EvaluationReport> {
    let started = Instant::now();
    let mut warnings = Vec::new();
    if params.end <= params.start {
        bail!("evaluation end must be after start");
    }

    // ---- prices ----------------------------------------------------------
    // Extra history before `start` for the covariance window.
    let lookback = (params.end - params.start).num_days() as u32 + 420;
    tracing::info!(
        "fetching prices for {} tickers + {}",
        params.tickers.len(),
        params.benchmark
    );
    let bench = prices::fetch_history(fetcher, &params.benchmark, lookback)
        .await
        .context("benchmark prices")?;
    let fetched = join_all(
        params
            .tickers
            .iter()
            .map(|t| prices::fetch_history(fetcher, t, lookback)),
    )
    .await;
    let mut histories: Vec<PriceHistory> = Vec::new();
    for (t, h) in params.tickers.iter().zip(fetched) {
        match h {
            Ok(h) => histories.push(h),
            Err(e) => warnings.push(format!("{t}: dropped — {e:#}")),
        }
    }
    if histories.len() < MIN_NAMES {
        bail!("need prices for at least {MIN_NAMES} tickers");
    }
    let universe: Vec<String> = histories.iter().map(|h| h.ticker.clone()).collect();

    // ---- headlines ---------------------------------------------------------
    let headlines: Vec<DatedHeadline> = match &params.headlines_csv {
        Some(path) => {
            tracing::info!("loading headlines from {}", path.display());
            archive::load_csv(path, &universe)?
                .into_iter()
                .filter(|h| h.date >= params.start && h.date < params.end)
                .collect()
        }
        None => {
            let mut all = Vec::new();
            for (k, h) in histories.iter().enumerate() {
                let got = archive::collect_google_news(
                    fetcher,
                    &h.ticker,
                    h.name.as_deref(),
                    params.start,
                    params.end,
                    params.per_week,
                )
                .await;
                tracing::info!(
                    "[{}/{}] {}: {} headlines",
                    k + 1,
                    histories.len(),
                    h.ticker,
                    got.len()
                );
                all.extend(got);
            }
            all
        }
    };
    if headlines.is_empty() {
        bail!("no headlines found for the evaluation period");
    }

    // ---- sentiment scoring (persistent cache) ------------------------------
    let texts: Vec<String> = headlines.iter().map(|h| h.title.clone()).collect();
    let newly_scored = store.missing(&texts).len();
    if newly_scored > 0 {
        tracing::info!(
            "scoring {newly_scored} new headlines with FinBERT (cached for future runs)"
        );
    }
    let probs = {
        let store = Arc::clone(&store);
        let t0 = Instant::now();
        tokio::task::spawn_blocking(move || {
            store.score(model.as_ref(), &texts, &|done, total| {
                let rate = done as f64 / t0.elapsed().as_secs_f64().max(1e-3);
                let eta = (total - done) as f64 / rate.max(1e-3);
                tracing::info!(
                    "  scored {done}/{total} ({rate:.0}/s, ~{:.0} min left)",
                    eta / 60.0
                );
            })
        })
        .await??
    };

    // ---- point-in-time fundamentals ------------------------------------------
    let facts: Vec<Option<(String, CompanyFacts)>> = match sec::ticker_map(fetcher).await {
        Ok(map) => {
            join_all(universe.iter().map(|t| async {
                let cik = sec::lookup_cik(&map, t).ok()?;
                let cf = sec::fetch_company_facts(fetcher, &cik).await.ok()?;
                Some((cik, cf))
            }))
            .await
        }
        Err(e) => {
            warnings.push(format!("fundamentals and market caps unavailable — {e:#}"));
            vec![None; universe.len()]
        }
    };

    // ---- panels ------------------------------------------------------------
    let refs: Vec<&PriceHistory> = histories.iter().collect();
    let panel = Panel::new(&bench, &refs);
    let index: HashMap<&str, usize> = universe
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let mut by_ticker: Vec<Vec<(NaiveDate, Probs)>> = vec![Vec::new(); universe.len()];
    for (h, p) in headlines.iter().zip(&probs) {
        if let Some(&i) = index.get(h.ticker.as_str()) {
            by_ticker[i].push((h.date, *p));
        }
    }
    let sentiment =
        panel::sentiment_panel(&panel, &by_ticker, NEWS_WINDOW_DAYS, params.min_headlines);
    let fundamentals = panel::fundamental_panel(&panel, &facts);

    let t_start = panel
        .dates
        .iter()
        .position(|d| *d >= params.start + chrono::Duration::days(NEWS_WINDOW_DAYS))
        .context("evaluation start is after the last price")?;
    let t_end = panel
        .dates
        .iter()
        .rposition(|d| *d < params.end)
        .unwrap_or(panel.len() - 1);
    let n = universe.len();

    // ---- signal variants -----------------------------------------------------
    let level: Vec<Vec<Option<f64>>> = sentiment
        .iter()
        .map(|row| row.iter().map(|s| s.as_ref().map(|s| s.score)).collect())
        .collect();
    let surprise = surprise(&level, 60, 20);
    let ex_momentum = ex_momentum(&panel, &level, 5);
    let fund: Vec<Vec<Option<f64>>> = fundamentals
        .score
        .iter()
        .map(|row| row.iter().map(|f| f.map(|(s, _)| s)).collect())
        .collect();
    let variants: [(&str, &SignalGrid, usize); 4] = [
        ("News sentiment (level)", &level, NEWS_PERSISTENCE),
        (
            "News sentiment (surprise vs 60d)",
            &surprise,
            NEWS_PERSISTENCE,
        ),
        (
            "News sentiment ex 5d momentum",
            &ex_momentum,
            NEWS_PERSISTENCE,
        ),
        ("Fundamentals quality", &fund, FUNDAMENTAL_PERSISTENCE),
    ];

    let mut ic = Vec::new();
    for (name, sig, persistence) in variants {
        for &h in &params.horizons {
            if let Some(row) = ic_row(&panel, sig, h, t_start, t_end, persistence) {
                ic.push(IcRow {
                    signal: name.to_string(),
                    ..row
                });
            }
        }
    }

    // ---- calibration ---------------------------------------------------------
    let h = params.calibration_horizon;
    let lags = h - 1 + NEWS_PERSISTENCE;
    let sent_sections = cross_sections(&panel, &level, h, t_start, t_end);
    let fund_sections = cross_sections(&panel, &fund, h, t_start, t_end);
    let est =
        |s: &[CrossSection]| backtest::estimate(s, h, None, lags).map(backtest::to_coefficient);
    let first_t = sent_sections.first().map_or(t_start, |s| s.t);
    let last_t = sent_sections.last().map_or(t_end, |s| s.t);
    let calibration = Calibration {
        generated_at: Utc::now(),
        horizon_days: h,
        sample_start: panel.dates[first_t],
        sample_end: panel.dates[last_t],
        universe: universe.clone(),
        min_headlines: params.min_headlines,
        sentiment: est(&sent_sections),
        fundamentals: est(&fund_sections),
    };

    // ---- walk-forward backtest -----------------------------------------------
    let defaults = AnalysisParams::default();
    let cfg = BacktestConfig {
        rebalance_every: params.rebalance_every,
        cov_window: 252,
        max_weight: params.max_weight,
        risk_aversion: defaults.risk_aversion,
        tau: defaults.tau,
        cost_bps: params.cost_bps,
        horizon: h,
        min_calibration_dates: 126,
        calibration_lags: lags,
        min_headlines: params.min_headlines,
        view_scale: defaults.view_scale,
        sentiment_weight: defaults.sentiment_weight,
    };
    let inputs = backtest::Inputs {
        panel: &panel,
        sentiment: &sentiment,
        fundamentals: &fundamentals,
        sentiment_sections: &sent_sections,
        fundamental_sections: &fund_sections,
    };
    let bt = backtest::run(&inputs, &cfg, t_start);
    if bt.is_none() {
        warnings.push("evaluation window too short for a backtest".into());
    }

    // ---- coverage --------------------------------------------------------------
    let covered = (0..n)
        .flat_map(|i| (t_start..=t_end).map(move |t| (i, t)))
        .filter(|&(i, t)| level[i][t].is_some())
        .count();
    let cells = n * (t_end + 1 - t_start);
    let coverage = Coverage {
        headlines: headlines.len(),
        headlines_per_ticker: universe
            .iter()
            .zip(&by_ticker)
            .map(|(t, h)| (t.clone(), h.len()))
            .collect(),
        signal_coverage: covered as f64 / cells.max(1) as f64,
        newly_scored,
    };
    if coverage.signal_coverage < 0.3 {
        warnings.push(format!(
            "only {:.0}% of ticker-days have ≥{} headlines; consider a denser headline source (--headlines-csv)",
            coverage.signal_coverage * 100.0,
            params.min_headlines
        ));
    }

    Ok(EvaluationReport {
        generated_at: Utc::now(),
        params,
        universe,
        coverage,
        ic,
        calibration,
        backtest: bt,
        warnings,
        elapsed_secs: started.elapsed().as_secs(),
    })
}

/// Level minus the ticker's own trailing mean: is today's tone unusual?
fn surprise(level: &[Vec<Option<f64>>], window: usize, min_obs: usize) -> Vec<Vec<Option<f64>>> {
    level
        .iter()
        .map(|row| {
            (0..row.len())
                .map(|t| {
                    let x = row[t]?;
                    let past: Vec<f64> = row[t.saturating_sub(window)..t]
                        .iter()
                        .flatten()
                        .copied()
                        .collect();
                    (past.len() >= min_obs).then(|| x - stats::mean(&past))
                })
                .collect()
        })
        .collect()
}

/// Residual of a per-date cross-sectional regression of the signal on the
/// trailing `lookback`-day excess return: the part of sentiment that is not
/// just a description of recent price moves.
fn ex_momentum(
    panel: &Panel,
    level: &[Vec<Option<f64>>],
    lookback: usize,
) -> Vec<Vec<Option<f64>>> {
    let n = level.len();
    let mut out = vec![vec![None; panel.len()]; n];
    for t in lookback..panel.len() {
        let pts: Vec<(usize, f64, f64)> = (0..n)
            .filter_map(|i| Some((i, panel.excess_return(i, t - lookback, t)?, level[i][t]?)))
            .collect();
        if pts.len() < MIN_NAMES {
            continue;
        }
        let xs: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.2).collect();
        let (mx, my) = (stats::mean(&xs), stats::mean(&ys));
        let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let beta = if sxx > 0.0 {
            xs.iter()
                .zip(&ys)
                .map(|(x, y)| (x - mx) * (y - my))
                .sum::<f64>()
                / sxx
        } else {
            0.0
        };
        for (i, x, y) in pts {
            out[i][t] = Some(y - my - beta * (x - mx));
        }
    }
    out
}

fn ic_row(
    panel: &Panel,
    sig: &[Vec<Option<f64>>],
    h: usize,
    t_start: usize,
    t_end: usize,
    persistence: usize,
) -> Option<IcRow> {
    let mut ics = Vec::new();
    let mut names = 0usize;
    for t in t_start..=t_end {
        let pairs: Vec<(f64, f64)> = (0..sig.len())
            .filter_map(|i| Some((sig[i][t]?, panel.excess_return(i, t, t + h)?)))
            .collect();
        if pairs.len() < MIN_NAMES {
            continue;
        }
        let (x, y): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
        let v = stats::spearman(&x, &y);
        if v.is_finite() {
            ics.push(v);
            names += x.len();
        }
    }
    if ics.len() < 20 {
        return None;
    }
    Some(IcRow {
        signal: String::new(),
        horizon: h,
        hit_rate: ics.iter().filter(|v| **v > 0.0).count() as f64 / ics.len() as f64,
        avg_names: names as f64 / ics.len() as f64,
        ic: stats::mean_with_nw(&ics, h - 1 + persistence),
    })
}

/// Per-date demeaned signal vs vol-scaled, annualized forward excess return.
fn cross_sections(
    panel: &Panel,
    sig: &[Vec<Option<f64>>],
    h: usize,
    t_start: usize,
    t_end: usize,
) -> Vec<CrossSection> {
    let ann = crate::model::TRADING_DAYS / h as f64;
    (t_start..=t_end)
        .filter_map(|t| {
            let pts: Vec<(f64, f64)> = (0..sig.len())
                .filter_map(|i| {
                    let x = sig[i][t]?;
                    let r = panel.excess_return(i, t, t + h)?;
                    let vol = panel.realized_vol(i, t, VOL_WINDOW).filter(|v| *v > 0.0)?;
                    Some((x, r * ann / vol))
                })
                .collect();
            if pts.len() < MIN_NAMES {
                return None;
            }
            let mx = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
            let my = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
            Some(CrossSection {
                t,
                x: pts.iter().map(|p| p.0 - mx).collect(),
                y: pts.iter().map(|p| p.1 - my).collect(),
            })
        })
        .collect()
}

fn pct(x: f64) -> String {
    format!("{:+.2}%", x * 100.0)
}

impl EvaluationReport {
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let p = &self.params;
        let _ = writeln!(
            s,
            "\nPathos evaluation · {} to {} · {} tickers vs {} · {} headlines ({:.0}% of ticker-days with a signal)",
            p.start,
            p.end,
            self.universe.len(),
            p.benchmark,
            self.coverage.headlines,
            self.coverage.signal_coverage * 100.0
        );

        let _ = writeln!(
            s,
            "\nInformation coefficient (Spearman, signal vs forward market-excess return)"
        );
        let _ = writeln!(
            s,
            "{:<36} {:>4} {:>8} {:>7} {:>6} {:>6} {:>6}",
            "Signal", "h", "Mean IC", "t (NW)", "Hit %", "Dates", "Names"
        );
        let _ = writeln!(s, "{}", "-".repeat(79));
        for r in &self.ic {
            let star = if r.ic.t.abs() >= 1.96 { "*" } else { " " };
            let _ = writeln!(
                s,
                "{:<36} {:>4} {:>+8.4} {:>+6.2}{} {:>5.1}% {:>6} {:>6.1}",
                r.signal,
                r.horizon,
                r.ic.mean,
                r.ic.t,
                star,
                r.hit_rate * 100.0,
                r.ic.n,
                r.avg_names
            );
        }
        let _ = writeln!(
            s,
            "  * |t| ≥ 1.96 (5% two-sided), Newey-West SEs with lag h − 1 + signal persistence"
        );

        let _ = writeln!(
            s,
            "\nCalibration ({}-day horizon, Driscoll-Kraay SEs)",
            self.calibration.horizon_days
        );
        for (name, c) in [
            ("News sentiment", &self.calibration.sentiment),
            ("Fundamentals", &self.calibration.fundamentals),
        ] {
            match c {
                Some(c) => {
                    let _ = writeln!(
                        s,
                        "  {:<15} κ = {:+.3} ± {:.3}  (t = {:+.2}, {} dates, {} obs) → used as {:+.3} after shrinkage",
                        name,
                        c.kappa,
                        c.se,
                        c.t,
                        c.dates,
                        c.observations,
                        c.shrunk().0
                    );
                }
                None => {
                    let _ = writeln!(s, "  {name:<15} not estimable (insufficient data)");
                }
            }
        }

        if let Some(bt) = &self.backtest {
            let c = &bt.config;
            let _ = writeln!(
                s,
                "\nWalk-forward backtest {} to {} · rebalance every {} days · {} bps costs · max weight {:.0}%",
                bt.start,
                bt.end,
                c.rebalance_every,
                c.cost_bps,
                c.max_weight * 100.0
            );
            let _ = writeln!(
                s,
                "{:<22} {:>8} {:>7} {:>6} {:>7} {:>7} {:>8} {:>6} {:>19}",
                "Strategy",
                "Return",
                "Vol",
                "Sharpe",
                "MaxDD",
                "Turn.",
                "Active",
                "IR",
                "Active 95% CI"
            );
            let _ = writeln!(s, "{}", "-".repeat(99));
            for st in &bt.strategies {
                if let Some(x) = &st.stats {
                    let _ = writeln!(
                        s,
                        "{:<22} {:>8} {:>6.1}% {:>6.2} {:>6.1}% {:>6.1}% {:>8} {:>6.2} {:>19}",
                        st.name,
                        pct(x.annual_return),
                        x.annual_volatility * 100.0,
                        x.sharpe,
                        x.max_drawdown * 100.0,
                        x.avg_turnover * 100.0,
                        pct(x.active_return),
                        x.information_ratio,
                        format!(
                            "[{}, {}]",
                            pct(x.active_return_ci.0),
                            pct(x.active_return_ci.1)
                        ),
                    );
                }
            }
            let _ = writeln!(
                s,
                "  Active = vs market cap. Calibrated views active on {}/{} rebalances (κ needs {} realized dates first).",
                bt.calibrated_rebalances, bt.rebalances, c.min_calibration_dates
            );
        }

        if !self.warnings.is_empty() {
            let _ = writeln!(s, "\nWarnings:");
            for w in &self.warnings {
                let _ = writeln!(s, "  - {w}");
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_is_relative_to_own_history() {
        let row: Vec<Option<f64>> = (0..30)
            .map(|t| Some(if t < 29 { 0.2 } else { 0.5 }))
            .collect();
        let s = surprise(&[row], 60, 20);
        assert!(s[0][10].is_none(), "not enough history yet");
        assert!((s[0][29].unwrap() - 0.3).abs() < 1e-12);
        assert!(s[0][28].unwrap().abs() < 1e-12);
    }
}
