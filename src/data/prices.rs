//! Daily adjusted close prices from the Yahoo Finance chart API.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, NaiveDate};
use serde::Deserialize;

use crate::http::{Fetcher, Upstream};

#[derive(Debug, Clone)]
pub struct PriceHistory {
    pub ticker: String,
    pub name: Option<String>,
    pub currency: Option<String>,
    pub last_price: f64,
    /// Trading-day close prices adjusted for splits and dividends.
    pub closes: BTreeMap<NaiveDate, f64>,
    /// Stock splits as (date, new shares per old share): 10.0 for a 10-for-1
    /// split, 1/30 for a 1-for-30 reverse split.
    pub splits: Vec<(NaiveDate, f64)>,
}

impl PriceHistory {
    /// Product of split ratios for splits strictly after `date`.
    pub fn split_factor_after(&self, date: NaiveDate) -> f64 {
        split_factor_after(&self.splits, date)
    }
}

pub fn split_factor_after(splits: &[(NaiveDate, f64)], date: NaiveDate) -> f64 {
    splits
        .iter()
        .filter(|(d, _)| *d > date)
        .map(|(_, r)| r)
        .product()
}

/// Market capitalisation from a split-adjusted price and an as-reported
/// share count.
///
/// Yahoo rewrites historical prices after every split, but SEC share counts
/// are as reported on `shares_date`. With `P̃(t) = P(t) / Π_{s>t} r_s` the
/// adjusted price and `N(t) = N_rep · Π_{shares_date<s≤t} r_s` the true share
/// count, the true cap `P(t)·N(t)` simplifies to
/// `P̃(t) · N_rep · Π_{s>shares_date} r_s`.
/// Multiplying the two raw inputs instead is off by the full split ratio
/// (30x for a 1-for-30 reverse split).
pub fn market_cap(
    adjusted_price: f64,
    shares: f64,
    shares_date: Option<NaiveDate>,
    splits: &[(NaiveDate, f64)],
) -> f64 {
    let factor = shares_date.map_or(1.0, |d| split_factor_after(splits, d));
    adjusted_price * shares * factor
}

#[derive(Deserialize)]
struct ChartResponse {
    chart: Chart,
}

#[derive(Deserialize)]
struct Chart {
    result: Option<Vec<ChartResult>>,
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChartResult {
    meta: Meta,
    #[serde(default)]
    timestamp: Vec<i64>,
    indicators: Indicators,
    #[serde(default)]
    events: Option<Events>,
}

#[derive(Deserialize)]
struct Events {
    #[serde(default)]
    splits: std::collections::HashMap<String, Split>,
}

#[derive(Deserialize)]
struct Split {
    date: i64,
    numerator: f64,
    denominator: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    regular_market_price: Option<f64>,
    long_name: Option<String>,
    short_name: Option<String>,
    currency: Option<String>,
    #[serde(default)]
    gmtoffset: i64,
}

#[derive(Deserialize)]
struct Indicators {
    #[serde(default)]
    adjclose: Vec<AdjClose>,
    #[serde(default)]
    quote: Vec<Quote>,
}

#[derive(Deserialize)]
struct AdjClose {
    adjclose: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Quote {
    #[serde(default)]
    close: Vec<Option<f64>>,
}

/// Fetch roughly `lookback_days` calendar days of daily prices.
pub async fn fetch_history(
    fetcher: &Fetcher,
    ticker: &str,
    lookback_days: u32,
) -> Result<PriceHistory> {
    let range = match lookback_days {
        0..=31 => "1mo",
        32..=93 => "3mo",
        94..=186 => "6mo",
        187..=366 => "1y",
        367..=731 => "2y",
        _ => "5y",
    };
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={range}&interval=1d&includeAdjustedClose=true&events=split"
    );
    let body = fetcher
        .get_text(&url, Upstream::Yahoo, Duration::from_secs(6 * 3600))
        .await
        .map_err(|e| {
            if e.to_string().contains("404") {
                anyhow!("unknown symbol on Yahoo Finance")
            } else {
                e.context("price history unavailable")
            }
        })?;
    parse_chart(ticker, &body)
}

fn parse_chart(ticker: &str, body: &str) -> Result<PriceHistory> {
    let resp: ChartResponse =
        serde_json::from_str(body).context("malformed Yahoo chart response")?;
    let result = resp.chart.result.and_then(|mut r| r.pop()).ok_or_else(|| {
        anyhow!(
            "Yahoo returned no chart for {ticker}: {:?}",
            resp.chart.error
        )
    })?;

    // Prefer adjusted closes; fall back to raw closes (e.g. for indices).
    let series: Vec<Option<f64>> = result
        .indicators
        .adjclose
        .into_iter()
        .next()
        .map(|a| a.adjclose)
        .or_else(|| result.indicators.quote.into_iter().next().map(|q| q.close))
        .unwrap_or_default();

    let offset = result.meta.gmtoffset;
    let closes: BTreeMap<NaiveDate, f64> = result
        .timestamp
        .iter()
        .zip(series)
        .filter_map(|(&ts, px)| {
            let px = px.filter(|p| p.is_finite() && *p > 0.0)?;
            let date = DateTime::from_timestamp(ts + offset, 0)?.date_naive();
            Some((date, px))
        })
        .collect();

    let mut splits: Vec<(NaiveDate, f64)> = result
        .events
        .map(|e| e.splits.into_values().collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter(|sp| sp.numerator > 0.0 && sp.denominator > 0.0)
        .filter_map(|sp| {
            let date = DateTime::from_timestamp(sp.date + offset, 0)?.date_naive();
            Some((date, sp.numerator / sp.denominator))
        })
        .collect();
    splits.sort_by_key(|(d, _)| *d);

    let last_price = result
        .meta
        .regular_market_price
        .or_else(|| closes.values().next_back().copied())
        .ok_or_else(|| anyhow!("no prices for {ticker}"))?;

    Ok(PriceHistory {
        ticker: ticker.to_string(),
        name: result.meta.long_name.or(result.meta.short_name),
        currency: result.meta.currency,
        last_price,
        closes,
        splits,
    })
}

/// Daily log returns on the dates every series has a price for.
///
/// Returns a `T x N` row-major matrix (one row per day) in the same order as
/// `histories`. Aligning on common dates avoids pairing returns from
/// different days, which silently corrupts covariance estimates.
pub fn aligned_log_returns(histories: &[&PriceHistory]) -> Vec<Vec<f64>> {
    let Some(first) = histories.first() else {
        return Vec::new();
    };
    let common: BTreeSet<NaiveDate> =
        histories[1..]
            .iter()
            .fold(first.closes.keys().copied().collect(), |acc, h| {
                acc.into_iter()
                    .filter(|d| h.closes.contains_key(d))
                    .collect()
            });
    let dates: Vec<NaiveDate> = common.into_iter().collect();
    dates
        .windows(2)
        .map(|w| {
            histories
                .iter()
                .map(|h| (h.closes[&w[1]] / h.closes[&w[0]]).ln())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(ticker: &str, points: &[(u32, f64)]) -> PriceHistory {
        PriceHistory {
            ticker: ticker.into(),
            name: None,
            currency: None,
            last_price: points.last().unwrap().1,
            closes: points
                .iter()
                .map(|&(d, p)| (NaiveDate::from_ymd_opt(2025, 1, d).unwrap(), p))
                .collect(),
            splits: vec![],
        }
    }

    #[test]
    fn parses_chart_and_skips_nulls() {
        let body = r#"{"chart":{"result":[{"meta":{"currency":"USD","regularMarketPrice":12.0,
            "longName":"Test Co","gmtoffset":-14400},
            "timestamp":[1735740000,1735826400,1735912800],
            "indicators":{"quote":[{"close":[10.0,null,12.0]}],
                          "adjclose":[{"adjclose":[10.0,null,12.0]}]}}],"error":null}}"#;
        let h = parse_chart("TST", body).unwrap();
        assert_eq!(h.closes.len(), 2);
        assert_eq!(h.last_price, 12.0);
        assert_eq!(h.name.as_deref(), Some("Test Co"));
    }

    #[test]
    fn parses_splits() {
        let body = r#"{"chart":{"result":[{"meta":{"regularMarketPrice":5.0,"gmtoffset":0},
            "timestamp":[1735740000],
            "indicators":{"quote":[{"close":[5.0]}],"adjclose":[{"adjclose":[5.0]}]},
            "events":{"splits":{"1731335400":{"date":1731335400,"numerator":1.0,"denominator":30.0,"splitRatio":"1:30"}}}}],"error":null}}"#;
        let h = parse_chart("FCEL", body).unwrap();
        assert_eq!(h.splits.len(), 1);
        assert!((h.splits[0].1 - 1.0 / 30.0).abs() < 1e-12);
    }

    #[test]
    fn market_cap_corrects_for_later_splits() {
        let d = |m, day| NaiveDate::from_ymd_opt(2025, m, day).unwrap();
        // 1-for-30 reverse split on 1 June. On 1 March the real price was $1
        // with 3,000 shares reported on 31 Jan: a $3,000 company. Yahoo now
        // shows that day's price as $30.
        let splits = vec![(d(6, 1), 1.0 / 30.0)];
        let cap = market_cap(30.0, 3000.0, Some(d(1, 31)), &splits);
        assert!((cap - 3000.0).abs() < 1e-9, "naive product would be 90,000");
        // 10-for-1 forward split: adjusted price $50 was really $500.
        let fwd = vec![(d(6, 1), 10.0)];
        assert!((market_cap(50.0, 100.0, Some(d(1, 31)), &fwd) - 50_000.0).abs() < 1e-9);
        // Share count reported after the split needs no correction.
        assert!((market_cap(50.0, 1000.0, Some(d(7, 1)), &fwd) - 50_000.0).abs() < 1e-9);
    }

    #[test]
    fn reports_missing_chart() {
        let body = r#"{"chart":{"result":null,"error":{"code":"Not Found"}}}"#;
        assert!(parse_chart("NOPE", body).is_err());
    }

    #[test]
    fn aligns_on_common_dates() {
        let a = history("A", &[(1, 100.0), (2, 110.0), (3, 121.0)]);
        // B is missing day 2, so only the day1 -> day3 return is usable.
        let b = history("B", &[(1, 50.0), (3, 25.0)]);
        let r = aligned_log_returns(&[&a, &b]);
        assert_eq!(r.len(), 1);
        assert!((r[0][0] - (1.21f64).ln()).abs() < 1e-12);
        assert!((r[0][1] - (0.5f64).ln()).abs() < 1e-12);
    }
}
