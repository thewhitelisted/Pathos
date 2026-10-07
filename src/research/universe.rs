//! Point-in-time random universe sampling.
//!
//! Hand-picking a universe today biases any backtest toward companies that
//! survived and grew. This sampler instead draws companies at random from
//! everything that reported a share count to the SEC for a quarter ending at
//! least 90 days before the start date, and keeps those whose market cap *on
//! the start date* falls in the requested range, using only prices up to that
//! date and share counts reported for that quarter.
//!
//! One bias remains and is reported: tickers come from the SEC's current
//! ticker list and prices from Yahoo, so companies delisted since the start
//! date cannot be sampled.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use chrono::{Datelike, NaiveDate};
use futures::future::join_all;
use serde::{Deserialize, Serialize};

use super::stats::Rng;
use crate::data::prices::{self, PriceHistory};
use crate::data::sec;
use crate::http::{Fetcher, Upstream};

#[derive(Debug, Clone, Serialize)]
pub struct SampledUniverse {
    pub as_of: NaiveDate,
    pub seed: u64,
    pub min_cap: f64,
    pub max_cap: f64,
    /// SEC frame used for share counts, e.g. CY2024Q2I.
    pub share_frame: String,
    /// Companies in the frame with a plain listed ticker.
    pub candidates: usize,
    /// Candidates examined before `tickers` was filled.
    pub screened: usize,
    pub tickers: Vec<String>,
    /// Market cap on `as_of` for each sampled ticker.
    pub caps: Vec<f64>,
}

#[derive(Deserialize)]
struct Frame {
    data: Vec<FrameRow>,
}

#[derive(Deserialize)]
struct FrameRow {
    cik: u64,
    end: NaiveDate,
    val: f64,
}

/// Latest calendar quarter ending at least 90 days before `as_of`, as an
/// instant-frame label like `CY2024Q2I`.
pub fn share_frame(as_of: NaiveDate) -> String {
    let cutoff = as_of - chrono::Duration::days(90);
    let quarter = (cutoff.month0() / 3) as i32; // 0..=3, the quarter containing the cutoff
    // The quarter containing the cutoff has not necessarily ended; step back one.
    let (mut year, mut q) = (cutoff.year(), quarter);
    let quarter_end = |y: i32, q: i32| {
        let month = (q as u32) * 3 + 3;
        NaiveDate::from_ymd_opt(y, month, 1).unwrap() + chrono::Months::new(1)
            - chrono::Duration::days(1)
    };
    if quarter_end(year, q) > cutoff {
        if q == 0 {
            year -= 1;
            q = 3;
        } else {
            q -= 1;
        }
    }
    format!("CY{year}Q{}I", q + 1)
}

pub async fn sample(
    fetcher: &Fetcher,
    as_of: NaiveDate,
    n: usize,
    seed: u64,
    (min_cap, max_cap): (f64, f64),
    lookback_days: u32,
) -> Result<SampledUniverse> {
    ensure!(n >= 5, "sample at least 5 tickers");
    let frame_name = share_frame(as_of);
    let url = format!(
        "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{frame_name}.json"
    );
    let body = fetcher
        .get_text(&url, Upstream::Sec, Duration::from_secs(30 * 24 * 3600))
        .await
        .context("SEC share-count frame")?;
    let frame: Frame = serde_json::from_str(&body).context("malformed SEC frame")?;

    // CIK -> shortest plain ticker (skips warrants, units and preferreds).
    let mut by_cik: HashMap<String, String> = HashMap::new();
    for (ticker, cik) in sec::ticker_map(fetcher).await? {
        if !(1..=5).contains(&ticker.len()) || !ticker.chars().all(|c| c.is_ascii_uppercase()) {
            continue;
        }
        by_cik
            .entry(cik)
            .and_modify(|t| {
                if ticker.len() < t.len() || (ticker.len() == t.len() && ticker < *t) {
                    *t = ticker.clone();
                }
            })
            .or_insert(ticker);
    }
    let mut candidates: Vec<(String, NaiveDate, f64)> = frame
        .data
        .into_iter()
        .filter(|r| r.val > 0.0)
        .filter_map(|r| Some((by_cik.get(&format!("{:010}", r.cik))?.clone(), r.end, r.val)))
        .collect();
    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    candidates.dedup_by(|a, b| a.0 == b.0);
    let total = candidates.len();

    // Seeded Fisher-Yates shuffle for a reproducible random order.
    let mut rng = Rng::new(seed);
    for i in (1..candidates.len()).rev() {
        candidates.swap(i, rng.below(i + 1));
    }

    let mut tickers = Vec::new();
    let mut caps = Vec::new();
    let mut screened = 0;
    for chunk in candidates.chunks(8) {
        let histories = join_all(
            chunk
                .iter()
                .map(|(t, _, _)| prices::fetch_history(fetcher, t, lookback_days)),
        )
        .await;
        for ((ticker, shares_end, shares), history) in chunk.iter().zip(histories) {
            screened += 1;
            let Ok(h) = history else { continue };
            if let Some(cap) = cap_on(&h, as_of, *shares, *shares_end)
                && (min_cap..=max_cap).contains(&cap)
                && history_ok(&h, as_of)
            {
                tickers.push(ticker.clone());
                caps.push(cap);
                if tickers.len() == n {
                    break;
                }
            }
        }
        if tickers.len() == n {
            break;
        }
        tracing::info!("  screened {screened}/{total}, kept {}", tickers.len());
    }
    ensure!(
        tickers.len() == n,
        "only {} of {n} eligible companies found",
        tickers.len()
    );

    Ok(SampledUniverse {
        as_of,
        seed,
        min_cap,
        max_cap,
        share_frame: frame_name,
        candidates: total,
        screened,
        tickers,
        caps,
    })
}

/// Market cap on the last trading day on or before `as_of`.
fn cap_on(h: &PriceHistory, as_of: NaiveDate, shares: f64, shares_end: NaiveDate) -> Option<f64> {
    let (date, px) = h.closes.range(..=as_of).next_back()?;
    ((as_of - *date).num_days() <= 7)
        .then(|| prices::market_cap(*px, shares, Some(shares_end), &h.splits))
}

/// At least a year of trading before `as_of` (for the covariance window).
fn history_ok(h: &PriceHistory, as_of: NaiveDate) -> bool {
    h.closes.range(..as_of).count() >= 300
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_a_quarter_filed_before_the_start_date() {
        let d = |y, m, day| NaiveDate::from_ymd_opt(y, m, day).unwrap();
        assert_eq!(share_frame(d(2024, 10, 5)), "CY2024Q2I");
        assert_eq!(share_frame(d(2024, 9, 1)), "CY2024Q1I");
        assert_eq!(share_frame(d(2024, 2, 15)), "CY2023Q3I");
    }
}
