//! Date-aligned price and signal panels.
//!
//! Timing convention (the most important correctness property here):
//! a signal "as of" trading day `t` uses only headlines dated **strictly
//! before** `t`'s calendar date and filings **filed on or before** `t`. It is
//! traded at the close of `t`, and its forward return runs from close `t` to
//! close `t + h`. Because archive timestamps are day-granular, a headline
//! dated `d` might have appeared after the close on `d`; excluding day `t`
//! itself guarantees no look-ahead.

use chrono::{Datelike, NaiveDate};

use crate::data::prices::{self, PriceHistory};
use crate::data::sec::{CompanyFacts, fundamentals_as_of};
use crate::model::TRADING_DAYS;
use crate::model::signals::fundamental_score;
use crate::sentiment::{Probs, SentimentSummary, summarize_aged};

pub struct Panel {
    /// Trading days, taken from the benchmark's calendar.
    pub dates: Vec<NaiveDate>,
    pub tickers: Vec<String>,
    /// `close[i][t]`: adjusted close of ticker `i` on `dates[t]`.
    pub close: Vec<Vec<Option<f64>>>,
    pub bench: Vec<f64>,
    /// Per-ticker split history, for converting share counts.
    pub splits: Vec<Vec<(NaiveDate, f64)>>,
}

impl Panel {
    pub fn new(bench: &PriceHistory, histories: &[&PriceHistory]) -> Self {
        let dates: Vec<NaiveDate> = bench.closes.keys().copied().collect();
        Self {
            bench: dates.iter().map(|d| bench.closes[d]).collect(),
            close: histories
                .iter()
                .map(|h| dates.iter().map(|d| h.closes.get(d).copied()).collect())
                .collect(),
            tickers: histories.iter().map(|h| h.ticker.clone()).collect(),
            splits: histories.iter().map(|h| h.splits.clone()).collect(),
            dates,
        }
    }

    pub fn len(&self) -> usize {
        self.dates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }

    /// Market-excess log return of ticker `i` from close `from` to close `to`.
    pub fn excess_return(&self, i: usize, from: usize, to: usize) -> Option<f64> {
        if to >= self.len() || from >= to {
            return None;
        }
        let (a, b) = (self.close[i][from]?, self.close[i][to]?);
        Some((b / a).ln() - (self.bench[to] / self.bench[from]).ln())
    }

    /// Simple (not log) return of ticker `i` over day `t − 1 → t`.
    pub fn simple_return(&self, i: usize, t: usize) -> Option<f64> {
        if t == 0 {
            return None;
        }
        Some(self.close[i][t]? / self.close[i][t - 1]? - 1.0)
    }

    /// `window x idx.len()` daily log returns ending at `t`, or `None` if any
    /// ticker is missing a price in the window.
    pub fn trailing_log_returns(
        &self,
        idx: &[usize],
        t: usize,
        window: usize,
    ) -> Option<Vec<Vec<f64>>> {
        if t < window {
            return None;
        }
        (t + 1 - window..=t)
            .map(|d| {
                idx.iter()
                    .map(|&i| Some((self.close[i][d]? / self.close[i][d - 1]?).ln()))
                    .collect()
            })
            .collect()
    }

    /// Annualized realized volatility of daily log returns over `window` days.
    pub fn realized_vol(&self, i: usize, t: usize, window: usize) -> Option<f64> {
        let r: Vec<f64> = self
            .trailing_log_returns(&[i], t, window)?
            .into_iter()
            .map(|r| r[0])
            .collect();
        let m = r.iter().sum::<f64>() / r.len() as f64;
        let var = r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (r.len() - 1) as f64;
        Some((var * TRADING_DAYS).sqrt())
    }
}

/// `[ticker][date]` sentiment summaries using headlines dated in
/// `[t − window_days, t − 1]`. Entries with fewer than `min_headlines` are
/// `None`.
pub fn sentiment_panel(
    panel: &Panel,
    headlines: &[Vec<(NaiveDate, Probs)>],
    window_days: i64,
    min_headlines: usize,
) -> Vec<Vec<Option<SentimentSummary>>> {
    headlines
        .iter()
        .map(|hs| {
            let mut sorted = hs.clone();
            sorted.sort_by_key(|(d, _)| *d);
            let (mut lo, mut hi) = (0, 0);
            panel
                .dates
                .iter()
                .map(|&t| {
                    let first = t - chrono::Duration::days(window_days);
                    while hi < sorted.len() && sorted[hi].0 < t {
                        hi += 1;
                    }
                    while lo < hi && sorted[lo].0 < first {
                        lo += 1;
                    }
                    if hi - lo < min_headlines {
                        return None;
                    }
                    let aged: Vec<(f64, Probs)> = sorted[lo..hi]
                        .iter()
                        .map(|(d, p)| ((t - *d).num_days() as f64, *p))
                        .collect();
                    Some(summarize_aged(&aged))
                })
                .collect()
        })
        .collect()
}

pub struct FundamentalPanel {
    /// Point-in-time quality score in [-1, 1] and the fraction of its
    /// component metrics that were available.
    pub score: Vec<Vec<Option<(f64, f64)>>>,
    /// Market cap from point-in-time share count × that day's close.
    pub market_cap: Vec<Vec<Option<f64>>>,
}

/// Point-in-time fundamentals, recomputed at each month start (filings are
/// quarterly, so finer resolution adds cost without information).
pub fn fundamental_panel(
    panel: &Panel,
    facts: &[Option<(String, CompanyFacts)>],
) -> FundamentalPanel {
    let n = panel.tickers.len();
    let mut score = vec![vec![None; panel.len()]; n];
    let mut market_cap = vec![vec![None; panel.len()]; n];
    for (i, f) in facts.iter().enumerate() {
        let Some((cik, cf)) = f else { continue };
        let mut current = None;
        for t in 0..panel.len() {
            let d = panel.dates[t];
            if t == 0 || d.month() != panel.dates[t - 1].month() {
                current = Some(fundamentals_as_of(cik, cf, d));
            }
            if let Some(fund) = &current {
                score[i][t] = fundamental_score(fund);
                market_cap[i][t] = fund
                    .shares_outstanding
                    .zip(panel.close[i][t])
                    .map(|(s, p)| prices::market_cap(p, s, fund.shares_as_of, &panel.splits[i]));
            }
        }
    }
    FundamentalPanel { score, market_cap }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn d(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2025, 3, day).unwrap()
    }

    fn history(ticker: &str, px: &[f64]) -> PriceHistory {
        PriceHistory {
            ticker: ticker.into(),
            name: None,
            currency: None,
            last_price: *px.last().unwrap(),
            splits: vec![],
            closes: px
                .iter()
                .enumerate()
                .map(|(k, p)| (d(k as u32 + 3), *p))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn excess_returns_subtract_benchmark() {
        let bench = history("SPY", &[100.0, 110.0, 121.0]);
        let a = history("A", &[10.0, 11.0, 13.31]);
        let p = Panel::new(&bench, &[&a]);
        let r = p.excess_return(0, 0, 2).unwrap();
        assert!((r - ((1.331f64).ln() - (1.21f64).ln())).abs() < 1e-12);
        assert!(p.excess_return(0, 0, 3).is_none(), "no data past the end");
    }

    #[test]
    fn sentiment_never_uses_same_day_headlines() {
        let bench = history("SPY", &[1.0, 1.0, 1.0, 1.0]);
        let p = Panel::new(&bench, &[&bench]);
        let pos = Probs {
            positive: 1.0,
            negative: 0.0,
            neutral: 0.0,
        };
        let neg = Probs {
            positive: 0.0,
            negative: 1.0,
            neutral: 0.0,
        };
        // A positive headline on day 0, a negative one on day 1.
        let s = sentiment_panel(&p, &[vec![(d(3), pos), (d(4), neg)]], 7, 1);
        assert!(s[0][0].is_none(), "day-0 headline is not visible on day 0");
        assert_eq!(
            s[0][1].as_ref().unwrap().score,
            1.0,
            "day 1 sees only day 0"
        );
        assert!(s[0][2].as_ref().unwrap().score < 1.0, "day 2 sees both");
    }
}
