//! Historical, point-in-time headlines for evaluation.
//!
//! Two sources:
//! - **Google News archive**: date-bounded RSS searches
//!   (`after:YYYY-MM-DD before:YYYY-MM-DD`), one per ticker-week. Free and
//!   self-contained, but timestamps are day-granular, which the evaluation
//!   handles by lagging signals a full day.
//! - **CSV import**: any dataset with date, ticker and headline columns
//!   (e.g. FNSPID), for longer or denser histories.

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use chrono::{Datelike, NaiveDate, NaiveDateTime, Utc};
use futures::future::join_all;
use serde::Serialize;

use crate::data::news::{
    clean_company_name, dedupe_key, is_relevant, parse_rss, relevance_terms, urlencode,
};
use crate::http::{Fetcher, Upstream};

#[derive(Debug, Clone, Serialize)]
pub struct DatedHeadline {
    pub ticker: String,
    pub date: NaiveDate,
    pub title: String,
    pub source: String,
}

/// Collect relevant headlines for `ticker` between `start` and `end`, at most
/// `per_week` per calendar week (in Google's relevance order).
pub async fn collect_google_news(
    fetcher: &Fetcher,
    ticker: &str,
    company_name: Option<&str>,
    start: NaiveDate,
    end: NaiveDate,
    per_week: usize,
) -> Vec<DatedHeadline> {
    let name = company_name
        .map(clean_company_name)
        .filter(|n| !n.is_empty());
    let terms = relevance_terms(ticker, company_name);
    let today = Utc::now().date_naive();

    // Monday-to-Monday weeks, independent of `start` and `end`, so the query
    // URLs (and therefore the HTTP cache) stay the same from one day to the
    // next; only the current week is ever re-fetched.
    let mut windows = Vec::new();
    let mut a = start - chrono::Duration::days(i64::from(start.weekday().num_days_from_monday()));
    while a < end {
        let b = a + chrono::Duration::days(7);
        windows.push((a, b));
        a = b;
    }

    let fetches = windows.iter().map(|&(a, b)| {
        let query = match &name {
            Some(n) => format!("({ticker} OR \"{n}\") stock after:{a} before:{b}"),
            None => format!("{ticker} stock after:{a} before:{b}"),
        };
        let url = format!(
            "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en",
            urlencode(&query)
        );
        // Closed windows never change; cache them for good.
        let ttl = if (today - b).num_days() > 3 {
            Duration::from_secs(10 * 365 * 24 * 3600)
        } else {
            Duration::from_secs(3600)
        };
        async move { (a, b, fetcher.get_text(&url, Upstream::News, ttl).await) }
    });

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (a, b, body) in join_all(fetches).await {
        let items = match body.and_then(|xml| parse_rss(&xml, "Google News")) {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!(ticker, %a, error = %e, "archive window failed");
                continue;
            }
        };
        out.extend(
            items
                .into_iter()
                .map(|h| DatedHeadline {
                    ticker: ticker.to_string(),
                    date: h.published.date_naive(),
                    title: h.title,
                    source: h.source,
                })
                // `before:` is inclusive in practice; keep windows disjoint.
                .filter(|h| h.date >= a && h.date < b)
                .filter(|h| is_relevant(&h.title, ticker, &terms))
                .filter(|h| seen.insert(dedupe_key(&h.title)))
                .take(per_week),
        );
    }
    out.sort_by_key(|h| h.date);
    out
}

/// Load headlines from a CSV with a header row. Column names are matched
/// case-insensitively; accepted aliases cover FNSPID's
/// `Date, Article_title, Stock_symbol` layout.
pub fn load_csv(path: &Path, tickers: &[String]) -> Result<Vec<DatedHeadline>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_lowercase())
        .collect();
    let find = |names: &[&str]| headers.iter().position(|h| names.contains(&h.as_str()));
    let date_col = find(&["date", "published", "datetime", "timestamp"])
        .ok_or_else(|| anyhow!("CSV needs a date column"))?;
    let ticker_col = find(&["ticker", "symbol", "stock_symbol", "stock"])
        .ok_or_else(|| anyhow!("CSV needs a ticker column"))?;
    let title_col = find(&["title", "headline", "article_title"])
        .ok_or_else(|| anyhow!("CSV needs a title/headline column"))?;
    let source_col = find(&["source", "publisher", "publisher_name"]);
    let wanted: HashSet<&str> = tickers.iter().map(String::as_str).collect();

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for record in reader.records() {
        let Ok(record) = record else { continue };
        let ticker = crate::data::normalize_ticker(record.get(ticker_col).unwrap_or(""));
        if !wanted.contains(ticker.as_str()) {
            continue;
        }
        let (Some(date), Some(title)) = (
            record.get(date_col).and_then(parse_date),
            record
                .get(title_col)
                .map(str::trim)
                .filter(|t| !t.is_empty()),
        ) else {
            continue;
        };
        if seen.insert((ticker.clone(), date, dedupe_key(title))) {
            out.push(DatedHeadline {
                ticker,
                date,
                title: title.to_string(),
                source: source_col
                    .and_then(|c| record.get(c))
                    .unwrap_or("CSV")
                    .to_string(),
            });
        }
    }
    out.sort_by_key(|h| h.date);
    Ok(out)
}

fn parse_date(s: &str) -> Option<NaiveDate> {
    let s = s.trim();
    // Timestamps with an offset are converted to UTC first.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc).date_naive());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S %z") {
        return Some(dt.with_timezone(&Utc).date_naive());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.date());
    }
    s.get(..10)
        .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_date_formats() {
        let d = NaiveDate::from_ymd_opt(2023, 5, 4).unwrap();
        assert_eq!(parse_date("2023-05-04"), Some(d));
        assert_eq!(parse_date("2023-05-04 13:00:00"), Some(d));
        // 22:30 New York time is already the next day in UTC.
        assert_eq!(parse_date("2023-05-04 22:30:00 -0400"), d.succ_opt());
        assert_eq!(parse_date("2023-05-04T10:00:00Z"), Some(d));
        assert_eq!(parse_date("not a date"), None);
    }

    #[test]
    fn loads_fnspid_style_csv() {
        let dir = std::env::temp_dir().join(format!("pathos-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("news.csv");
        std::fs::write(
            &path,
            "Date,Article_title,Stock_symbol,Url\n\
             2023-05-04 13:00:00 UTC,\"Apple beats, raises\",aapl,x\n\
             2023-05-04 14:00:00 UTC,\"Apple beats, raises\",AAPL,y\n\
             2023-05-05 09:00:00 UTC,Tesla cuts prices,TSLA,z\n\
             2023-05-05 09:00:00 UTC,Unrelated,XOM,z\n",
        )
        .unwrap();
        let rows = load_csv(&path, &["AAPL".into(), "TSLA".into()]).unwrap();
        assert_eq!(rows.len(), 2, "duplicate and unwanted tickers dropped");
        assert_eq!(rows[0].title, "Apple beats, raises");
        assert_eq!(rows[1].ticker, "TSLA");
        let _ = std::fs::remove_dir_all(dir);
    }
}
