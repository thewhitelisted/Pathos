//! Recent headlines from Yahoo Finance and Google News RSS feeds.
//!
//! RSS gives us clean headlines without scraping arbitrary article HTML (which
//! is mostly cookie banners and navigation), and headlines are exactly the
//! kind of short financial sentence FinBERT was trained on.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::http::{Fetcher, Upstream};

#[derive(Debug, Clone, Serialize)]
pub struct Headline {
    pub title: String,
    pub url: String,
    pub source: String,
    pub published: DateTime<Utc>,
}

/// Fetch, merge and de-duplicate headlines from every feed, newest first.
///
/// A feed failing is not fatal: we log it and use whatever the others return.
pub async fn fetch_headlines(
    fetcher: &Fetcher,
    ticker: &str,
    company_name: Option<&str>,
    max_age_days: u32,
    limit: usize,
) -> Vec<Headline> {
    let ttl = Duration::from_secs(30 * 60);
    let yahoo_url =
        format!("https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US");
    let query = match company_name
        .map(clean_company_name)
        .filter(|n| !n.is_empty())
    {
        Some(name) => format!("({ticker} OR \"{name}\") stock when:{max_age_days}d"),
        None => format!("{ticker} stock when:{max_age_days}d"),
    };
    let google_url = format!(
        "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en",
        urlencode(&query)
    );

    let (yahoo, google) = tokio::join!(
        fetcher.get_text(&yahoo_url, Upstream::News, ttl),
        fetcher.get_text(&google_url, Upstream::News, ttl),
    );

    let mut all = Vec::new();
    for (feed, body) in [("Yahoo Finance", yahoo), ("Google News", google)] {
        match body.and_then(|b| parse_rss(&b, feed)) {
            Ok(items) => all.extend(items),
            Err(e) => tracing::warn!(ticker, feed, error = %e, "news feed failed"),
        }
    }

    let cutoff = Utc::now() - chrono::Duration::days(i64::from(max_age_days));
    let terms = relevance_terms(ticker, company_name);
    let mut seen = HashSet::new();
    all.retain(|h| {
        h.published >= cutoff
            && is_relevant(&h.title, ticker, &terms)
            && seen.insert(dedupe_key(&h.title))
    });
    all.sort_by_key(|h| std::cmp::Reverse(h.published));
    all.truncate(limit);
    all
}

pub fn parse_rss(xml: &str, default_source: &str) -> Result<Vec<Headline>> {
    let doc = roxmltree::Document::parse(xml).context("malformed RSS")?;
    let child_text = |node: roxmltree::Node, name: &str| -> Option<String> {
        node.children()
            .find(|c| c.has_tag_name(name))
            .and_then(|c| c.text())
            .map(|t| t.trim().to_string())
    };

    let items = doc
        .descendants()
        .filter(|n| n.has_tag_name("item"))
        .filter_map(|item| {
            let raw_title = child_text(item, "title")?;
            let url = child_text(item, "link").unwrap_or_default();
            let published = child_text(item, "pubDate")
                .and_then(|d| DateTime::parse_from_rfc2822(&d).ok())?
                .with_timezone(&Utc);
            // Google News appends " - Publisher" to titles and also provides
            // the publisher in a <source> element.
            let source = child_text(item, "source");
            let title = match &source {
                Some(src) => raw_title
                    .strip_suffix(src.as_str())
                    .map(|t| {
                        t.trim_end()
                            .trim_end_matches(['-', '|', '–'])
                            .trim_end()
                            .to_string()
                    })
                    .unwrap_or(raw_title),
                None => raw_title,
            };
            Some(Headline {
                title: decode_entities(&title),
                url,
                source: source.unwrap_or_else(|| default_source.to_string()),
                published,
            })
        })
        .filter(|h| h.title.split_whitespace().count() >= 3)
        .collect();
    Ok(items)
}

/// Lower-case company names / aliases that mark a headline as being about
/// this company. Ticker feeds also carry general market stories ("3 growth
/// stocks to watch") that would otherwise pollute the signal.
pub(crate) fn relevance_terms(ticker: &str, company_name: Option<&str>) -> Vec<String> {
    const GENERIC: &[&str] = &[
        "american",
        "united",
        "general",
        "first",
        "national",
        "international",
        "global",
        "bank",
        "energy",
        "capital",
        "financial",
        "technologies",
        "systems",
    ];
    let mut terms = Vec::new();
    if let Some(name) = company_name
        .map(clean_company_name)
        .map(|n| n.to_lowercase())
    {
        if let Some(first) = name.split_whitespace().next()
            && first.len() >= 4
            && !GENERIC.contains(&first)
        {
            terms.push(first.to_string());
        }
        if !name.is_empty() {
            terms.push(name);
        }
    }
    let aliases: &[&str] = match ticker {
        "GOOGL" | "GOOG" => &["google", "alphabet"],
        "META" => &["meta", "facebook", "instagram"],
        "BRK-A" | "BRK-B" => &["berkshire", "buffett"],
        "AMZN" => &["amazon", "aws"],
        _ => &[],
    };
    terms.extend(aliases.iter().map(|a| a.to_string()));
    terms
}

pub(crate) fn is_relevant(title: &str, ticker: &str, terms: &[String]) -> bool {
    // Tickers appear upper-case in headlines; matching case-sensitively
    // avoids false hits for tickers that are also words (e.g. "ALL", "NOW").
    let words: Vec<&str> = title
        .split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .filter(|w| !w.is_empty())
        .collect();
    if words.contains(&ticker) {
        return true;
    }
    let lower = title.to_lowercase();
    let lower_words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    terms.iter().any(|t| {
        if t.contains(' ') {
            lower.contains(t.as_str())
        } else {
            lower_words.contains(&t.as_str())
        }
    })
}

/// Case/punctuation-insensitive key so syndicated copies collapse together.
pub(crate) fn dedupe_key(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

pub(crate) fn clean_company_name(name: &str) -> String {
    const SUFFIXES: &[&str] = &[
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "company",
        "com",
        "co",
        "ltd",
        "limited",
        "plc",
        "holdings",
        "group",
        "class a",
        "class b",
        "class c",
        "the",
    ];
    let mut n = name.replace([',', '.'], " ").to_lowercase();
    loop {
        let trimmed = n.trim().to_string();
        let stripped = SUFFIXES
            .iter()
            .find_map(|s| trimmed.strip_suffix(s).filter(|r| r.ends_with(' ')))
            .map(str::to_string);
        match stripped {
            Some(s) => n = s,
            None => break,
        }
    }
    // Restore title case for readability in the query.
    n.split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_uppercase().chain(c).collect::<String>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

pub(crate) fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOGLE: &str = r#"<?xml version="1.0"?><rss><channel>
      <item><title>Apple Stock Falls After Downgrade - Yahoo Finance</title>
        <link>https://example.com/a</link><pubDate>Tue, 29 Sep 2026 16:09:04 GMT</pubDate>
        <source url="https://finance.yahoo.com">Yahoo Finance</source></item>
      <item><title>Too short</title><link>x</link><pubDate>Tue, 29 Sep 2026 16:09:04 GMT</pubDate></item>
      <item><title>No date on this one</title><link>x</link></item>
    </channel></rss>"#;

    const YAHOO: &str = r#"<?xml version="1.0"?><rss><channel>
      <item><title>Nvidia, Apple &amp; Microsoft keep the market afloat</title>
        <link>https://example.com/b</link><pubDate>Tue, 06 Oct 2026 14:08:45 +0000</pubDate></item>
    </channel></rss>"#;

    #[test]
    fn parses_google_news_and_strips_publisher() {
        let items = parse_rss(GOOGLE, "Google News").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "Apple Stock Falls After Downgrade");
        assert_eq!(items[0].source, "Yahoo Finance");
    }

    #[test]
    fn parses_yahoo_feed() {
        let items = parse_rss(YAHOO, "Yahoo Finance").unwrap();
        assert_eq!(
            items[0].title,
            "Nvidia, Apple & Microsoft keep the market afloat"
        );
        assert_eq!(items[0].source, "Yahoo Finance");
    }

    #[test]
    fn cleans_company_names() {
        assert_eq!(clean_company_name("Apple Inc."), "Apple");
        assert_eq!(clean_company_name("MICROSOFT CORP"), "Microsoft");
        assert_eq!(clean_company_name("Alphabet Inc. Class A"), "Alphabet");
        assert_eq!(clean_company_name("Amazon.com, Inc."), "Amazon");
    }

    #[test]
    fn filters_irrelevant_market_news() {
        let terms = relevance_terms("TSLA", Some("Tesla, Inc."));
        assert!(is_relevant(
            "Tesla recalls Model Y over seatbelt issue",
            "TSLA",
            &terms
        ));
        assert!(is_relevant("Why TSLA shares jumped today", "TSLA", &terms));
        assert!(is_relevant(
            "Tesla's robotaxi push draws scrutiny",
            "TSLA",
            &terms
        ));
        assert!(!is_relevant(
            "3 Growth Stocks To Watch With Revenue Growth Up To 43%",
            "TSLA",
            &terms
        ));

        let terms = relevance_terms("GOOGL", Some("Alphabet Inc."));
        assert!(is_relevant(
            "Google faces UK class action over Play Store fees",
            "GOOGL",
            &terms
        ));

        let terms = relevance_terms("NOW", Some("ServiceNow, Inc."));
        assert!(!is_relevant(
            "Stocks to buy now before earnings",
            "NOW",
            &terms
        ));
        assert!(is_relevant("ServiceNow beats estimates", "NOW", &terms));
    }

    #[test]
    fn dedupes_syndicated_titles() {
        assert_eq!(dedupe_key("Apple beats!"), dedupe_key("apple BEATS"));
    }
}
