//! Fundamentals from SEC EDGAR XBRL "company facts".
//!
//! Companies tag the same concept under different us-gaap elements (Apple
//! moved from `Revenues` to `RevenueFromContractWithCustomerExcludingAssessedTax`
//! in 2018; Amazon never used `Revenues`), and each filing repeats prior-period
//! comparatives. So for every metric we try several tags, keep only genuine
//! annual (or instant) observations, de-duplicate by period end keeping the
//! latest filing, and pick the tag with the freshest data.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::http::{Fetcher, Upstream};

const REVENUE_TAGS: &[&str] = &[
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
];
const NET_INCOME_TAGS: &[&str] = &["NetIncomeLoss", "ProfitLoss"];
const DEBT_TAGS: &[&str] = &[
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
];
const EQUITY_TAGS: &[&str] = &[
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
];
const SHARE_TAGS: &[&str] = &[
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
];

/// Annual figures older than this are considered stale and ignored.
const MAX_ANNUAL_AGE_DAYS: i64 = 550;
/// Balance-sheet / share-count figures older than this are ignored.
const MAX_INSTANT_AGE_DAYS: i64 = 400;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Fundamentals {
    pub cik: String,
    pub entity_name: Option<String>,
    /// Latest fiscal year-over-year revenue growth, as a fraction (0.07 = 7%).
    pub revenue_growth: Option<f64>,
    /// Latest fiscal year net income / revenue.
    pub net_margin: Option<f64>,
    /// Total long-term debt / stockholders' equity at the latest balance sheet.
    pub debt_to_equity: Option<f64>,
    pub shares_outstanding: Option<f64>,
    pub fiscal_year_end: Option<NaiveDate>,
}

// ---------------------------------------------------------------------------
// Ticker -> CIK
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TickerEntry {
    cik_str: u64,
    ticker: String,
}

/// Ticker -> zero-padded CIK, from SEC's official mapping (cached for a week).
pub async fn ticker_map(fetcher: &Fetcher) -> Result<HashMap<String, String>> {
    let body = fetcher
        .get_text(
            "https://www.sec.gov/files/company_tickers.json",
            Upstream::Sec,
            Duration::from_secs(7 * 24 * 3600),
        )
        .await?;
    let entries: HashMap<String, TickerEntry> =
        serde_json::from_str(&body).context("malformed SEC ticker map")?;
    Ok(entries
        .into_values()
        .map(|e| (e.ticker.to_uppercase(), format!("{:010}", e.cik_str)))
        .collect())
}

pub async fn fetch_fundamentals(fetcher: &Fetcher, cik: &str) -> Result<Fundamentals> {
    let facts = fetch_company_facts(fetcher, cik).await?;
    Ok(extract(cik, &facts, Utc::now().date_naive()))
}

pub async fn fetch_company_facts(fetcher: &Fetcher, cik: &str) -> Result<CompanyFacts> {
    let url = format!("https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json");
    let body = fetcher
        .get_text(&url, Upstream::Sec, Duration::from_secs(24 * 3600))
        .await?;
    serde_json::from_str(&body).context("malformed SEC company facts")
}

/// Fundamentals exactly as an investor could have known them on `as_of`:
/// only facts *filed* on or before that date are visible. This is what makes
/// historical evaluation free of look-ahead bias.
pub fn fundamentals_as_of(cik: &str, cf: &CompanyFacts, as_of: NaiveDate) -> Fundamentals {
    extract(cik, &cf.filed_by(as_of), as_of)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone)]
pub struct CompanyFacts {
    #[serde(rename = "entityName")]
    entity_name: Option<String>,
    #[serde(default)]
    facts: HashMap<String, HashMap<String, Concept>>,
}

#[derive(Deserialize, Clone)]
struct Concept {
    #[serde(default)]
    units: HashMap<String, Vec<Fact>>,
}

#[derive(Deserialize, Clone)]
struct Fact {
    start: Option<NaiveDate>,
    end: NaiveDate,
    val: f64,
    form: String,
    filed: NaiveDate,
}

impl CompanyFacts {
    /// A copy containing only facts filed on or before `date`.
    fn filed_by(&self, date: NaiveDate) -> CompanyFacts {
        let facts = self
            .facts
            .iter()
            .map(|(taxonomy, concepts)| {
                let concepts = concepts
                    .iter()
                    .map(|(tag, concept)| {
                        let units = concept
                            .units
                            .iter()
                            .map(|(unit, facts)| {
                                let kept =
                                    facts.iter().filter(|f| f.filed <= date).cloned().collect();
                                (unit.clone(), kept)
                            })
                            .collect();
                        (tag.clone(), Concept { units })
                    })
                    .collect();
                (taxonomy.clone(), concepts)
            })
            .collect();
        CompanyFacts {
            entity_name: self.entity_name.clone(),
            facts,
        }
    }
}

impl Fact {
    fn is_periodic_report(&self) -> bool {
        self.form.starts_with("10-K")
            || self.form.starts_with("10-Q")
            || self.form.starts_with("20-F")
    }

    fn is_annual(&self) -> bool {
        let annual_form = self.form.starts_with("10-K") || self.form.starts_with("20-F");
        let days = self.start.map(|s| (self.end - s).num_days());
        annual_form && matches!(days, Some(350..=380))
    }
}

fn facts<'a>(cf: &'a CompanyFacts, taxonomy: &str, tag: &str, unit: &str) -> &'a [Fact] {
    cf.facts
        .get(taxonomy)
        .and_then(|t| t.get(tag))
        .and_then(|c| c.units.get(unit))
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// Keep one observation per period end: the most recently filed one, so
/// restatements win over the originally reported figure.
fn dedupe_by_end<'a>(items: impl Iterator<Item = &'a Fact>) -> BTreeMap<NaiveDate, &'a Fact> {
    let mut by_end: BTreeMap<NaiveDate, &Fact> = BTreeMap::new();
    for f in items {
        by_end
            .entry(f.end)
            .and_modify(|cur| {
                if f.filed > cur.filed {
                    *cur = f;
                }
            })
            .or_insert(f);
    }
    by_end
}

/// Annual series for the first tag (in order of preference) with the most
/// recent fiscal year.
fn annual_series(cf: &CompanyFacts, tags: &[&str]) -> BTreeMap<NaiveDate, f64> {
    tags.iter()
        .map(|tag| {
            dedupe_by_end(
                facts(cf, "us-gaap", tag, "USD")
                    .iter()
                    .filter(|f| f.is_annual()),
            )
            .into_iter()
            .map(|(d, f)| (d, f.val))
            .collect::<BTreeMap<_, _>>()
        })
        .filter(|s| !s.is_empty())
        // max_by_key returns the last max; reverse so earlier tags win ties.
        .rev()
        .max_by_key(|s| s.keys().next_back().copied())
        .unwrap_or_default()
}

/// Point-in-time series (balance sheet items) from 10-K and 10-Q filings.
fn instant_series(
    cf: &CompanyFacts,
    taxonomy: &str,
    tags: &[&str],
    unit: &str,
) -> BTreeMap<NaiveDate, f64> {
    tags.iter()
        .map(|tag| {
            dedupe_by_end(
                facts(cf, taxonomy, tag, unit)
                    .iter()
                    .filter(|f| f.is_periodic_report()),
            )
            .into_iter()
            .map(|(d, f)| (d, f.val))
            .collect::<BTreeMap<_, _>>()
        })
        .filter(|s| !s.is_empty())
        .rev()
        .max_by_key(|s| s.keys().next_back().copied())
        .unwrap_or_default()
}

fn latest_recent(
    series: &BTreeMap<NaiveDate, f64>,
    today: NaiveDate,
    max_age: i64,
) -> Option<(NaiveDate, f64)> {
    series
        .iter()
        .next_back()
        .filter(|(d, _)| (today - **d).num_days() <= max_age)
        .map(|(d, v)| (*d, *v))
}

/// dei share counts are reported per class for multi-class companies, so sum
/// every class reported for the latest cover-page date.
fn dei_shares(cf: &CompanyFacts) -> Option<(NaiveDate, f64)> {
    let all = facts(cf, "dei", "EntityCommonStockSharesOutstanding", "shares");
    let latest_filed = all.iter().map(|f| f.filed).max()?;
    let latest: Vec<&Fact> = all.iter().filter(|f| f.filed == latest_filed).collect();
    let end = latest.iter().map(|f| f.end).max()?;
    let total: f64 = latest.iter().filter(|f| f.end == end).map(|f| f.val).sum();
    (total > 0.0).then_some((end, total))
}

pub(crate) fn extract(cik: &str, cf: &CompanyFacts, today: NaiveDate) -> Fundamentals {
    let revenue = annual_series(cf, REVENUE_TAGS);
    let mut out = Fundamentals {
        cik: cik.to_string(),
        entity_name: cf.entity_name.clone(),
        ..Default::default()
    };

    if let Some((end, rev)) = latest_recent(&revenue, today, MAX_ANNUAL_AGE_DAYS) {
        out.fiscal_year_end = Some(end);
        // Previous fiscal year: the closest period end roughly a year earlier.
        let prev = revenue
            .range(..end)
            .next_back()
            .filter(|(d, _)| (300..=430).contains(&(end - **d).num_days()));
        if let Some((_, &prev_rev)) = prev
            && prev_rev > 0.0
        {
            out.revenue_growth = Some((rev - prev_rev) / prev_rev);
        }
        let income = annual_series(cf, NET_INCOME_TAGS);
        if let Some(&ni) = income.get(&end)
            && rev > 0.0
        {
            out.net_margin = Some(ni / rev);
        }
    }

    let equity = instant_series(cf, "us-gaap", EQUITY_TAGS, "USD");
    let debt = instant_series(cf, "us-gaap", DEBT_TAGS, "USD");
    if let Some((end, eq)) = latest_recent(&equity, today, MAX_INSTANT_AGE_DAYS)
        && let Some(&d) = debt.get(&end)
    {
        // Negative equity makes the ratio meaningless; flag it as very high.
        out.debt_to_equity = Some(if eq > 0.0 { d / eq } else { f64::INFINITY });
    }

    let gaap_shares = latest_recent(
        &instant_series(cf, "us-gaap", SHARE_TAGS, "shares"),
        today,
        MAX_INSTANT_AGE_DAYS,
    );
    out.shares_outstanding = dei_shares(cf)
        .filter(|(d, _)| (today - *d).num_days() <= MAX_INSTANT_AGE_DAYS)
        .or(gaap_shares)
        .map(|(_, v)| v);

    out
}

pub fn lookup_cik(map: &HashMap<String, String>, ticker: &str) -> Result<String> {
    map.get(ticker)
        .cloned()
        .ok_or_else(|| anyhow!("{ticker} is not an SEC registrant (ETF, fund or foreign listing?)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        s.parse().unwrap()
    }

    const FIXTURE: &str = r#"{
      "entityName": "Example Corp",
      "facts": {
        "dei": {
          "EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end":"2025-01-20","val":900,"form":"10-K","filed":"2025-02-01"},
            {"end":"2025-07-20","val":600,"form":"10-Q","filed":"2025-08-01"},
            {"end":"2025-07-20","val":400,"form":"10-Q","filed":"2025-08-01"}
          ]}}
        },
        "us-gaap": {
          "Revenues": {"units": {"USD": [
            {"start":"2015-01-01","end":"2015-12-31","val":1,"form":"10-K","filed":"2016-02-01"}
          ]}},
          "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start":"2023-01-01","end":"2023-12-31","val":1000,"form":"10-K","filed":"2024-02-01"},
            {"start":"2024-01-01","end":"2024-12-31","val":1100,"form":"10-K","filed":"2025-02-01"},
            {"start":"2023-01-01","end":"2023-12-31","val":1000,"form":"10-K","filed":"2025-02-01"},
            {"start":"2024-10-01","end":"2024-12-31","val":300,"form":"10-K","filed":"2025-02-01"},
            {"start":"2025-01-01","end":"2025-06-30","val":600,"form":"10-Q","filed":"2025-08-01"}
          ]}},
          "NetIncomeLoss": {"units": {"USD": [
            {"start":"2024-01-01","end":"2024-12-31","val":110,"form":"10-K","filed":"2025-02-01"}
          ]}},
          "StockholdersEquity": {"units": {"USD": [
            {"end":"2024-12-31","val":500,"form":"10-K","filed":"2025-02-01"},
            {"end":"2025-06-30","val":400,"form":"10-Q","filed":"2025-08-01"}
          ]}},
          "LongTermDebt": {"units": {"USD": [
            {"end":"2025-06-30","val":200,"form":"10-Q","filed":"2025-08-01"}
          ]}}
        }
      }
    }"#;

    #[test]
    fn extracts_fundamentals_from_messy_facts() {
        let cf: CompanyFacts = serde_json::from_str(FIXTURE).unwrap();
        let f = extract("0000000001", &cf, d("2025-09-01"));
        assert_eq!(f.fiscal_year_end, Some(d("2024-12-31")));
        // Quarter-length "10-K" entries and stale `Revenues` must be ignored.
        assert!((f.revenue_growth.unwrap() - 0.10).abs() < 1e-12);
        assert!((f.net_margin.unwrap() - 0.10).abs() < 1e-12);
        assert!((f.debt_to_equity.unwrap() - 0.5).abs() < 1e-12);
        // Two share classes on the latest cover page are summed.
        assert_eq!(f.shares_outstanding, Some(1000.0));
    }

    #[test]
    fn point_in_time_ignores_later_filings() {
        let cf: CompanyFacts = serde_json::from_str(FIXTURE).unwrap();
        // Before the FY2024 10-K was filed only FY2023 revenue was known, so
        // growth is unavailable and the latest fiscal year is 2023.
        let f = fundamentals_as_of("0000000001", &cf, d("2024-06-01"));
        assert_eq!(f.fiscal_year_end, Some(d("2023-12-31")));
        assert!(f.revenue_growth.is_none());
        assert!(f.debt_to_equity.is_none());
        assert_eq!(f.shares_outstanding, None);
    }

    #[test]
    fn stale_data_is_dropped() {
        let cf: CompanyFacts = serde_json::from_str(FIXTURE).unwrap();
        let f = extract("0000000001", &cf, d("2030-01-01"));
        assert!(f.revenue_growth.is_none());
        assert!(f.debt_to_equity.is_none());
        assert!(f.shares_outstanding.is_none());
    }
}
