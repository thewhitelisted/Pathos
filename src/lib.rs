//! Pathos: sentiment-aware portfolio construction.
//!
//! The pipeline is:
//!
//! 1. **Data** ([`data`]): adjusted price history (Yahoo Finance chart API),
//!    fundamentals and share counts (SEC XBRL company facts), and recent
//!    headlines (Yahoo Finance + Google News RSS).
//! 2. **Sentiment** ([`sentiment`]): headlines are scored by FinBERT running
//!    natively in Rust via `candle`, then aggregated per ticker into a
//!    recency-weighted score and a confidence.
//! 3. **Model** ([`model`]): a Ledoit-Wolf shrunk covariance matrix, a
//!    market-cap implied equilibrium prior, and Black-Litterman views built
//!    from the sentiment + fundamentals signal.
//! 4. **Optimization**: a long-only, position-capped mean-variance portfolio
//!    solved with accelerated projected gradient ascent.
//!
//! [`pipeline::Analyzer`] ties it together and produces a serializable
//! [`report::Report`], consumed by both the CLI and the web dashboard.

pub mod data;
pub mod http;
pub mod model;
pub mod params;
pub mod pipeline;
pub mod report;
pub mod research;
pub mod sentiment;
pub mod server;
