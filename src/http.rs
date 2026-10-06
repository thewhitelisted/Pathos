//! Shared HTTP client with a small on-disk response cache.
//!
//! Every upstream we talk to (SEC, Yahoo, Google News) is rate limited or
//! slow, so responses are cached on disk keyed by URL with a per-call TTL.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

/// Yahoo rate-limits full browser UAs that arrive without cookies, but
/// accepts an honest client identifier.
const CLIENT_UA: &str = concat!(
    "Mozilla/5.0 (compatible; pathos/",
    env!("CARGO_PKG_VERSION"),
    ")"
);

/// SEC EDGAR rejects requests whose User-Agent lacks a contact email
/// (<https://www.sec.gov/os/accessing-edgar-data>), so it must be configured.
pub const SEC_UA_ENV: &str = "SEC_USER_AGENT";

/// Root directory for cached responses and downloaded model weights.
pub fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("PATHOS_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("pathos");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("pathos");
    }
    PathBuf::from(".pathos-cache")
}

#[derive(Clone, Copy, Debug)]
pub enum Upstream {
    Sec,
    Yahoo,
    News,
}

#[derive(Clone)]
pub struct Fetcher {
    client: reqwest::Client,
    cache_dir: PathBuf,
    sec_user_agent: Option<String>,
    /// SEC allows at most 10 requests/second; keep well under that.
    sec_permits: Arc<Semaphore>,
    /// Be polite to news feeds during bulk historical collection.
    news_permits: Arc<Semaphore>,
}

const MAX_ATTEMPTS: u32 = 4;

impl Fetcher {
    pub fn new() -> Result<Self> {
        Self::with_cache_dir(cache_root().join("http"))
    }

    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(CLIENT_UA)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("building HTTP client")?;
        let sec_user_agent = std::env::var(SEC_UA_ENV)
            .ok()
            .map(|ua| ua.trim().to_string())
            .filter(|ua| ua.contains('@'));
        Ok(Self {
            client,
            cache_dir,
            sec_user_agent,
            sec_permits: Arc::new(Semaphore::new(4)),
            news_permits: Arc::new(Semaphore::new(3)),
        })
    }

    pub fn has_sec_access(&self) -> bool {
        self.sec_user_agent.is_some()
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// GET `url` as text, serving from the disk cache if a copy younger than
    /// `ttl` exists. Failed requests are never cached.
    pub async fn get_text(&self, url: &str, upstream: Upstream, ttl: Duration) -> Result<String> {
        let path = self.cache_path(url);
        if let Some(body) = read_fresh(&path, ttl).await {
            tracing::debug!(url, "cache hit");
            return Ok(body);
        }

        if let Upstream::Sec = upstream
            && self.sec_user_agent.is_none()
        {
            bail!("SEC data needs a contact email: set {SEC_UA_ENV}=\"Your Name you@example.com\"");
        }
        let _permit = match upstream {
            Upstream::Sec => Some(self.sec_permits.acquire().await?),
            Upstream::News => Some(self.news_permits.acquire().await?),
            Upstream::Yahoo => None,
        };
        let body = self.fetch_with_retry(url, upstream).await?;

        if let Err(e) = write_cache(&path, &body).await {
            tracing::warn!(error = %e, "failed to write HTTP cache");
        }
        Ok(body)
    }

    /// Retry transient failures (timeouts, 429, 5xx) with exponential backoff.
    async fn fetch_with_retry(&self, url: &str, upstream: Upstream) -> Result<String> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut req = self.client.get(url);
            if let (Upstream::Sec, Some(ua)) = (upstream, &self.sec_user_agent) {
                req = req.header(reqwest::header::USER_AGENT, ua);
            }
            tracing::debug!(url, attempt, "fetching");
            let outcome = match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp
                            .text()
                            .await
                            .with_context(|| format!("reading body of {url}"));
                    }
                    let transient = status.as_u16() == 429 || status.is_server_error();
                    (anyhow!("GET {url} returned HTTP {status}"), transient)
                }
                Err(e) => {
                    let transient = e.is_timeout() || e.is_connect() || e.is_request();
                    (anyhow!(e).context(format!("GET {url}")), transient)
                }
            };
            match outcome {
                (err, true) if attempt < MAX_ATTEMPTS => {
                    let wait = Duration::from_secs(2u64.pow(attempt));
                    tracing::debug!(url, ?wait, error = %err, "retrying");
                    tokio::time::sleep(wait).await;
                }
                (err, _) => return Err(err),
            }
        }
    }

    fn cache_path(&self, url: &str) -> PathBuf {
        let digest = Sha256::digest(url.as_bytes());
        let name: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
        self.cache_dir.join(name)
    }
}

async fn read_fresh(path: &Path, ttl: Duration) -> Option<String> {
    let meta = tokio::fs::metadata(path).await.ok()?;
    let age = SystemTime::now()
        .duration_since(meta.modified().ok()?)
        .ok()?;
    if age > ttl {
        return None;
    }
    tokio::fs::read_to_string(path).await.ok()
}

async fn write_cache(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    // Write-then-rename so concurrent readers never see a partial file.
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, body).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}
