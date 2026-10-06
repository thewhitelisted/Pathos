//! Fetch FinBERT weights from the Hugging Face Hub, pinned and verified.
//!
//! `ProsusAI/finbert` only publishes a legacy (pre-zip) PyTorch pickle on
//! `main`, which candle cannot read. Hugging Face's own `SFconvertbot` opened
//! a pull request adding a `model.safetensors` conversion; we pin that exact
//! commit and verify every file's SHA-256 so the download is reproducible and
//! tamper-evident.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

const REPO: &str = "ProsusAI/finbert";
/// refs/pr/29 — "Adding `safetensors` variant of this model" by SFconvertbot.
const REVISION: &str = "7db323f79b751944bcfa66298ec06977e4518306";

pub const FILES: &[(&str, &str)] = &[
    (
        "config.json",
        "f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
    ),
    (
        "vocab.txt",
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
    (
        "model.safetensors",
        "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00",
    ),
];

pub fn default_model_dir() -> PathBuf {
    std::env::var("PATHOS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| crate::http::cache_root().join("models").join("finbert"))
}

/// Ensure all model files exist in `dir`, downloading any that are missing.
pub async fn ensure_model(client: &reqwest::Client, dir: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("creating {}", dir.display()))?;
    for (name, sha) in FILES {
        let path = dir.join(name);
        if tokio::fs::try_exists(&path).await? {
            continue;
        }
        let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{name}");
        tracing::info!(
            file = name,
            "downloading FinBERT weights (one-time, ~440 MB)"
        );
        download_verified(client, &url, &path, sha).await?;
    }
    Ok(())
}

async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha: &str,
) -> Result<()> {
    let mut resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let part = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&part).await?;
    let mut hasher = Sha256::new();
    let mut done: u64 = 0;
    let mut next_log = 0.0;
    while let Some(chunk) = resp.chunk().await.context("download interrupted")? {
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if let Some(total) = total {
            let pct = done as f64 / total as f64 * 100.0;
            if pct >= next_log {
                tracing::info!("  {pct:.0}% of {:.0} MB", total as f64 / 1e6);
                next_log += 25.0;
            }
        }
    }
    file.flush().await?;
    drop(file);

    let got: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if got != sha {
        let _ = tokio::fs::remove_file(&part).await;
        bail!("checksum mismatch for {url}: expected {sha}, got {got}");
    }
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}
