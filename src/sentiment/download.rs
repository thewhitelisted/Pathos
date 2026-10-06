//! Fetch FinBERT weights from the Hugging Face Hub, pinned and verified.
//!
//! `ProsusAI/finbert` only publishes a legacy (pre-zip) PyTorch pickle on
//! `main`, which candle cannot read. Hugging Face's own `SFconvertbot` opened
//! a pull request adding a `model.safetensors` conversion; we pin that exact
//! commit and verify every file's SHA-256 so the download is reproducible and
//! tamper-evident.
//!
//! For [`Precision::Q8`] the verified checkpoint is then quantized locally to
//! int8 and the 438 MB original is deleted (unless asked to keep it), so the
//! model occupies ~120 MB on disk and in memory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use super::Precision;

const REPO: &str = "ProsusAI/finbert";
/// refs/pr/29 — "Adding `safetensors` variant of this model" by SFconvertbot.
const REVISION: &str = "7db323f79b751944bcfa66298ec06977e4518306";

/// The verified f32 checkpoint and the locally quantized int8 model.
pub const F32_FILE: &str = "model.safetensors";
pub const Q8_FILE: &str = "finbert-q8_0.gguf";

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

/// Ensure everything `precision` needs exists in `dir`, downloading and
/// quantizing as necessary. `keep_f32` keeps the original checkpoint after
/// quantizing (needed to switch back to f32 without re-downloading).
pub async fn ensure_model(
    client: &reqwest::Client,
    dir: &Path,
    precision: Precision,
    keep_f32: bool,
) -> Result<()> {
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("creating {}", dir.display()))?;
    let q8 = dir.join(Q8_FILE);
    let need_f32 = precision == Precision::F32 || !tokio::fs::try_exists(&q8).await?;
    for (name, sha) in FILES {
        let path = dir.join(name);
        if (*name == F32_FILE && !need_f32) || tokio::fs::try_exists(&path).await? {
            continue;
        }
        let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{name}");
        tracing::info!(file = name, "downloading FinBERT (one-time)");
        download_verified(client, &url, &path, sha).await?;
    }

    if precision == Precision::Q8 && !tokio::fs::try_exists(&q8).await? {
        tracing::info!("quantizing FinBERT to int8 (one-time)");
        let (src, dst) = (dir.join(F32_FILE), q8.clone());
        tokio::task::spawn_blocking(move || super::qbert::quantize_checkpoint(&src, &dst))
            .await??;
        let size = tokio::fs::metadata(&q8).await?.len();
        tracing::info!("wrote {} ({:.0} MB)", q8.display(), size as f64 / 1e6);
        if !keep_f32 {
            tokio::fs::remove_file(dir.join(F32_FILE)).await?;
            tracing::info!("removed the f32 checkpoint (use --keep-f32 to keep it)");
        }
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
