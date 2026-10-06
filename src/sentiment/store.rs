//! Persistent memo of headline -> FinBERT probabilities.
//!
//! Inference is by far the most expensive step (~13 headlines/s on a 4-core
//! CPU), and historical evaluation scores tens of thousands of headlines, so
//! scores are appended to a JSON-lines file and reused across runs.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{Probs, SentimentModel};

/// Texts scored per model call; also how often progress is flushed to disk.
const CHUNK: usize = 256;

#[derive(Serialize, Deserialize)]
struct Record {
    t: String,
    p: [f64; 3],
}

pub struct ScoreStore {
    path: Option<PathBuf>,
    map: Mutex<HashMap<String, Probs>>,
}

impl ScoreStore {
    /// Scores depend on the model weights, so each precision has its own file.
    pub fn default_path(precision: super::Precision) -> PathBuf {
        crate::http::cache_root().join(format!("scores-{}.jsonl", precision.tag()))
    }

    /// All texts in the store, in arbitrary order.
    pub fn texts(&self) -> Vec<String> {
        self.map
            .lock()
            .expect("score store poisoned")
            .keys()
            .cloned()
            .collect()
    }

    pub fn in_memory() -> Self {
        Self {
            path: None,
            map: Mutex::default(),
        }
    }

    /// Open (or create) a store backed by `path`. Corrupt lines, e.g. from an
    /// interrupted write, are skipped.
    pub fn open(path: PathBuf) -> Result<Self> {
        let mut map = HashMap::new();
        if path.exists() {
            let file = std::fs::File::open(&path)
                .with_context(|| format!("opening {}", path.display()))?;
            for line in BufReader::new(file).lines() {
                if let Ok(r) = serde_json::from_str::<Record>(&line?) {
                    let [positive, negative, neutral] = r.p;
                    map.insert(
                        r.t,
                        Probs {
                            positive,
                            negative,
                            neutral,
                        },
                    );
                }
            }
        }
        Ok(Self {
            path: Some(path),
            map: Mutex::new(map),
        })
    }

    pub fn len(&self) -> usize {
        self.map.lock().expect("score store poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Texts not yet in the store (de-duplicated, order preserved).
    pub fn missing(&self, texts: &[String]) -> Vec<String> {
        let map = self.map.lock().expect("score store poisoned");
        let mut seen = HashSet::new();
        texts
            .iter()
            .filter(|t| !map.contains_key(*t) && seen.insert(t.as_str()))
            .cloned()
            .collect()
    }

    /// Score `texts`, running `model` only on unseen ones. Blocking: call from
    /// a blocking thread. `progress(done, total)` is called after each chunk.
    pub fn score(
        &self,
        model: &dyn SentimentModel,
        texts: &[String],
        progress: &dyn Fn(usize, usize),
    ) -> Result<Vec<Probs>> {
        let missing = self.missing(texts);
        let total = missing.len();
        for (i, chunk) in missing.chunks(CHUNK).enumerate() {
            let probs = model.predict(chunk)?;
            self.insert(chunk, &probs)?;
            progress((i * CHUNK + chunk.len()).min(total), total);
        }
        let map = self.map.lock().expect("score store poisoned");
        Ok(texts
            .iter()
            .map(|t| map.get(t).copied().unwrap_or_default())
            .collect())
    }

    fn insert(&self, texts: &[String], probs: &[Probs]) -> Result<()> {
        if let Some(path) = &self.path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            let mut buf = String::new();
            for (t, p) in texts.iter().zip(probs) {
                let rec = Record {
                    t: t.clone(),
                    p: [p.positive, p.negative, p.neutral],
                };
                buf.push_str(&serde_json::to_string(&rec)?);
                buf.push('\n');
            }
            file.write_all(buf.as_bytes())?;
        }
        let mut map = self.map.lock().expect("score store poisoned");
        map.extend(texts.iter().cloned().zip(probs.iter().copied()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting(AtomicUsize);
    impl SentimentModel for Counting {
        fn predict(&self, texts: &[String]) -> Result<Vec<Probs>> {
            self.0.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| Probs {
                    positive: t.len() as f64 / 100.0,
                    negative: 0.0,
                    neutral: 0.0,
                })
                .collect())
        }
    }

    #[test]
    fn persists_and_only_scores_new_texts() {
        let dir = std::env::temp_dir().join(format!("pathos-store-{}", std::process::id()));
        let path = dir.join("scores.jsonl");
        let _ = std::fs::remove_file(&path);
        let model = Counting(AtomicUsize::new(0));
        let texts: Vec<String> = vec!["aa".into(), "bbb".into(), "aa".into()];

        let store = ScoreStore::open(path.clone()).unwrap();
        let first = store.score(&model, &texts, &|_, _| {}).unwrap();
        assert_eq!(model.0.load(Ordering::SeqCst), 2, "duplicates scored once");
        assert_eq!(first[0], first[2]);

        let reopened = ScoreStore::open(path.clone()).unwrap();
        assert_eq!(reopened.len(), 2);
        let again = reopened.score(&model, &texts, &|_, _| {}).unwrap();
        assert_eq!(
            model.0.load(Ordering::SeqCst),
            2,
            "nothing rescored after reopen"
        );
        assert_eq!(again, first);
        let _ = std::fs::remove_dir_all(dir);
    }
}
