//! FinBERT (`ProsusAI/finbert`) inference in pure Rust with candle.
//!
//! The architecture is `BertForSequenceClassification`: a BERT-base encoder,
//! a tanh pooler over the `[CLS]` token, and a linear classifier over the
//! labels `positive`, `negative`, `neutral` (in that index order).

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use super::{Probs, SentimentModel};

/// Headlines are short; 128 tokens covers essentially all of them and keeps
/// attention cost (quadratic in length) low.
const MAX_TOKENS: usize = 128;
const BATCH_SIZE: usize = 32;

pub struct FinBert {
    bert: BertModel,
    pooler: Linear,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl FinBert {
    /// Load from a directory containing `config.json`, `vocab.txt` and
    /// `model.safetensors` (see [`super::download::ensure_model`]).
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?,
        )
        .context("parsing config.json")?;

        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only and was checksum
        // verified on download; nothing else in-process mutates it.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let bert = BertModel::load(vb.pp("bert"), &config).context("loading BERT encoder")?;
        let pooler = linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("bert.pooler.dense"),
        )?;
        let classifier = linear(config.hidden_size, 3, vb.pp("classifier"))?;

        let tokenizer = build_tokenizer(&dir.join("vocab.txt"))?;
        Ok(Self {
            bert,
            pooler,
            classifier,
            tokenizer,
            device,
        })
    }

    fn predict_batch(&self, texts: &[String]) -> Result<Vec<Probs>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        let to_tensor = |f: &dyn Fn(&tokenizers::Encoding) -> &[u32]| -> Result<Tensor> {
            let rows = encodings
                .iter()
                .map(|e| Tensor::new(f(e), &self.device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Ok(Tensor::stack(&rows, 0)?)
        };
        let ids = to_tensor(&|e| e.get_ids())?;
        let type_ids = to_tensor(&|e| e.get_type_ids())?;
        let mask = to_tensor(&|e| e.get_attention_mask())?;

        let hidden = self.bert.forward(&ids, &type_ids, Some(&mask))?;
        let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let pooled = self.pooler.forward(&cls)?.tanh()?;
        let logits = self.classifier.forward(&pooled)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;

        Ok(probs
            .into_iter()
            .map(|p| Probs {
                positive: f64::from(p[0]),
                negative: f64::from(p[1]),
                neutral: f64::from(p[2]),
            })
            .collect())
    }
}

impl SentimentModel for FinBert {
    fn predict(&self, texts: &[String]) -> Result<Vec<Probs>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(BATCH_SIZE) {
            out.extend(self.predict_batch(chunk)?);
        }
        Ok(out)
    }
}

/// Recreate the `bert-base-uncased` WordPiece tokenizer from `vocab.txt`
/// (the repo ships no `tokenizer.json`).
fn build_tokenizer(vocab: &Path) -> Result<Tokenizer> {
    let vocab = vocab
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF-8 vocab path"))?;
    let wordpiece = WordPiece::from_file(vocab)
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| anyhow!("loading vocab: {e}"))?;
    let token_id =
        |t: &str| wordpiece_id(&wordpiece, t).ok_or_else(|| anyhow!("vocab is missing {t}"));
    let (cls, sep, pad) = (token_id("[CLS]")?, token_id("[SEP]")?, token_id("[PAD]")?);

    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer
        .with_normalizer(Some(BertNormalizer::new(true, true, None, true)))
        .with_pre_tokenizer(Some(BertPreTokenizer))
        .with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), sep),
            ("[CLS]".into(), cls),
        )))
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: pad,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length: MAX_TOKENS,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("configuring truncation: {e}"))?;
    Ok(tokenizer)
}

fn wordpiece_id(wp: &WordPiece, token: &str) -> Option<u32> {
    use tokenizers::Model;
    wp.token_to_id(token)
}
