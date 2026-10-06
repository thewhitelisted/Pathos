//! Int8-quantized BERT-for-sequence-classification.
//!
//! candle ships quantized versions of many decoder LLMs but not of BERT, so
//! this is a straightforward re-implementation of the encoder on top of
//! `QMatMul`. Every weight matrix (embeddings, attention, feed-forward,
//! pooler, classifier) is stored as GGML `Q8_0` — 8-bit integers with one f16
//! scale per block of 32 values, about 1.06 bytes per parameter instead of 4.
//! Biases and LayerNorm parameters stay f32: they are tiny and precision-
//! sensitive.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{Device, Module, Tensor};
use candle_nn::LayerNorm;
use candle_transformers::models::bert::Config;
use candle_transformers::quantized_nn::{Embedding, Linear, layer_norm, linear};
use candle_transformers::quantized_var_builder::VarBuilder;

/// Convert the f32 safetensors checkpoint into a Q8_0 GGUF file.
///
/// Tensors are written in sorted order so the output is byte-for-byte
/// deterministic for a given input.
pub fn quantize_checkpoint(safetensors: &Path, gguf: &Path) -> Result<()> {
    let tensors = candle_core::safetensors::load(safetensors, &Device::Cpu)
        .with_context(|| format!("reading {}", safetensors.display()))?;
    let mut names: Vec<&String> = tensors
        .keys()
        .filter(|n| !n.ends_with("position_ids"))
        .collect();
    names.sort();

    let mut quantized: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    for name in names {
        let t = &tensors[name];
        let is_matrix = t.rank() == 2 && t.dim(1)? % GgmlDType::Q8_0.block_size() == 0;
        let dtype = if is_matrix {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        quantized.push((name.clone(), QTensor::quantize(t, dtype)?));
    }

    let part = gguf.with_extension("part");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&part)?);
    let arch = gguf_file::Value::String("bert".to_string());
    let refs: Vec<(&str, &QTensor)> = quantized.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs)?;
    drop(file);
    std::fs::rename(&part, gguf)?;
    Ok(())
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    norm: LayerNorm,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            query: linear(h, h, vb.pp("self.query"))?,
            key: linear(h, h, vb.pp("self.key"))?,
            value: linear(h, h, vb.pp("self.value"))?,
            output: linear(h, h, vb.pp("output.dense"))?,
            norm: layer_norm(h, c.layer_norm_eps, vb.pp("output.LayerNorm"))?,
            heads: c.num_attention_heads,
            head_dim: h / c.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, s, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = split(self.query.forward(x)?)?;
        let k = split(self.key.forward(x)?)?;
        let v = split(self.value.forward(x)?)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.reshape((b, s, h))?;
        Ok(self.norm.forward(&(self.output.forward(&ctx)? + x)?)?)
    }
}

struct Layer {
    attention: Attention,
    intermediate: Linear,
    output: Linear,
    norm: LayerNorm,
}

impl Layer {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("attention"), c)?,
            intermediate: linear(
                c.hidden_size,
                c.intermediate_size,
                vb.pp("intermediate.dense"),
            )?,
            output: linear(c.intermediate_size, c.hidden_size, vb.pp("output.dense"))?,
            norm: layer_norm(c.hidden_size, c.layer_norm_eps, vb.pp("output.LayerNorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = self.attention.forward(x, mask)?;
        // BERT's "gelu" is the exact erf form.
        let ff = self
            .output
            .forward(&self.intermediate.forward(&x)?.gelu_erf()?)?;
        Ok(self.norm.forward(&(ff + x)?)?)
    }
}

pub struct QuantizedBert {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    pooler: Linear,
    classifier: Linear,
}

impl QuantizedBert {
    pub fn load(gguf: &Path, config: &Config, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_gguf(gguf, device)
            .with_context(|| format!("reading {}", gguf.display()))?;
        let e = vb.pp("bert.embeddings");
        let h = config.hidden_size;
        let layers = (0..config.num_hidden_layers)
            .map(|i| Layer::load(vb.pp(format!("bert.encoder.layer.{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        if layers.is_empty() {
            bail!("config has no encoder layers");
        }
        Ok(Self {
            word: Embedding::new(config.vocab_size, h, e.pp("word_embeddings"))?,
            position: Embedding::new(
                config.max_position_embeddings,
                h,
                e.pp("position_embeddings"),
            )?,
            token_type: Embedding::new(config.type_vocab_size, h, e.pp("token_type_embeddings"))?,
            embed_norm: layer_norm(h, config.layer_norm_eps, e.pp("LayerNorm"))?,
            layers,
            pooler: linear(h, h, vb.pp("bert.pooler.dense"))?,
            classifier: linear(h, 3, vb.pp("classifier"))?,
        })
    }

    /// Class logits for a padded batch.
    pub fn forward(&self, ids: &Tensor, type_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (_, s) = ids.dims2()?;
        let positions = Tensor::arange(0u32, s as u32, ids.device())?.unsqueeze(0)?;
        let x = self
            .word
            .forward(ids)?
            .broadcast_add(&self.position.forward(&positions)?)?
            .add(&self.token_type.forward(type_ids)?)?;
        let mut x = self.embed_norm.forward(&x)?;

        // (b, s) {0,1} -> additive (b, 1, 1, s) mask: 0 to attend, -inf-ish to skip.
        let mask = mask.to_dtype(x.dtype())?.unsqueeze(1)?.unsqueeze(1)?;
        let mask = ((mask.ones_like()? - &mask)? * f64::from(f32::MIN))?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        let cls = x.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let pooled = self.pooler.forward(&cls)?.tanh()?;
        Ok(self.classifier.forward(&pooled)?)
    }
}
