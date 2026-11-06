//! T3-Nano forward pass: GPT-2 small, `gelu_new`, absolute positions, Turbo sampler.
//!
//! This produces speech-token ids. It does not produce audio, and
//! [`crate::text_to_cloned_speech_ready`] stays false.
//!
//! `inference_turbo` concatenates speaker projection, prompt-speech embeddings,
//! text embeddings, and a single start-of-speech embedding (id 6561). GPT-2 adds
//! `wpe` over that whole sequence. Each new speech id is one more token. The
//! returned ids omit the start id and a trailing stop id (6562). The first
//! sample is not tested for stop until the final trim, matching upstream: the
//! loop is what breaks early, and the trim always drops a final stop.
//!
//! Logit warpers run in upstream order: temperature, top-k, top-p, then
//! repetition penalty. The first step's penalty ids are just the start token.
//! Later steps penalize the ids sampled so far, not the text and not the start
//! token. `wte` is in the checkpoint and is not loaded.

use crate::t3::{
    self, TurboDecode, N_EMBD, N_HEAD, N_LAYER, N_POSITIONS, SPEAKER_EMBED, SPEECH_TOKENS,
    START_SPEECH, STOP_SPEECH, TEXT_TOKENS,
};
use crate::tokenizer::{self, REPO_NANO, REPO_TURBO};
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{softmax, tanh};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};

pub const T3_FILENAME: &str = "t3_nano_v1.safetensors";

struct Block {
    n_embd: usize,
    n_head: usize,
    c_attn_w: Tensor<2>,
    c_attn_b: Tensor<1>,
    c_proj_w: Tensor<2>,
    c_proj_b: Tensor<1>,
    ln1_w: Tensor<1>,
    ln1_b: Tensor<1>,
    ln2_w: Tensor<1>,
    ln2_b: Tensor<1>,
    fc_w: Tensor<2>,
    fc_b: Tensor<1>,
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
}

struct Kv {
    k: Tensor<3>,
    v: Tensor<3>,
}

/// One Nano (or test-sized) T3 network on the flex CPU device.
pub struct T3Nano {
    device: Device,
    n_embd: usize,
    n_head: usize,
    n_positions: usize,
    start_speech: u32,
    stop_speech: u32,
    spkr_w: Tensor<2>,
    spkr_b: Tensor<1>,
    text_emb: Tensor<2>,
    speech_emb: Tensor<2>,
    speech_head_w: Tensor<2>,
    speech_head_b: Tensor<1>,
    wpe: Tensor<2>,
    blocks: Vec<Block>,
    ln_f_w: Tensor<1>,
    ln_f_b: Tensor<1>,
}

/// What the caller feeds `inference_turbo` before sampling.
#[derive(Clone, Debug)]
pub struct T3Context {
    /// Voice-encoder embedding, length `speaker_embed_size` (256 for Nano).
    pub speaker: Vec<f32>,
    /// S3 prompt tokens. Empty is allowed. Nano crops these to 375 upstream.
    pub prompt_speech: Vec<u32>,
    /// GPT-2 text ids. No start/stop text tokens; Turbo does not add them.
    pub text: Vec<u32>,
}

impl T3Nano {
    pub fn cpu() -> Device {
        Device::flex()
    }

    /// `PYRE_CHATTERBOX_T3` (file or directory), then `PYRE_CHATTERBOX_NANO`,
    /// then the HF hub cache. Does not download.
    pub fn discover() -> Option<PathBuf> {
        discover_t3()
    }

    /// Load `t3_nano_v1.safetensors`. Refuses a file whose header is not the
    /// Nano contract. Does not copy `tfmr.wte`.
    pub fn load(path: &Path) -> Result<Self> {
        t3::check_checkpoint(path)?;
        Self::load_shaped(path, N_HEAD, START_SPEECH, STOP_SPEECH)
    }

    /// Load a safetensors file with the Nano names and a caller-supplied head
    /// count. Used by the tiny NumPy fixture as well as the real checkpoint.
    pub fn load_shaped(
        path: &Path,
        n_head: usize,
        start_speech: u32,
        stop_speech: u32,
    ) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("safetensors")?;
        let device = Self::cpu();
        let spkr_w = named2(&st, "cond_enc.spkr_enc.weight", &device)?;
        let n_embd = spkr_w.dims()[0];
        let text_emb = named2(&st, "text_emb.weight", &device)?;
        let speech_emb = named2(&st, "speech_emb.weight", &device)?;
        let speech_head_w = named2(&st, "speech_head.weight", &device)?;
        let wpe = named2(&st, "tfmr.wpe.weight", &device)?;
        if text_emb.dims()[1] != n_embd || speech_emb.dims()[1] != n_embd {
            bail!("embedding width does not match the speaker projection");
        }
        if n_embd % n_head != 0 {
            bail!("n_embd {n_embd} is not divisible by n_head {n_head}");
        }
        let mut blocks = Vec::new();
        for i in 0.. {
            let key = format!("tfmr.h.{i}.attn.c_attn.weight");
            if st.tensor(&key).is_err() {
                break;
            }
            let p = format!("tfmr.h.{i}");
            blocks.push(Block {
                n_embd,
                n_head,
                c_attn_w: named2(&st, &format!("{p}.attn.c_attn.weight"), &device)?,
                c_attn_b: named1(&st, &format!("{p}.attn.c_attn.bias"), &device)?,
                c_proj_w: named2(&st, &format!("{p}.attn.c_proj.weight"), &device)?,
                c_proj_b: named1(&st, &format!("{p}.attn.c_proj.bias"), &device)?,
                ln1_w: named1(&st, &format!("{p}.ln_1.weight"), &device)?,
                ln1_b: named1(&st, &format!("{p}.ln_1.bias"), &device)?,
                ln2_w: named1(&st, &format!("{p}.ln_2.weight"), &device)?,
                ln2_b: named1(&st, &format!("{p}.ln_2.bias"), &device)?,
                fc_w: named2(&st, &format!("{p}.mlp.c_fc.weight"), &device)?,
                fc_b: named1(&st, &format!("{p}.mlp.c_fc.bias"), &device)?,
                proj_w: named2(&st, &format!("{p}.mlp.c_proj.weight"), &device)?,
                proj_b: named1(&st, &format!("{p}.mlp.c_proj.bias"), &device)?,
            });
        }
        if blocks.is_empty() {
            bail!("no GPT-2 blocks in {}", path.display());
        }
        let _ = (
            N_LAYER,
            N_EMBD,
            N_POSITIONS,
            TEXT_TOKENS,
            SPEECH_TOKENS,
            SPEAKER_EMBED,
        );
        Ok(Self {
            device: device.clone(),
            n_embd,
            n_head,
            n_positions: wpe.dims()[0],
            start_speech,
            stop_speech,
            spkr_w,
            spkr_b: named1(&st, "cond_enc.spkr_enc.bias", &device)?,
            text_emb,
            speech_emb,
            speech_head_w,
            speech_head_b: named1(&st, "speech_head.bias", &device)?,
            wpe,
            blocks,
            ln_f_w: named1(&st, "tfmr.ln_f.weight", &device)?,
            ln_f_b: named1(&st, "tfmr.ln_f.bias", &device)?,
        })
    }

    /// Speech-head logits at the last position of `cond + text + speech`.
    /// `speech` is the speech side only. Pass `[START_SPEECH]` to match the
    /// first `inference_turbo` forward.
    pub fn speech_logits(&self, ctx: &T3Context, speech: &[u32]) -> Result<Vec<f32>> {
        let embeds = self.input_embeds(ctx, speech)?;
        let (hidden, _) = self.forward_hidden(embeds, None, 0)?;
        self.head(&hidden)
    }

    /// Greedy `inference_turbo`: argmax after the Turbo warpers. Deterministic.
    /// `max_gen_len` is the loop bound after the first sample, as in upstream.
    pub fn generate_greedy(
        &self,
        ctx: &T3Context,
        decode: &TurboDecode,
        max_gen_len: usize,
    ) -> Result<Vec<u32>> {
        self.generate(ctx, decode, max_gen_len, NextId::Greedy)
    }

    /// Same warpers as [`generate_greedy`](Self::generate_greedy), then one
    /// categorical draw. The seed fixes the draw. It does not skip it.
    pub fn generate_seeded(
        &self,
        ctx: &T3Context,
        decode: &TurboDecode,
        max_gen_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        self.generate(ctx, decode, max_gen_len, NextId::Seeded(seed))
    }

    fn generate(
        &self,
        ctx: &T3Context,
        decode: &TurboDecode,
        max_gen_len: usize,
        next: NextId,
    ) -> Result<Vec<u32>> {
        let mut rng = match next {
            NextId::Seeded(seed) => Some(SampleRng::new(seed)),
            NextId::Greedy => None,
        };
        let mut speech = vec![self.start_speech];
        let mut generated: Vec<u32> = Vec::new();
        let mut cache: Option<Vec<Kv>> = None;
        let mut pos = 0usize;
        for step in 0..(1 + max_gen_len) {
            let embeds = if step == 0 {
                self.input_embeds(ctx, &speech)?
            } else {
                self.embed(&self.speech_emb, &[*speech.last().unwrap()])?
            };
            let t = embeds.dims()[0];
            if pos + t > self.n_positions {
                bail!(
                    "sequence length {} exceeds wpe {}",
                    pos + t,
                    self.n_positions
                );
            }
            let (hidden, new_cache) = self.forward_hidden(embeds, cache, pos)?;
            pos += t;
            cache = Some(new_cache);
            let raw = self.head(&hidden)?;
            let penalty_ids: Vec<u32> = if step == 0 {
                vec![self.start_speech]
            } else {
                generated.clone()
            };
            let processed = process_logits(&raw, &penalty_ids, decode);
            if !processed.iter().any(|v| v.is_finite()) {
                break;
            }
            let id = match next {
                NextId::Greedy => argmax(&processed),
                NextId::Seeded(_) => sample_logits(&processed, rng.as_mut().unwrap()),
            };
            generated.push(id);
            speech.push(id);
            if step > 0 && id == self.stop_speech {
                break;
            }
        }
        if generated.last() == Some(&self.stop_speech) {
            generated.pop();
        }
        Ok(generated)
    }

    fn input_embeds(&self, ctx: &T3Context, speech: &[u32]) -> Result<Tensor<2>> {
        if ctx.text.is_empty() {
            bail!("text tokens are empty");
        }
        if ctx.speaker.len() != self.spkr_w.dims()[1] {
            bail!(
                "speaker embedding is {}, model expects {}",
                ctx.speaker.len(),
                self.spkr_w.dims()[1]
            );
        }
        let speaker = from2(&ctx.speaker, 1, ctx.speaker.len(), &self.device);
        let spkr = speaker
            .matmul(self.spkr_w.clone().transpose())
            .add(self.spkr_b.clone().unsqueeze());
        let mut parts = vec![spkr];
        if !ctx.prompt_speech.is_empty() {
            parts.push(self.embed(&self.speech_emb, &ctx.prompt_speech)?);
        }
        parts.push(self.embed(&self.text_emb, &ctx.text)?);
        if !speech.is_empty() {
            parts.push(self.embed(&self.speech_emb, speech)?);
        }
        Ok(Tensor::cat(parts, 0))
    }

    fn embed(&self, table: &Tensor<2>, ids: &[u32]) -> Result<Tensor<2>> {
        let vocab = table.dims()[0];
        let d = table.dims()[1];
        let mut rows = Vec::with_capacity(ids.len());
        for &id in ids {
            if id as usize >= vocab {
                bail!("token {id} is outside the embedding table ({vocab})");
            }
            let i = id as usize;
            rows.push(table.clone().slice([i..i + 1, 0..d]));
        }
        Ok(Tensor::cat(rows, 0))
    }

    fn forward_hidden(
        &self,
        embeds: Tensor<2>,
        cache: Option<Vec<Kv>>,
        pos0: usize,
    ) -> Result<(Tensor<2>, Vec<Kv>)> {
        let t = embeds.dims()[0];
        let mut h = embeds.add(self.wpe.clone().slice([pos0..pos0 + t, 0..self.n_embd]));
        let mut next = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let past = cache.as_ref().map(|c| &c[i]);
            let (nh, kv) = block.forward(h, past);
            h = nh;
            next.push(kv);
        }
        Ok((layer_norm(h, &self.ln_f_w, &self.ln_f_b), next))
    }

    fn head(&self, hidden: &Tensor<2>) -> Result<Vec<f32>> {
        let t = hidden.dims()[0];
        let last = hidden.clone().slice([t - 1..t, 0..self.n_embd]);
        let logits = last
            .matmul(self.speech_head_w.clone().transpose())
            .add(self.speech_head_b.clone().unsqueeze());
        to_vec2(logits).map(|v| v)
    }
}

enum NextId {
    Greedy,
    Seeded(u64),
}

struct SampleRng {
    state: u64,
}

impl SampleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample_logits(scores: &[f32], rng: &mut SampleRng) -> u32 {
    let max = scores
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(scores);
    }
    let mut sum = 0.0f64;
    let mut weights = Vec::with_capacity(scores.len());
    for s in scores {
        let w = if s.is_finite() {
            (*s as f64 - max as f64).exp()
        } else {
            0.0
        };
        weights.push(w);
        sum += w;
    }
    if sum == 0.0 {
        return argmax(scores);
    }
    let mut pick = rng.unit() * sum;
    for (i, w) in weights.iter().enumerate() {
        pick -= *w;
        if pick <= 0.0 {
            return i as u32;
        }
    }
    weights.len().saturating_sub(1) as u32
}

impl Block {
    fn forward(&self, x: Tensor<2>, past: Option<&Kv>) -> (Tensor<2>, Kv) {
        let residual = x.clone();
        let n = layer_norm(x, &self.ln1_w, &self.ln1_b);
        let qkv = n
            .matmul(self.c_attn_w.clone())
            .add(self.c_attn_b.clone().unsqueeze());
        let t = qkv.dims()[0];
        let d = self.n_embd;
        let q = split_heads(qkv.clone().slice([0..t, 0..d]), self.n_head);
        let mut k = split_heads(qkv.clone().slice([0..t, d..2 * d]), self.n_head);
        let mut v = split_heads(qkv.slice([0..t, 2 * d..3 * d]), self.n_head);
        let past_len = if let Some(past) = past {
            let s = past.k.dims()[1];
            k = Tensor::cat(vec![past.k.clone(), k], 1);
            v = Tensor::cat(vec![past.v.clone(), v], 1);
            s
        } else {
            0
        };
        let hd = self.n_embd / self.n_head;
        let scores = q
            .matmul(k.clone().swap_dims(1, 2))
            .mul_scalar((hd as f32).sqrt().recip());
        let scores = causal_add(scores, past_len);
        let ctx = merge_heads(softmax(scores, 2).matmul(v.clone()));
        let x = residual.add(
            ctx.matmul(self.c_proj_w.clone())
                .add(self.c_proj_b.clone().unsqueeze()),
        );
        let residual = x.clone();
        let mid = gelu_new(
            layer_norm(x, &self.ln2_w, &self.ln2_b)
                .matmul(self.fc_w.clone())
                .add(self.fc_b.clone().unsqueeze()),
        );
        let x = residual.add(
            mid.matmul(self.proj_w.clone())
                .add(self.proj_b.clone().unsqueeze()),
        );
        (x, Kv { k, v })
    }
}

/// Temperature, top-k, top-p, then repetition penalty. Same order as
/// `T3.inference_turbo`.
pub fn process_logits(logits: &[f32], input_ids: &[u32], decode: &TurboDecode) -> Vec<f32> {
    let mut scores: Vec<f64> = logits.iter().map(|v| *v as f64).collect();
    if decode.temperature > 0.0 && decode.temperature != 1.0 {
        let t = decode.temperature as f64;
        for s in &mut scores {
            *s /= t;
        }
    }
    if decode.top_k > 0 && scores.len() > 1 {
        let k = decode.top_k.min(scores.len()).max(1);
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kth = sorted[sorted.len() - k];
        for s in &mut scores {
            if *s < kth {
                *s = f64::NEG_INFINITY;
            }
        }
    }
    if decode.top_p < 1.0 && scores.len() > 1 {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let finite_max = order
            .iter()
            .filter_map(|&i| scores[i].is_finite().then_some(scores[i]))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut exps = vec![0.0f64; scores.len()];
        if finite_max.is_finite() {
            let mut sum = 0.0;
            for &i in &order {
                if scores[i].is_finite() {
                    let e = (scores[i] - finite_max).exp();
                    exps[i] = e;
                    sum += e;
                }
            }
            if sum > 0.0 {
                let mut cum = 0.0;
                let mut remove = vec![false; scores.len()];
                for (n, &i) in order.iter().enumerate() {
                    cum += exps[i] / sum;
                    remove[i] = cum <= (1.0 - decode.top_p as f64);
                    if n + 1 == order.len() {
                        remove[i] = false;
                    }
                }
                for (i, ban) in remove.into_iter().enumerate() {
                    if ban {
                        scores[i] = f64::NEG_INFINITY;
                    }
                }
            }
        }
    }
    if decode.repetition_penalty != 1.0 {
        let mut seen = input_ids.to_vec();
        seen.sort_unstable();
        seen.dedup();
        let p = decode.repetition_penalty as f64;
        for id in seen {
            let i = id as usize;
            if i >= scores.len() {
                continue;
            }
            if scores[i] < 0.0 {
                scores[i] *= p;
            } else {
                scores[i] /= p;
            }
        }
    }
    scores.into_iter().map(|v| v as f32).collect()
}

fn argmax(scores: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in scores.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    best_i as u32
}

fn layer_norm(x: Tensor<2>, w: &Tensor<1>, b: &Tensor<1>) -> Tensor<2> {
    let mean = x.clone().mean_dim(1);
    let centered = x.sub(mean);
    let var = centered.clone().powi_scalar(2).mean_dim(1);
    let normed = centered.div(var.add_scalar(1.0e-5).sqrt());
    normed.mul(w.clone().unsqueeze()).add(b.clone().unsqueeze())
}

fn gelu_new(x: Tensor<2>) -> Tensor<2> {
    let inner = x
        .clone()
        .add(x.clone().powi_scalar(3).mul_scalar(0.044715))
        .mul_scalar(0.7978845608028654);
    x.mul_scalar(0.5).mul(tanh(inner).add_scalar(1.0))
}

fn split_heads(x: Tensor<2>, n_head: usize) -> Tensor<3> {
    let [t, d] = x.dims();
    let hd = d / n_head;
    x.reshape([t, n_head, hd]).swap_dims(0, 1)
}

fn merge_heads(x: Tensor<3>) -> Tensor<2> {
    let [h, t, hd] = x.dims();
    x.swap_dims(0, 1).reshape([t, h * hd])
}

fn causal_add(scores: Tensor<3>, past: usize) -> Tensor<3> {
    let [h, t, s] = scores.dims();
    let mut mask = vec![0.0f32; t * s];
    for i in 0..t {
        let allow = past + i;
        for j in (allow + 1)..s {
            mask[i * s + j] = f32::NEG_INFINITY;
        }
    }
    let device = scores.device();
    let mask = from2(&mask, t, s, &device).unsqueeze::<3>();
    let _ = h;
    scores.add(mask)
}

fn named2(st: &safetensors::SafeTensors<'_>, name: &str, device: &Device) -> Result<Tensor<2>> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let shape = v.shape().to_vec();
    if shape.len() != 2 {
        bail!("{name}: rank {}", shape.len());
    }
    Ok(from2(&f32s(v)?, shape[0], shape[1], device))
}

fn named1(st: &safetensors::SafeTensors<'_>, name: &str, device: &Device) -> Result<Tensor<1>> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let rank = v.shape().len();
    if rank != 1 {
        bail!("{name}: rank {rank}");
    }
    Ok(from1(&f32s(v)?, device))
}

fn f32s(view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != safetensors::Dtype::F32 {
        bail!("expected f32");
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn from1(data: &[f32], device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

fn from2(data: &[f32], rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

fn to_vec2(t: Tensor<2>) -> Result<Vec<f32>> {
    t.into_data()
        .try_to_vec::<f32>()
        .map_err(|e| anyhow!("tensor to vec: {e:?}"))
}

/// `PYRE_CHATTERBOX_T3`, then `PYRE_CHATTERBOX_NANO`, then the HF hub cache
/// for Nano and Turbo. A directory must contain [`T3_FILENAME`] as a real
/// file or a blob symlink. Nothing is downloaded.
pub fn discover_t3() -> Option<PathBuf> {
    for key in ["PYRE_CHATTERBOX_T3", "PYRE_CHATTERBOX_NANO"] {
        if let Some(p) = std::env::var_os(key) {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
            if let Ok(found) = tokenizer::safe_file(&p, T3_FILENAME) {
                return Some(found);
            }
        }
    }
    let cache = hf_cache_dir();
    for repo in [REPO_NANO, REPO_TURBO] {
        let root = cache.join(format!("models--{}", repo.replace('/', "--")));
        if let Some(path) = snapshot_weight(&root) {
            return Some(path);
        }
    }
    None
}

fn snapshot_weight(repo_root: &Path) -> Option<PathBuf> {
    let mut snaps = Vec::new();
    if let Ok(rev) = std::fs::read_to_string(repo_root.join("refs").join("main")) {
        snaps.push(repo_root.join("snapshots").join(rev.trim()));
    }
    if let Ok(dir) = std::fs::read_dir(repo_root.join("snapshots")) {
        for e in dir.flatten() {
            snaps.push(e.path());
        }
    }
    snaps
        .into_iter()
        .find_map(|snap| tokenizer::safe_file(&snap, T3_FILENAME).ok())
}

fn hf_cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("huggingface").join("hub");
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("huggingface").join("hub")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warpers_match_the_hf_order() {
        let logits = [1.0f32, 2.0, 0.5, -1.0];
        let decode = TurboDecode {
            temperature: 0.8,
            top_k: 2,
            top_p: 0.95,
            repetition_penalty: 1.2,
            max_gen_len: 1,
        };
        let out = process_logits(&logits, &[0], &decode);
        assert!(out[2].is_infinite() && out[2].is_sign_negative());
        assert!(out[3].is_infinite() && out[3].is_sign_negative());
        assert!((out[0] - (1.25 / 1.2)).abs() < 1e-5, "{}", out[0]);
        assert!((out[1] - 2.5).abs() < 1e-5, "{}", out[1]);
        assert_eq!(argmax(&out), 1);
    }

    #[test]
    fn outside_weight_links_are_refused_and_cache_blobs_are_found() {
        let tmp = std::env::temp_dir().join(format!("pyre-t3-disc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let repo = tmp.join("models--ResembleAI--chatterbox-nano");
        let snap = repo.join("snapshots").join("abc");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), b"abc\n").unwrap();
        let blob = blobs.join("t3");
        std::fs::write(&blob, b"not a real checkpoint").unwrap();
        symlink_file(&blob, &snap.join(T3_FILENAME));
        let found = snapshot_weight(&repo).unwrap();
        assert_eq!(found, tokenizer::safe_file(&snap, T3_FILENAME).unwrap());
        let outside = tmp.join("outside.bin");
        std::fs::write(&outside, b"nope").unwrap();
        let sneaky = tmp.join("sneaky");
        std::fs::create_dir_all(&sneaky).unwrap();
        symlink_file(&outside, &sneaky.join(T3_FILENAME));
        assert!(tokenizer::safe_file(&sneaky, T3_FILENAME).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tiny_numpy_golden_matches_logits_and_greedy_ids() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model =
            T3Nano::load_shaped(&root.join("testdata/t3_tiny.safetensors"), 2, 14, 15).unwrap();
        let g: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/t3_tiny_golden.json")).unwrap();
        let speaker = nums(&g["speaker"]);
        let prompt = ids(&g["prompt_speech"]);
        let text = ids(&g["text"]);
        let ctx = T3Context {
            speaker,
            prompt_speech: prompt,
            text,
        };
        let got = model.speech_logits(&ctx, &[14]).unwrap();
        let expect = nums(&g["raw_logits"]);
        assert_close(&got, &expect, 2.0e-4);
        let decode = TurboDecode::default();
        let tokens = model
            .generate_greedy(&ctx, &decode, g["max_gen_len"].as_u64().unwrap() as usize)
            .unwrap();
        let expect_ids = ids(&g["greedy_tokens"]);
        assert_eq!(tokens, expect_ids);
        assert_eq!(model.n_head, 2);
    }

    #[test]
    #[ignore = "set PYRE_CHATTERBOX_T3 to t3_nano_v1.safetensors and pass --ignored"]
    fn real_weights_match_numpy_greedy_ids() {
        let Some(path) = std::env::var_os("PYRE_CHATTERBOX_T3") else {
            eprintln!("PYRE_CHATTERBOX_T3 unset");
            return;
        };
        let g: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/t3_real_golden.json")).unwrap();
        let model = T3Nano::load(Path::new(&path)).unwrap();
        let ctx = T3Context {
            speaker: nums(&g["speaker"]),
            prompt_speech: ids(&g["prompt_speech"]),
            text: ids(&g["text"]),
        };
        let got = model.speech_logits(&ctx, &[START_SPEECH]).unwrap();
        assert_close(&got, &nums(&g["raw_logits"]), 1.0e-4);
        let decode = TurboDecode::default();
        let tokens = model
            .generate_greedy(&ctx, &decode, g["max_gen_len"].as_u64().unwrap() as usize)
            .unwrap();
        assert_eq!(tokens, ids(&g["greedy_tokens"]));
    }

    fn nums(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn ids(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < tol, "max abs {max} tol {tol}");
    }

    fn symlink_file(src: &Path, dst: &Path) {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(src, dst).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(src, dst).unwrap();
    }
}
