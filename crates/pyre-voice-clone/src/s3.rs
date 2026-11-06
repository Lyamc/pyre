//! S3Tokenizer v2 reference encoder (25 Hz, finite scalar quantization).
//!
//! Nano's `prepare_conditionals` takes the first 15 s of 16 kHz audio and asks
//! for at most [`MAX_PROMPT_TOKENS`] (375) speech ids. Those ids are the prompt
//! half of T3's conditioning concat. They are not a waveform, and a
//! speaker-encoder cosine on a synthetic tone is not evidence that a clip is a
//! voice.
//!
//! Mel filters and the Hann window are the buffers stored on
//! `tokenizer._mel_filters` and `tokenizer.window` inside
//! `s3gen_meanflow.safetensors`. The encoder is six FSMN attention blocks
//! (state 1280, 20 heads) with RoPE, then a linear map to 8 channels whose
//! rounded values are a base-3 code in `0..6561`.

use crate::tokenizer::{self, REPO_NANO};
use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};

pub const S3_SR: u32 = 16_000;
pub const S3_HOP: usize = 160;
pub const S3_N_FFT: usize = 400;
pub const S3_N_MELS: usize = 128;
pub const S3_N_FREQ: usize = S3_N_FFT / 2 + 1;
pub const S3_TOKEN_RATE: usize = 25;
/// First 15 seconds at 25 Hz. `speech_cond_prompt_len` on T3-Nano.
pub const MAX_PROMPT_TOKENS: usize = 375;
pub const MEL_FRAMES_PER_TOKEN: usize = 4;
const N_STATE: usize = 1280;
const N_HEAD: usize = 20;
const HEAD: usize = N_STATE / N_HEAD;
const N_LAYER: usize = 6;
const FSQ_DIM: usize = 8;
const FSQ_SCALE: f32 = 0.9990000128746033;

pub const S3_FILENAME: &str = "s3gen_meanflow.safetensors";

struct Block {
    q_w: Tensor<2>,
    q_b: Tensor<1>,
    k_w: Tensor<2>,
    v_w: Tensor<2>,
    v_b: Tensor<1>,
    o_w: Tensor<2>,
    o_b: Tensor<1>,
    attn_ln_w: Tensor<1>,
    attn_ln_b: Tensor<1>,
    mlp_ln_w: Tensor<1>,
    mlp_ln_b: Tensor<1>,
    fc_w: Tensor<2>,
    fc_b: Tensor<1>,
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
    fsmn_w: Vec<f32>,
}

/// Speech tokenizer loaded from the Nano mean-flow checkpoint.
pub struct S3Tokenizer {
    device: Device,
    filters: Vec<f32>,
    window: Vec<f32>,
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    blocks: Vec<Block>,
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl S3Tokenizer {
    pub fn cpu() -> Device {
        Device::flex()
    }

    pub fn discover() -> Option<PathBuf> {
        discover_s3()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("safetensors")?;
        let device = Self::cpu();
        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let p = format!("tokenizer.encoder.blocks.{i}");
            blocks.push(Block {
                q_w: named2(&st, &format!("{p}.attn.query.weight"), &device)?,
                q_b: named1(&st, &format!("{p}.attn.query.bias"), &device)?,
                k_w: named2(&st, &format!("{p}.attn.key.weight"), &device)?,
                v_w: named2(&st, &format!("{p}.attn.value.weight"), &device)?,
                v_b: named1(&st, &format!("{p}.attn.value.bias"), &device)?,
                o_w: named2(&st, &format!("{p}.attn.out.weight"), &device)?,
                o_b: named1(&st, &format!("{p}.attn.out.bias"), &device)?,
                attn_ln_w: named1(&st, &format!("{p}.attn_ln.weight"), &device)?,
                attn_ln_b: named1(&st, &format!("{p}.attn_ln.bias"), &device)?,
                mlp_ln_w: named1(&st, &format!("{p}.mlp_ln.weight"), &device)?,
                mlp_ln_b: named1(&st, &format!("{p}.mlp_ln.bias"), &device)?,
                fc_w: named2(&st, &format!("{p}.mlp.0.weight"), &device)?,
                fc_b: named1(&st, &format!("{p}.mlp.0.bias"), &device)?,
                proj_w: named2(&st, &format!("{p}.mlp.2.weight"), &device)?,
                proj_b: named1(&st, &format!("{p}.mlp.2.bias"), &device)?,
                fsmn_w: named_vec(
                    &st,
                    &format!("{p}.attn.fsmn_block.weight"),
                    &[N_STATE, 1, 31],
                )?,
            });
        }
        let (rope_cos, rope_sin) = rope_tables();
        Ok(Self {
            device: device.clone(),
            filters: named_vec(&st, "tokenizer._mel_filters", &[S3_N_MELS, S3_N_FREQ])?,
            window: named_vec(&st, "tokenizer.window", &[S3_N_FFT])?,
            conv1_w: named_vec(
                &st,
                "tokenizer.encoder.conv1.weight",
                &[N_STATE, S3_N_MELS, 3],
            )?,
            conv1_b: named_vec(&st, "tokenizer.encoder.conv1.bias", &[N_STATE])?,
            conv2_w: named_vec(
                &st,
                "tokenizer.encoder.conv2.weight",
                &[N_STATE, N_STATE, 3],
            )?,
            conv2_b: named_vec(&st, "tokenizer.encoder.conv2.bias", &[N_STATE])?,
            blocks,
            proj_w: named2(
                &st,
                "tokenizer.quantizer._codebook.project_down.weight",
                &device,
            )?,
            proj_b: named1(
                &st,
                "tokenizer.quantizer._codebook.project_down.bias",
                &device,
            )?,
            rope_cos,
            rope_sin,
        })
    }

    /// Prompt ids for T3. Crops the mel to `max_tokens * 4` frames (15 s → 375).
    pub fn encode_prompt(&self, wav_16k: &[f32], max_tokens: usize) -> Result<Vec<u32>> {
        if wav_16k.is_empty() || max_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut mel = log_mel(wav_16k, &self.filters, &self.window);
        let frames = mel.len() / S3_N_MELS;
        let keep = frames.min(max_tokens.saturating_mul(MEL_FRAMES_PER_TOKEN));
        mel.truncate(S3_N_MELS * keep);
        if keep == 0 {
            return Ok(Vec::new());
        }
        self.quantize_mel(&mel, keep)
    }

    fn quantize_mel(&self, mel: &[f32], frames: usize) -> Result<Vec<u32>> {
        let h1 = conv_gelu(
            mel,
            S3_N_MELS,
            frames,
            &self.conv1_w,
            &self.conv1_b,
            N_STATE,
            2,
            1,
            &self.device,
        )?;
        let t1 = h1.len() / N_STATE;
        let h2 = conv_gelu(
            &h1,
            N_STATE,
            t1,
            &self.conv2_w,
            &self.conv2_b,
            N_STATE,
            2,
            1,
            &self.device,
        )?;
        let t = h2.len() / N_STATE;
        // conv output is [channels, time]; the transformer wants [time, channels].
        let mut hidden = from2(&transpose_ct(&h2, N_STATE, t), t, N_STATE, &self.device);
        for block in &self.blocks {
            hidden = block.forward(hidden, &self.rope_cos, &self.rope_sin);
        }
        let z = hidden
            .matmul(self.proj_w.clone().transpose())
            .add(self.proj_b.clone().unsqueeze());
        Ok(fsq_ids(&to_vec2(z)?, t))
    }
}

impl Block {
    fn forward(&self, x: Tensor<2>, cos: &[f32], sin: &[f32]) -> Tensor<2> {
        let t = x.dims()[0];
        let residual = x.clone();
        let n = layer_norm(x, &self.attn_ln_w, &self.attn_ln_b);
        let q = n
            .clone()
            .matmul(self.q_w.clone().transpose())
            .add(self.q_b.clone().unsqueeze());
        let k = n.clone().matmul(self.k_w.clone().transpose());
        let v = n
            .matmul(self.v_w.clone().transpose())
            .add(self.v_b.clone().unsqueeze());
        let mut qv = to_vec2(q.clone()).expect("q");
        let mut kv = to_vec2(k.clone()).expect("k");
        let vv = to_vec2(v).expect("v");
        apply_rope(&mut qv, t, cos, sin);
        apply_rope(&mut kv, t, cos, sin);
        let scale = (HEAD as f32).powf(-0.25);
        for z in qv.iter_mut().chain(kv.iter_mut()) {
            *z *= scale;
        }
        let device = q.device();
        let qh = split_heads(from2(&qv, t, N_STATE, &device));
        let kh = split_heads(from2(&kv, t, N_STATE, &device)).swap_dims(1, 2);
        let vh = split_heads(from2(&vv, t, N_STATE, &device));
        let ctx = merge_heads(softmax(qh.matmul(kh), 2).matmul(vh));
        let attn = ctx
            .matmul(self.o_w.clone().transpose())
            .add(self.o_b.clone().unsqueeze());
        let fsm = from2(&fsmn(&vv, t, &self.fsmn_w), t, N_STATE, &device);
        let x = residual.add(attn).add(fsm);
        let residual = x.clone();
        let mid = gelu(
            layer_norm(x, &self.mlp_ln_w, &self.mlp_ln_b)
                .matmul(self.fc_w.clone().transpose())
                .add(self.fc_b.clone().unsqueeze()),
        );
        residual.add(
            mid.matmul(self.proj_w.clone().transpose())
                .add(self.proj_b.clone().unsqueeze()),
        )
    }
}

/// Whisper-style log mel used by S3: center reflect STFT, drop the last frame,
/// power, the checkpoint filterbank, then `clamp / log10 / (max-8) / (+4)/4`.
pub fn log_mel(wav: &[f32], filters: &[f32], window: &[f32]) -> Vec<f32> {
    if wav.is_empty() || window.len() != S3_N_FFT || filters.len() != S3_N_MELS * S3_N_FREQ {
        return Vec::new();
    }
    let frames = wav.len() / S3_HOP;
    if frames == 0 {
        return Vec::new();
    }
    let mut power = vec![0f64; S3_N_FREQ * frames];
    for t in 0..frames {
        let start = t as isize * S3_HOP as isize;
        let mut re = [0f64; S3_N_FREQ];
        let mut im = [0f64; S3_N_FREQ];
        for n in 0..S3_N_FFT {
            let s = reflect(wav, start + n as isize - (S3_N_FFT / 2) as isize) as f64
                * window[n] as f64;
            for k in 0..S3_N_FREQ {
                let ang = std::f64::consts::TAU * (k * n) as f64 / S3_N_FFT as f64;
                re[k] += s * ang.cos();
                im[k] -= s * ang.sin();
            }
        }
        for k in 0..S3_N_FREQ {
            power[k * frames + t] = re[k] * re[k] + im[k] * im[k];
        }
    }
    let mut mel = vec![0f32; S3_N_MELS * frames];
    for m in 0..S3_N_MELS {
        for t in 0..frames {
            let mut acc = 0f64;
            for k in 0..S3_N_FREQ {
                acc += filters[m * S3_N_FREQ + k] as f64 * power[k * frames + t];
            }
            mel[m * frames + t] = acc as f32;
        }
    }
    let mut peak = f32::NEG_INFINITY;
    for v in &mut mel {
        *v = v.max(1e-10).log10();
        peak = peak.max(*v);
    }
    let floor = peak - 8.0;
    for v in &mut mel {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
    mel
}

fn reflect(wav: &[f32], index: isize) -> f32 {
    let n = wav.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return wav[0];
    }
    let period = ((n - 1) * 2) as isize;
    let mut i = index.rem_euclid(period);
    if i >= n as isize {
        i = period - i;
    }
    wav[i as usize]
}

fn conv_gelu(
    x: &[f32],
    cin: usize,
    t: usize,
    w: &[f32],
    bias: &[f32],
    cout: usize,
    stride: usize,
    pad: usize,
    device: &Device,
) -> Result<Vec<f32>> {
    let k = 3usize;
    let tout = (t + 2 * pad - k) / stride + 1;
    let col_h = cin * k;
    let mut col = vec![0f32; col_h * tout];
    for ot in 0..tout {
        for ic in 0..cin {
            for kk in 0..k {
                let src = ot as isize * stride as isize + kk as isize - pad as isize;
                if src >= 0 && (src as usize) < t {
                    col[(ic * k + kk) * tout + ot] = x[ic * t + src as usize];
                }
            }
        }
    }
    // Weight is PyTorch `[out, in, k]`. The product is `[out, time]`, so the
    // bias is a column, not a row.
    let y = from2(w, cout, col_h, device)
        .matmul(from2(&col, col_h, tout, device))
        .add(from1(bias, device).unsqueeze_dim::<2>(1));
    Ok(to_vec2(gelu(y))?)
}

fn transpose_ct(v: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut o = vec![0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            o[ti * c + ci] = v[ci * t + ti];
        }
    }
    o
}

fn fsmn(v: &[f32], t: usize, w: &[f32]) -> Vec<f32> {
    let k = 31usize;
    let pad = 15isize;
    let mut y = vec![0f32; t * N_STATE];
    for ci in 0..N_STATE {
        for ti in 0..t {
            let mut acc = 0f32;
            for kk in 0..k {
                let src = ti as isize + kk as isize - pad;
                if src >= 0 && (src as usize) < t {
                    acc += v[src as usize * N_STATE + ci] * w[ci * k + kk];
                }
            }
            y[ti * N_STATE + ci] = acc + v[ti * N_STATE + ci];
        }
    }
    y
}

fn apply_rope(flat: &mut [f32], t: usize, cos: &[f32], sin: &[f32]) {
    for ti in 0..t {
        for h in 0..N_HEAD {
            let base = (ti * N_HEAD + h) * HEAD;
            let mut rot = [0f32; HEAD];
            for i in 0..HEAD {
                let x = flat[base + i];
                let xr = if i < HEAD / 2 {
                    -flat[base + HEAD / 2 + i]
                } else {
                    flat[base + (i - HEAD / 2)]
                };
                let c = cos[ti * HEAD + i];
                let s = sin[ti * HEAD + i];
                rot[i] = x * c + xr * s;
            }
            flat[base..base + HEAD].copy_from_slice(&rot);
        }
    }
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let end = 2048usize;
    let theta = 10_000.0f64;
    let mut cos = vec![0f32; end * HEAD];
    let mut sin = vec![0f32; end * HEAD];
    for t in 0..end {
        for i in 0..(HEAD / 2) {
            let freq = 1.0 / theta.powf((2 * i) as f64 / HEAD as f64);
            let ang = t as f64 * freq;
            let (s, c) = ang.sin_cos();
            // Frequencies are duplicated across the two halves, matching `cat(cis, cis)`.
            cos[t * HEAD + i] = c as f32;
            cos[t * HEAD + HEAD / 2 + i] = c as f32;
            sin[t * HEAD + i] = s as f32;
            sin[t * HEAD + HEAD / 2 + i] = s as f32;
        }
    }
    (cos, sin)
}

/// Base-3 codes. A zero projection is the middle bin on every axis (index 3280).
pub fn fsq_ids(z: &[f32], t: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(t);
    for ti in 0..t {
        let mut idx = 0i64;
        let mut pow = 1i64;
        for d in 0..FSQ_DIM {
            let r = (z[ti * FSQ_DIM + d].tanh() * FSQ_SCALE).round_ties_even();
            let q = (r + 1.0).round() as i64;
            let q = q.clamp(0, 2);
            idx += q * pow;
            pow *= 3;
        }
        ids.push(idx as u32);
    }
    ids
}

fn layer_norm(x: Tensor<2>, w: &Tensor<1>, b: &Tensor<1>) -> Tensor<2> {
    let mean = x.clone().mean_dim(1);
    let centered = x.sub(mean);
    let var = centered.clone().powi_scalar(2).mean_dim(1);
    centered
        .div(var.add_scalar(1.0e-5).sqrt())
        .mul(w.clone().unsqueeze())
        .add(b.clone().unsqueeze())
}

fn split_heads(x: Tensor<2>) -> Tensor<3> {
    let t = x.dims()[0];
    x.reshape([t, N_HEAD, HEAD]).swap_dims(0, 1)
}

fn merge_heads(x: Tensor<3>) -> Tensor<2> {
    let [h, t, hd] = x.dims();
    x.swap_dims(0, 1).reshape([t, h * hd])
}

fn named_vec(st: &safetensors::SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let got = v.shape().to_vec();
    if got != shape {
        bail!("{name}: shape {got:?} != {shape:?}");
    }
    f32s(v)
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
    if v.shape().len() != 1 {
        bail!("{name}: rank {}", v.shape().len());
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

pub fn discover_s3() -> Option<PathBuf> {
    for key in ["PYRE_CHATTERBOX_S3", "PYRE_CHATTERBOX_NANO"] {
        if let Some(p) = std::env::var_os(key) {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
            if let Ok(found) = tokenizer::safe_file(&p, S3_FILENAME) {
                return Some(found);
            }
        }
    }
    let cache = hf_cache_dir();
    let root = cache.join(format!("models--{}", REPO_NANO.replace('/', "--")));
    snapshot_weight(&root)
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
        .find_map(|snap| tokenizer::safe_file(&snap, S3_FILENAME).ok())
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
    fn fifteen_seconds_of_mel_is_375_prompt_tokens() {
        let frames = 15 * S3_SR as usize / S3_HOP;
        assert_eq!(frames, MAX_PROMPT_TOKENS * MEL_FRAMES_PER_TOKEN);
        assert_eq!(MAX_PROMPT_TOKENS, 15 * S3_TOKEN_RATE);
    }

    #[test]
    fn one_second_log_mel_keeps_one_hundred_frames() {
        let mel = log_mel(
            &vec![0f32; S3_SR as usize],
            &vec![0f32; S3_N_MELS * S3_N_FREQ],
            &vec![0f32; S3_N_FFT],
        );
        assert_eq!(mel.len(), S3_N_MELS * 100);
        assert!(mel.iter().all(|v| (*v + 1.5).abs() < 1e-5));
    }

    #[test]
    fn zero_projection_is_the_middle_code() {
        let ids = fsq_ids(&[0f32; FSQ_DIM], 1);
        let expect = (1 + 3 + 9 + 27 + 81 + 243 + 729 + 2187) as u32;
        assert_eq!(ids, vec![expect]);
        assert!(ids[0] < 6561);
    }

    /// Eight harmonics of 140 Hz, compared with `tools/s3_reference.py` (PyTorch).
    /// Matching ids proves the encoder. It does not prove the clip is a voice:
    /// [`crate::verify::CLONE_COSINE_FLOOR`] is a speaker-encoder threshold and
    /// is not consulted here.
    #[test]
    #[ignore = "set PYRE_CHATTERBOX_S3 to s3gen_meanflow.safetensors and pass --ignored"]
    fn real_formant_tokens_match_torch_reference() {
        let Some(path) = std::env::var_os("PYRE_CHATTERBOX_S3") else {
            eprintln!("PYRE_CHATTERBOX_S3 unset");
            return;
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(root.join("testdata/s3_formant.f32")).unwrap();
        let wav: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let g: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/s3_formant_golden.json")).unwrap();
        assert_eq!(g["note"].as_str().unwrap().contains("Not a voice"), true);
        let model = S3Tokenizer::load(Path::new(&path)).unwrap();
        let ids = model
            .encode_prompt(&wav, g["max_tokens"].as_u64().unwrap() as usize)
            .unwrap();
        let expect = g["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(ids, expect, "S3 ids diverged from the PyTorch reference");
        assert!(ids.iter().all(|id| *id < 6561));
        assert!(
            ids.iter().any(|id| *id != ids[0]),
            "a flat code is not a useful reference"
        );
    }

    /// Feed the reference S3 ids into T3 as prompt-speech embeddings.
    /// Requires both checkpoints. A harmonic cosine is not part of this check.
    #[test]
    #[ignore = "set PYRE_CHATTERBOX_S3 and PYRE_CHATTERBOX_T3 and pass --ignored"]
    fn real_s3_ids_change_t3_prompt_logits() {
        let Some(s3_path) = std::env::var_os("PYRE_CHATTERBOX_S3") else {
            eprintln!("PYRE_CHATTERBOX_S3 unset");
            return;
        };
        let Some(t3_path) = std::env::var_os("PYRE_CHATTERBOX_T3") else {
            eprintln!("PYRE_CHATTERBOX_T3 unset");
            return;
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(root.join("testdata/s3_formant.f32")).unwrap();
        let wav: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let s3 = S3Tokenizer::load(Path::new(&s3_path)).unwrap();
        let prompt = s3.encode_prompt(&wav, MAX_PROMPT_TOKENS).unwrap();
        assert!(!prompt.is_empty());
        let t3 = crate::t3_model::T3Nano::load(Path::new(&t3_path)).unwrap();
        let speaker = vec![0.01f32; 256];
        let text = vec![15496u32, 11, 995, 13];
        let with = t3
            .speech_logits(
                &crate::t3_model::T3Context {
                    speaker: speaker.clone(),
                    prompt_speech: prompt,
                    text: text.clone(),
                },
                &[crate::t3::START_SPEECH],
            )
            .unwrap();
        let without = t3
            .speech_logits(
                &crate::t3_model::T3Context {
                    speaker,
                    prompt_speech: Vec::new(),
                    text,
                },
                &[crate::t3::START_SPEECH],
            )
            .unwrap();
        assert!(with.iter().all(|v| v.is_finite()) && without.iter().all(|v| v.is_finite()));
        let max = with
            .iter()
            .zip(&without)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max > 1.0,
            "prompt embeddings did not move T3 logits (max abs {max})"
        );
    }

    /// A pure sine is not a voice. This only checks the crop math and that the
    /// speaker-encoder cosine floor is not the acceptance test for these ids.
    #[test]
    fn synthetic_tone_is_not_a_voice_proof() {
        assert!(crate::verify::CLONE_COSINE_FLOOR > 0.0);
        let mut wav = vec![0f32; S3_SR as usize];
        for (i, s) in wav.iter_mut().enumerate() {
            *s = (2.0 * std::f64::consts::PI * 140.0 * i as f64 / S3_SR as f64).sin() as f32 * 0.1;
        }
        let frames = wav.len() / S3_HOP;
        assert_eq!(frames, 100);
        let kept = frames.min(MAX_PROMPT_TOKENS * MEL_FRAMES_PER_TOKEN);
        assert_eq!(kept, 100);
    }
}
