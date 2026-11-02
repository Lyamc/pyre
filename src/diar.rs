//! Speaker diarization: a hand-written Burn port of nvidia/Nemotron-3-Diarization (Streaming
//! Sortformer; HF `Nemotron3DiarizationForAudioFrameClassification`). No Python, no NeMo.
//!
//! Front end (host): pre-emphasis 0.97, centred STFT (n_fft 512, Hann 400, hop 160), power,
//! 128 Slaney mel bins, ln(x + 2^-24), no normalisation. Model (Burn): stack 8 mel frames ->
//! linear 1024->512 (80 ms frames) -> LayerNorm -> 31 pre-LN Transformer layers (8 heads, RoPE,
//! GELU MLP 2048) -> LayerNorm -> linear 512->192 -> sub-pixel Conv1D upsampler (x8, back to
//! 10 ms) -> ReLU/linear/ReLU/linear -> 8 speaker logits per 10 ms frame, speakers ordered by
//! first arrival. Long audio is chunked exactly like the reference "offline" configuration
//! (340-frame chunks, 40 frames look-ahead) with the Arrival-Order Speaker Cache (264 frames) and
//! a FIFO (40 frames) carried between chunks, so any length works.
use crate::model::loader::Weights;
use anyhow::{bail, Context, Result};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::module::{conv1d, linear};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The loaded diarizer (at most one; `--diar-model`).
pub static DIAR: OnceLock<Arc<Diarizer>> = OnceLock::new();

const SR: usize = 16_000;
const HOP: usize = 160;
const NFFT: usize = 512;
const WIN: usize = 400;
const NMEL: usize = 128;
const SUB: usize = 8;
const HID: usize = 512;
const HEADS: usize = 8;
const HD: usize = 64;
const NSPK: usize = 8;
// offline chunking (config.json top level) and the speaker-cache policy (streaming_config)
const CHUNK: usize = 340;
const RIGHT: usize = 40;
const FIFO: usize = 40;
const UPDATE: usize = 300;
const CACHE: usize = 264;
const SIL_FRAMES: usize = 1;
const SCORE_THR: f32 = 0.25;
const LATEST_BOOST: f32 = 0.05;

fn opts(d: &Device) -> TensorCreationOptions { TensorCreationOptions::new(d.clone()).with_dtype(DType::F32) }
fn host<const D: usize>(t: Tensor<D>) -> Vec<f32> { t.into_data().convert::<f32>().try_to_vec::<f32>().expect("f32 tensor") }

struct Lin { w: Tensor<2>, b: Option<Tensor<1>> }
impl Lin { fn f(&self, x: Tensor<2>) -> Tensor<2> { linear(x, self.w.clone(), self.b.clone()) } }
struct Ln { w: Tensor<1>, b: Tensor<1> }
impl Ln {
    fn f(&self, x: Tensor<2>) -> Tensor<2> {
        let m = x.clone().mean_dim(1);
        let xc = x.sub(m);
        let v = xc.clone().powi_scalar(2).mean_dim(1);
        xc.div(v.add_scalar(1e-5).sqrt()).mul(self.w.clone().unsqueeze::<2>()).add(self.b.clone().unsqueeze::<2>())
    }
}
struct Layer { ln1: Ln, q: Lin, k: Lin, v: Lin, o: Lin, ln2: Ln, fc1: Lin, fc2: Lin }

pub struct Net {
    device: Device,
    embed: Lin,
    in_ln: Ln,
    layers: Vec<Layer>,
    out_ln: Ln,
    proj: Lin,
    up_w: Tensor<3>,
    up_b: Tensor<1>,
    dense: Lin,
    out: Lin,
    silence: Vec<f32>,
    mel: Vec<f32>, // [NMEL, NFFT/2+1]
    win: Vec<f32>, // [WIN]
}

/// librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney"), Slaney scale.
fn mel_filters() -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let logstep = 6.4f64.ln() / 27.0;
    let hz2mel = |f: f64| if f >= 1000.0 { 15.0 + (f / 1000.0).ln() / logstep } else { f / f_sp };
    let mel2hz = |m: f64| if m >= 15.0 { 1000.0 * (logstep * (m - 15.0)).exp() } else { f_sp * m };
    let nb = NFFT / 2 + 1;
    let fft: Vec<f64> = (0..nb).map(|k| k as f64 * (SR as f64 / 2.0) / (nb - 1) as f64).collect();
    let (m0, m1) = (hz2mel(0.0), hz2mel(SR as f64 / 2.0));
    let mf: Vec<f64> = (0..NMEL + 2).map(|i| mel2hz(m0 + (m1 - m0) * i as f64 / (NMEL + 1) as f64)).collect();
    let mut w = vec![0f32; NMEL * nb];
    for i in 0..NMEL {
        let (d0, d1) = (mf[i + 1] - mf[i], mf[i + 2] - mf[i + 1]);
        let en = 2.0 / (mf[i + 2] - mf[i]);
        for k in 0..nb {
            let lower = -(mf[i] - fft[k]) / d0;
            let upper = (mf[i + 2] - fft[k]) / d1;
            w[i * nb + k] = (lower.min(upper).max(0.0) * en) as f32;
        }
    }
    w
}

impl Net {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
        let a = &cfg["audio_config"];
        if a["hidden_size"] != json!(HID) || a["num_hidden_layers"].as_u64().is_none() || a["num_mel_bins"] != json!(NMEL) || cfg["head_config"]["num_speakers"] != json!(NSPK) {
            bail!("unexpected Nemotron-3-Diarization config in {}", dir.display());
        }
        let nl = a["num_hidden_layers"].as_u64().unwrap() as usize;
        let ws = Weights::open(dir, device.clone(), DType::F32)?;
        let lin = |p: &str, bias: bool| -> Result<Lin> {
            Ok(Lin { w: ws.get::<2>(&format!("{p}.weight"))?.transpose(), b: if bias { Some(ws.get::<1>(&format!("{p}.bias"))?) } else { None } })
        };
        let ln = |p: &str| -> Result<Ln> { Ok(Ln { w: ws.get::<1>(&format!("{p}.weight"))?, b: ws.get::<1>(&format!("{p}.bias"))? }) };
        let t = "model.audio_tower";
        let mut layers = vec![];
        for i in 0..nl {
            let p = format!("{t}.layers.{i}");
            layers.push(Layer {
                ln1: ln(&format!("{p}.layer_norm1"))?,
                q: lin(&format!("{p}.self_attn.q_proj"), false)?, k: lin(&format!("{p}.self_attn.k_proj"), false)?,
                v: lin(&format!("{p}.self_attn.v_proj"), false)?, o: lin(&format!("{p}.self_attn.o_proj"), true)?,
                ln2: ln(&format!("{p}.layer_norm2"))?,
                fc1: lin(&format!("{p}.mlp.fc1"), true)?, fc2: lin(&format!("{p}.mlp.fc2"), true)?,
            });
        }
        let silence = host(ws.get::<1>("silence_embeds")?);
        let win: Vec<f32> = (0..WIN).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos()) as f32).collect();
        Ok(Self {
            embed: lin(&format!("{t}.embedder.projection"), false)?,
            in_ln: ln(&format!("{t}.input_layer_norm"))?,
            layers,
            out_ln: ln(&format!("{t}.layer_norm"))?,
            proj: lin("model.proj", true)?,
            up_w: ws.get::<3>("model.upsampler.conv.weight")?,
            up_b: ws.get::<1>("model.upsampler.conv.bias")?,
            dense: lin("classifier.dense", true)?,
            out: lin("classifier.out_proj", true)?,
            silence, mel: mel_filters(), win, device,
        })
    }

    /// Log-mel features `[T, 128]` (row-major) and the number of valid frames (`floor(L/hop)`).
    pub fn features(&self, x: &[f32]) -> (Vec<f32>, usize, usize) {
        let l = x.len();
        let mut y = vec![0f32; l];
        for i in 0..l { y[i] = if i == 0 { x[0] } else { x[i] - 0.97 * x[i - 1] }; }
        let t = 1 + l / HOP;
        let valid = l / HOP;
        let nb = NFFT / 2 + 1;
        // DFT tables over the 400 non-zero window taps (the window sits at offset 56 inside 512)
        let off = (NFFT - WIN) / 2;
        let tw: Vec<(f32, f32)> = (0..nb * WIN).map(|i| {
            let (k, n) = (i / WIN, i % WIN);
            let a = -2.0 * std::f64::consts::PI * (k * (n + off) % NFFT) as f64 / NFFT as f64;
            (a.cos() as f32, a.sin() as f32)
        }).collect();
        let mut feat = vec![0f32; t * NMEL];
        feat.par_chunks_mut(NMEL).enumerate().for_each(|(f, row)| {
            if f >= valid { return; }
            let mut seg = [0f32; WIN];
            for n in 0..WIN {
                let s = (f * HOP + off + n) as isize - (NFFT / 2) as isize;
                seg[n] = if s >= 0 && (s as usize) < l { y[s as usize] * self.win[n] } else { 0.0 };
            }
            let mut pw = vec![0f64; nb];
            for k in 0..nb {
                let (mut re, mut im) = (0f64, 0f64);
                let tk = &tw[k * WIN..(k + 1) * WIN];
                for n in 0..WIN { re += (seg[n] * tk[n].0) as f64; im += (seg[n] * tk[n].1) as f64; }
                pw[k] = re * re + im * im;
            }
            for m in 0..NMEL {
                let wm = &self.mel[m * nb..(m + 1) * nb];
                let mut s = 0f64;
                for k in 0..nb { s += wm[k] as f64 * pw[k]; }
                row[m] = (s as f32 + 2f32.powi(-24)).ln();
            }
        });
        (feat, t, valid)
    }

    /// Feature stacking + projection: `[T,128]` -> `[ceil(T/8), 512]` (host rows).
    fn embed(&self, feat: &[f32], t: usize) -> (Vec<f32>, usize) {
        let n = t.div_ceil(SUB);
        let mut st = feat.to_vec();
        st.resize(n * SUB * NMEL, 0.0);
        let x = Tensor::<2>::from_data(TensorData::new(st, [n, SUB * NMEL]), opts(&self.device));
        (host(self.embed.f(x)), n)
    }

    /// Encoder + head on one step input `[L,512]` (positions restart at 0) -> logits `[L*8, 8]`.
    fn step(&self, input: Vec<f32>, mask: &[bool]) -> Vec<f32> {
        let l = mask.len();
        let d = &self.device;
        let mut x = self.in_ln.f(Tensor::<2>::from_data(TensorData::new(input, [l, HID]), opts(d)));
        // RoPE (theta 1e4, full head dim, rotate_half layout)
        let mut cs = vec![0f32; l * HD];
        let mut sn = vec![0f32; l * HD];
        for p in 0..l {
            for i in 0..HD / 2 {
                let f = p as f64 / 10000f64.powf(2.0 * i as f64 / HD as f64);
                let (c, s) = (f.cos() as f32, f.sin() as f32);
                cs[p * HD + i] = c; cs[p * HD + i + HD / 2] = c;
                sn[p * HD + i] = s; sn[p * HD + i + HD / 2] = s;
            }
        }
        let cos = Tensor::<2>::from_data(TensorData::new(cs, [l, HD]), opts(d)).unsqueeze::<3>();
        let sin = Tensor::<2>::from_data(TensorData::new(sn, [l, HD]), opts(d)).unsqueeze::<3>();
        let am: Vec<f32> = mask.iter().map(|&m| if m { 0.0 } else { f32::MIN }).collect();
        let am = Tensor::<1>::from_data(TensorData::new(am, [l]), opts(d)).reshape([1, 1, l]);
        let rope = |t: Tensor<3>| -> Tensor<3> {
            let a = t.clone().narrow(2, 0, HD / 2);
            let b = t.clone().narrow(2, HD / 2, HD / 2);
            let rot = Tensor::cat(vec![b.neg(), a], 2);
            t.mul(cos.clone()).add(rot.mul(sin.clone()))
        };
        for ly in &self.layers {
            let h = ly.ln1.f(x.clone());
            let q = rope(ly.q.f(h.clone()).reshape([l, HEADS, HD]).swap_dims(0, 1));
            let k = rope(ly.k.f(h.clone()).reshape([l, HEADS, HD]).swap_dims(0, 1));
            let v = ly.v.f(h).reshape([l, HEADS, HD]).swap_dims(0, 1);
            let s = q.matmul(k.swap_dims(1, 2)).mul_scalar(1.0 / (HD as f64).sqrt()).add(am.clone());
            let o = softmax(s, 2).matmul(v).swap_dims(0, 1).reshape([l, HID]);
            x = x.add(ly.o.f(o));
            let h = ly.ln2.f(x.clone());
            x = x.add(ly.fc2.f(gelu(ly.fc1.f(h))));
        }
        let z = self.proj.f(self.out_ln.f(x)); // [L,192]
        let c = 192;
        let up = conv1d(z.transpose().unsqueeze::<3>(), self.up_w.clone(), Some(self.up_b.clone()), ConvOptions::new([1], [1], [1], 1)); // [1, 1536, L]
        let up = up.squeeze_dim::<2>(0).transpose().reshape([l * SUB, c]);
        let lg = self.out.f(relu(self.dense.f(relu(up))));
        host(lg)
    }

    /// Whole-recording logits `[T, 8]` at 10 ms (offline chunked mode with speaker cache + FIFO).
    pub fn logits(&self, samples: &[f32]) -> (Vec<f32>, usize, usize) {
        let (feat, t, valid) = self.features(samples);
        let (emb, n) = self.embed(&feat, t);
        let emask: Vec<bool> = (0..n).map(|j| j * SUB < valid).collect();
        let mut st = Cache::default();
        let mut out = Vec::with_capacity(n * SUB * NSPK);
        let mut s = 0;
        while s < n {
            let end = (s + CHUNK).min(n);
            let nchunk = end - s;
            let cend = (end + RIGHT).min(n);
            let cached = st.ncache + st.nfifo;
            let mut input = Vec::with_capacity((cached + cend - s) * HID);
            input.extend_from_slice(&st.embeds[..st.ncache * HID]);
            input.extend_from_slice(&st.fifo[..st.nfifo * HID]);
            input.extend_from_slice(&emb[s * HID..cend * HID]);
            let mut mask = vec![true; cached];
            mask.extend_from_slice(&emask[s..cend]);
            let lg = self.step(input.clone(), &mask);
            st.update(&input, &lg, &self.silence, nchunk, &mask);
            out.extend_from_slice(&lg[cached * SUB * NSPK..(cached + nchunk) * SUB * NSPK]);
            s = end;
        }
        out.truncate(t * NSPK);
        (out, t, valid)
    }
}

#[derive(Default)]
struct Cache { embeds: Vec<f32>, probs: Vec<f32>, fifo: Vec<f32>, ncache: usize, nfifo: usize, compressed: bool }

fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

impl Cache {
    fn update(&mut self, input: &[f32], logits: &[f32], silence: &[f32], nchunk: usize, mask: &[bool]) {
        let lin = mask.len();
        // speaker probabilities pooled to the encoder rate, zeroed on padding
        let mut probs = vec![0f32; lin * NSPK];
        for i in 0..lin {
            if !mask[i] { continue; }
            for s in 0..NSPK {
                let mut a = 0f32;
                for k in 0..SUB { a += sig(logits[(i * SUB + k) * NSPK + s]); }
                probs[i * NSPK + s] = a / SUB as f32;
            }
        }
        let (nc, nf) = (self.ncache, self.nfifo);
        let cs = nc + nf;
        let mut fifo: Vec<f32> = self.fifo[..nf * HID].to_vec();
        fifo.extend_from_slice(&input[cs * HID..(cs + nchunk) * HID]);
        let flen = nf + nchunk;
        let npop = if flen <= FIFO { 0 } else { UPDATE.max(flen - FIFO).min(flen) };
        if npop > 0 {
            let mut fprobs: Vec<f32> = probs[nc * NSPK..(nc + nf) * NSPK].to_vec();
            fprobs.extend_from_slice(&probs[cs * NSPK..(cs + nchunk) * NSPK]);
            let stored: Vec<f32> = if self.compressed { self.probs[..nc * NSPK].to_vec() } else { probs[..nc * NSPK].to_vec() };
            let mut ce: Vec<f32> = self.embeds[..nc * HID].to_vec();
            ce.extend_from_slice(&fifo[..npop * HID]);
            let mut cp = stored;
            cp.extend_from_slice(&fprobs[..npop * NSPK]);
            fifo.drain(..npop * HID);
            if ce.len() / HID > CACHE {
                let (e, p) = compress(&ce, &cp, silence);
                ce = e; cp = p;
                self.compressed = true;
            }
            self.ncache = ce.len() / HID;
            self.embeds = ce;
            self.probs = cp;
        }
        self.nfifo = fifo.len() / HID;
        self.fifo = fifo;
    }
}

/// Indices of the `k` largest values (ties: lower index first), like an unsorted torch.topk.
fn topk(v: &[f32], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    ix.truncate(k);
    ix
}

/// Arrival-Order Speaker Cache compression: keep the `CACHE` most informative frames, grouped by
/// speaker in their original order, plus `SIL_FRAMES` silence slots per speaker.
fn compress(embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let f = embeds.len() / HID;
    let budget = CACHE / NSPK - SIL_FRAMES;
    let min_pos = (budget as f64 * 0.5).floor() as usize;
    let strong = (budget as f64 * 0.75).floor() as usize;
    let weak = (budget as f64 * 1.5).floor() as usize;
    let ln05 = 0.5f32.ln();
    // frame scores [F, S]
    let mut sc = vec![0f32; f * NSPK];
    for i in 0..f {
        let lc: Vec<f32> = (0..NSPK).map(|s| (1.0 - probs[i * NSPK + s]).max(SCORE_THR).ln()).collect();
        let sum: f32 = lc.iter().sum();
        for s in 0..NSPK {
            let p = probs[i * NSPK + s];
            sc[i * NSPK + s] = if p > 0.5 { p.max(SCORE_THR).ln() - lc[s] + sum - ln05 } else { f32::NEG_INFINITY };
        }
    }
    for s in 0..NSPK {
        let npos = (0..f).filter(|&i| sc[i * NSPK + s] > 0.0).count();
        if npos >= min_pos {
            for i in 0..f { let p = probs[i * NSPK + s]; if p > 0.5 && !(sc[i * NSPK + s] > 0.0) { sc[i * NSPK + s] = f32::NEG_INFINITY; } }
        }
    }
    for i in CACHE..f { for s in 0..NSPK { sc[i * NSPK + s] += LATEST_BOOST; } }
    for (k, b) in [(strong, -2.0 * ln05), (weak, -ln05)] {
        for s in 0..NSPK {
            let col: Vec<f32> = (0..f).map(|i| sc[i * NSPK + s]).collect();
            for i in topk(&col, k) { sc[i * NSPK + s] += b; }
        }
    }
    // pad SIL_FRAMES rows of +inf, flatten speaker-major
    let fs = f + SIL_FRAMES;
    let mut flat = vec![0f32; NSPK * fs];
    for s in 0..NSPK { for i in 0..fs { flat[s * fs + i] = if i < f { sc[i * NSPK + s] } else { f32::INFINITY }; } }
    let sentinel = fs * NSPK;
    let mut sel: Vec<usize> = topk(&flat, CACHE).into_iter().map(|j| if flat[j] == f32::NEG_INFINITY { sentinel } else { j }).collect();
    sel.sort();
    let mut e = Vec::with_capacity(CACHE * HID);
    let mut p = Vec::with_capacity(CACHE * NSPK);
    for j in sel {
        let fi = if j == sentinel { f } else { (j % fs).min(f) };
        if fi >= f { e.extend_from_slice(silence); p.extend(std::iter::repeat_n(0f32, NSPK)); }
        else { e.extend_from_slice(&embeds[fi * HID..(fi + 1) * HID]); p.extend_from_slice(&probs[fi * NSPK..(fi + 1) * NSPK]); }
    }
    (e, p)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Seg { pub speaker: usize, pub start: f64, pub end: f64 }

/// Thresholded per-speaker activity -> segments (10 ms frames), sorted by (start, speaker).
pub fn segments(logits: &[f32], valid: usize, thr: f32) -> Vec<Seg> {
    let t = logits.len() / NSPK;
    let mut out = vec![];
    for s in 0..NSPK {
        let mut start: Option<usize> = None;
        for i in 0..=t {
            let on = i < t && i < valid && sig(logits[i * NSPK + s]) > thr;
            match (on, start) {
                (true, None) => start = Some(i),
                (false, Some(b)) => { out.push(Seg { speaker: s, start: (b as f64 * 0.01 * 100.0).round() / 100.0, end: (i as f64 * 0.01 * 100.0).round() / 100.0 }); start = None; }
                _ => {}
            }
        }
    }
    out.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap().then(a.speaker.cmp(&b.speaker)));
    out
}

/// Merges a speaker's segments separated by less than `gap` s and drops ones shorter than `min` s.
pub fn tidy(segs: &[Seg], gap: f64, min: f64) -> Vec<Seg> {
    let mut by: Vec<Vec<Seg>> = vec![vec![]; NSPK];
    for s in segs { by[s.speaker].push(s.clone()); }
    let mut out = vec![];
    for v in by {
        let mut cur: Option<Seg> = None;
        for s in v {
            cur = match cur {
                Some(mut c) if s.start - c.end < gap => { c.end = s.end; Some(c) }
                Some(c) => { if c.end - c.start >= min { out.push(c); } Some(s) }
                None => Some(s),
            };
        }
        if let Some(c) = cur { if c.end - c.start >= min { out.push(c); } }
    }
    out.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap().then(a.speaker.cmp(&b.speaker)));
    // renumber speakers densely in arrival order
    let mut map = vec![usize::MAX; NSPK];
    let mut next = 0;
    for s in out.iter_mut() { if map[s.speaker] == usize::MAX { map[s.speaker] = next; next += 1; } s.speaker = map[s.speaker]; }
    out
}

pub struct Diarizer { pub name: String, pub dir: PathBuf, pub backend: String, net: Mutex<Net>, pub busy: AtomicBool, pub queued: AtomicU64 }

impl Diarizer {
    /// `r`: a local directory, or an HF repo id (looked up in / downloaded to the HF cache).
    pub fn load(r: &str, backend: &str, hf_cache: &Path) -> Result<Self> {
        let p = PathBuf::from(r);
        let dir = if p.join("model.safetensors").is_file() { p } else {
            let snap = crate::store::hfcache::find(hf_cache, r, None).filter(|d| d.join("model.safetensors").is_file());
            match snap { Some(d) => d, None => crate::store::hfcache::download(hf_cache, r, None, &|f, i, n| tracing::info!("diarizer download {i}/{n} {f}"))? }
        };
        let be = match backend {
            "cpu" | "flex" | "ndarray" => "cpu",
            // GPU only when it leaves >= 1 GiB free after the ~0.5 GiB of f32 weights + activations
            "auto" => if free_vram_mib().map(|f| f > 1024 + 1024).unwrap_or(false) { "cuda" } else { "cpu" },
            b => b,
        };
        let (dev, _, label) = crate::backend::select(be, "f32", None)?;
        let t = std::time::Instant::now();
        let net = Net::load(&dir, dev)?;
        tracing::info!("Nemotron-3-Diarization loaded from {} on {label} in {:.1}s", dir.display(), t.elapsed().as_secs_f64());
        Ok(Self { name: r.to_string(), dir, backend: label.to_string(), net: Mutex::new(net), busy: AtomicBool::new(false), queued: AtomicU64::new(0) })
    }
    /// Raw logits `[T,8]`, T, valid frames.
    pub fn logits(&self, samples: &[f32]) -> (Vec<f32>, usize, usize) {
        self.queued.fetch_add(1, Ordering::SeqCst);
        let net = self.net.lock().unwrap();
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.busy.store(true, Ordering::SeqCst);
        let r = net.logits(samples);
        self.busy.store(false, Ordering::SeqCst);
        r
    }
    /// Speaker turns (threshold 0.5, merged across < 0.3 s gaps, min 0.2 s).
    pub fn diarize(&self, samples: &[f32]) -> Vec<Seg> {
        let (lg, _, valid) = self.logits(samples);
        tidy(&segments(&lg, valid, 0.5), 0.3, 0.2)
    }
    /// Speaker turns with a custom same-speaker merge gap (s).
    pub fn turns(&self, samples: &[f32], gap: f64) -> Vec<Seg> {
        let (lg, _, valid) = self.logits(samples);
        tidy(&segments(&lg, valid, 0.5), gap, 0.2)
    }
    pub fn info(&self) -> Value { json!({"model": self.name, "backend": self.backend, "path": self.dir.display().to_string(), "busy": self.busy.load(Ordering::SeqCst), "max_speakers": NSPK}) }
}

pub fn label(s: usize) -> String { format!("SPEAKER_{s:02}") }

/// `GET /api/diarizer`: whether a diarizer is loaded (the web UI enables "Separate speakers" from it).
pub async fn status() -> axum::response::Response {
    use axum::response::IntoResponse;
    axum::Json(match DIAR.get() { Some(d) => json!({"loaded": true, "info": d.info()}), None => json!({"loaded": false}) }).into_response()
}

/// `POST /v1/audio/diarize` (multipart `file`): who spoke when, no text.
pub async fn handle(mut mp: axum::extract::Multipart) -> axum::response::Response {
    use crate::server::err;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    let Some(d) = DIAR.get().cloned() else { return err(StatusCode::SERVICE_UNAVAILABLE, "no diarization model loaded (start pyre with --load diar=nvidia/Nemotron-3-Diarization)") };
    let (mut file, mut name) = (vec![], "audio.wav".to_string());
    while let Ok(Some(f)) = mp.next_field().await {
        if f.name() == Some("file") { if let Some(n) = f.file_name() { name = n.to_string(); } file = f.bytes().await.map(|b| b.to_vec()).unwrap_or_default(); } else { let _ = f.bytes().await; }
    }
    if file.is_empty() { return err(StatusCode::BAD_REQUEST, "missing file field"); }
    let r = tokio::task::spawn_blocking(move || -> Result<Value> {
        let ext = Path::new(&name).extension().and_then(|e| e.to_str()).unwrap_or("wav").to_ascii_lowercase();
        let wav = pyre_whisper_audio::decode_bytes_to_mono_pcm(&file, &ext).map_err(|e| anyhow::anyhow!("could not decode audio ({ext}): {e}"))?;
        let wav = wav.ensure_sample_rate(16_000).map_err(|e| anyhow::anyhow!("resample: {e}"))?;
        let dur = wav.samples.len() as f64 / SR as f64;
        let t = std::time::Instant::now();
        let turns = d.diarize(&wav.samples);
        let secs = t.elapsed().as_secs_f64();
        tracing::info!("diarize: {dur:.2}s audio in {secs:.2}s, {} turns", turns.len());
        let n = turns.iter().map(|s| s.speaker + 1).max().unwrap_or(0);
        Ok(json!({"model": d.name, "duration": dur, "elapsed": secs, "num_speakers": n,
            "segments": turns.iter().map(|s| json!({"speaker": label(s.speaker), "start": s.start, "end": s.end})).collect::<Vec<_>>()}))
    }).await;
    match r { Ok(Ok(v)) => axum::Json(v).into_response(), Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")), Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()) }
}

/// Free VRAM (MiB) of GPU 0 per nvidia-smi.
fn free_vram_mib() -> Option<u64> {
    let mut c = std::process::Command::new("nvidia-smi");
    c.args(["--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"]);
    #[cfg(windows)]
    { use std::os::windows::process::CommandExt; c.creation_flags(0x0800_0000); }
    let o = c.output().ok()?;
    String::from_utf8_lossy(&o.stdout).trim().parse().ok()
}

/// `pyre diarize FILE`: load, diarize, print JSON (segments, timing); optional raw logits dump.
pub fn cli(hf_cache: &Path, file: &Path, model: &str, backend: &str, logits: Option<&Path>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let d = Diarizer::load(model, backend, hf_cache)?;
    let load = t0.elapsed().as_secs_f64();
    let bytes = std::fs::read(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("wav").to_ascii_lowercase();
    let wav = pyre_whisper_audio::decode_bytes_to_mono_pcm(&bytes, &ext).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let wav = wav.ensure_sample_rate(16_000).map_err(|e| anyhow::anyhow!("resample: {e}"))?;
    let dur = wav.samples.len() as f64 / SR as f64;
    let _ = d.logits(&wav.samples[..wav.samples.len().min(SR)]); // warm-up
    let t = std::time::Instant::now();
    let (lg, frames, valid) = d.logits(&wav.samples);
    let secs = t.elapsed().as_secs_f64();
    if let Some(p) = logits {
        let mut b = Vec::with_capacity(lg.len() * 4);
        for v in &lg { b.extend_from_slice(&v.to_le_bytes()); }
        std::fs::write(p, b)?;
    }
    let raw = segments(&lg, valid, 0.5);
    let turns = tidy(&raw, 0.3, 0.2);
    println!("{}", serde_json::to_string_pretty(&json!({"backend": d.backend, "load_s": load, "audio_s": dur, "diarize_s": secs, "rtf": secs / dur, "frames": frames, "raw_segments": raw, "turns": turns}))?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mel_bank_shape() {
        let m = mel_filters();
        assert_eq!(m.len(), NMEL * 257);
        assert!(m.iter().all(|v| *v >= 0.0));
        // every filter has some weight
        for i in 0..NMEL { assert!(m[i * 257..(i + 1) * 257].iter().any(|v| *v > 0.0), "filter {i}"); }
    }
    #[test]
    fn seg_tidy() {
        let s = vec![Seg { speaker: 3, start: 0.0, end: 1.0 }, Seg { speaker: 3, start: 1.1, end: 2.0 }, Seg { speaker: 5, start: 1.5, end: 1.6 }, Seg { speaker: 5, start: 2.5, end: 4.0 }];
        let t = tidy(&s, 0.3, 0.2);
        assert_eq!(t.len(), 2);
        assert_eq!((t[0].speaker, t[0].start, t[0].end), (0, 0.0, 2.0));
        assert_eq!((t[1].speaker, t[1].start), (1, 2.5));
    }
}
