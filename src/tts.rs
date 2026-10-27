//! Text-to-speech: a hand-written Burn port of VITS inference (HF `VitsModel`, e.g.
//! `facebook/mms-tts-eng`). No ONNX, no espeak: MMS-TTS models are character-level, so the
//! "phonemizer" is the model's own grapheme vocabulary (lowercase, filter, intersperse blanks).
//!
//! Pipeline: relative-position transformer text encoder -> stochastic duration predictor
//! (reverse flows; the rational-quadratic spline runs on the host, it is tiny) -> expand
//! prior by durations -> residual-coupling flow (reverse, WaveNet) -> HiFi-GAN -> 16 kHz wav.
use crate::model::loader::Weights;
use anyhow::{Context, Result};
use burn::tensor::activation::{gelu, leaky_relu, relu, sigmoid, softmax};
use burn::tensor::module::{conv1d, conv_transpose1d, embedding};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{DType, Device, Int, Tensor, TensorCreationOptions, TensorData};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Mutex;

#[derive(Deserialize, Clone, Debug)]
pub struct VitsConfig {
    hidden_size: usize, num_hidden_layers: usize, num_attention_heads: usize, window_size: usize,
    ffn_kernel_size: usize, flow_size: usize, layer_norm_eps: f64,
    depth_separable_num_layers: usize, duration_predictor_kernel_size: usize, duration_predictor_num_flows: usize,
    duration_predictor_flow_bins: usize, duration_predictor_tail_bound: f64,
    prior_encoder_num_flows: usize, prior_encoder_num_wavenet_layers: usize, wavenet_kernel_size: usize,
    upsample_rates: Vec<usize>, upsample_kernel_sizes: Vec<usize>, upsample_initial_channel: usize,
    resblock_kernel_sizes: Vec<usize>, resblock_dilation_sizes: Vec<Vec<usize>>, leaky_relu_slope: f64,
    pub sampling_rate: u32, noise_scale: f64, noise_scale_duration: f64, speaking_rate: f64,
    use_stochastic_duration_prediction: bool,
}

struct Conv { w: Tensor<3>, b: Option<Tensor<1>>, pad: usize, dil: usize, groups: usize }
impl Conv {
    fn f(&self, x: Tensor<3>) -> Tensor<3> { conv1d(x, self.w.clone(), self.b.clone(), ConvOptions::new([1], [self.pad], [self.dil], self.groups)) }
}
struct Ln { w: Tensor<1>, b: Tensor<1>, eps: f64 }
impl Ln {
    /// LayerNorm over the channel axis of `[1, C, T]` (or the last axis of `[1, T, C]` when `last`).
    fn f(&self, x: Tensor<3>, last: bool) -> Tensor<3> {
        let x = if last { x } else { x.swap_dims(1, 2) };
        let m = x.clone().mean_dim(2);
        let xc = x.sub(m);
        let v = xc.clone().powi_scalar(2).mean_dim(2);
        let y = xc.div(v.add_scalar(self.eps).sqrt()).mul(self.w.clone().unsqueeze::<3>()).add(self.b.clone().unsqueeze::<3>());
        if last { y } else { y.swap_dims(1, 2) }
    }
}
struct Lin { w: Tensor<2>, b: Tensor<1> }
impl Lin { fn f(&self, x: Tensor<3>) -> Tensor<3> { burn::tensor::module::linear(x, self.w.clone(), Some(self.b.clone())) } }

struct EncLayer { q: Lin, k: Lin, v: Lin, o: Lin, rel_k: Tensor<2>, rel_v: Tensor<2>, ln1: Ln, ff1: Conv, ff2: Conv, ln2: Ln }
struct Dds { dil: Vec<Conv>, pw: Vec<Conv>, n1: Vec<Ln>, n2: Vec<Ln> }
struct ConvFlow { pre: Conv, dds: Dds, proj: Conv }
struct Affine { t: [f32; 2], ls: [f32; 2] }
struct WaveNet { inl: Vec<Conv>, rs: Vec<Conv>, h: usize }
struct Coupling { pre: Conv, wn: WaveNet, post: Conv }
struct ResBlock { c1: Vec<Conv>, c2: Vec<Conv> }

pub struct Vits {
    pub cfg: VitsConfig,
    device: Device,
    vocab: HashMap<char, i32>,
    embed: Tensor<2>,
    enc: Vec<EncLayer>,
    proj: Conv,
    dp_pre: Conv, dp_dds: Dds, dp_proj: Conv,
    dp_flows: Vec<ConvFlow>, dp_affine: Affine,
    flows: Vec<Coupling>,
    dec_pre: Conv, ups: Vec<(Tensor<3>, Tensor<1>, usize, usize)>, res: Vec<ResBlock>, dec_post: Conv,
}

fn opts(d: &Device) -> TensorCreationOptions { TensorCreationOptions::new(d.clone()).with_dtype(DType::F32) }

impl Vits {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let cfg: VitsConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
        let vocab_j: HashMap<String, i32> = serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).context("vocab.json")?)?;
        let vocab = vocab_j.into_iter().filter_map(|(k, v)| { let mut c = k.chars(); let ch = c.next()?; c.next().is_none().then_some((ch, v)) }).collect();
        let ws = Weights::open(dir, device.clone(), DType::F32)?;
        let t1 = |n: &str| ws.get::<1>(n);
        let conv = |p: &str, pad: usize, dil: usize, groups: usize, bias: bool| -> Result<Conv> {
            let w = if ws.has(&format!("{p}.weight_v")) {
                let g: Tensor<3> = ws.get(&format!("{p}.weight_g"))?;
                let v: Tensor<3> = ws.get(&format!("{p}.weight_v"))?;
                let n = v.clone().powi_scalar(2).sum_dim(2).sum_dim(1).sqrt();
                v.div(n).mul(g)
            } else { ws.get(&format!("{p}.weight"))? };
            Ok(Conv { w, b: if bias { Some(ws.get(&format!("{p}.bias"))?) } else { None }, pad, dil, groups })
        };
        let ln = |p: &str| -> Result<Ln> { Ok(Ln { w: t1(&format!("{p}.weight"))?, b: t1(&format!("{p}.bias"))?, eps: cfg.layer_norm_eps }) };
        let lin = |p: &str| -> Result<Lin> { Ok(Lin { w: ws.get::<2>(&format!("{p}.weight"))?.transpose(), b: t1(&format!("{p}.bias"))? }) };
        let k = cfg.ffn_kernel_size;
        let mut enc = vec![];
        for i in 0..cfg.num_hidden_layers {
            let p = format!("text_encoder.encoder.layers.{i}");
            enc.push(EncLayer {
                q: lin(&format!("{p}.attention.q_proj"))?, k: lin(&format!("{p}.attention.k_proj"))?,
                v: lin(&format!("{p}.attention.v_proj"))?, o: lin(&format!("{p}.attention.out_proj"))?,
                rel_k: ws.get::<3>(&format!("{p}.attention.emb_rel_k"))?.squeeze_dim(0),
                rel_v: ws.get::<3>(&format!("{p}.attention.emb_rel_v"))?.squeeze_dim(0),
                ln1: ln(&format!("{p}.layer_norm"))?,
                ff1: conv(&format!("{p}.feed_forward.conv_1"), (k - 1) / 2, 1, 1, true)?,
                ff2: conv(&format!("{p}.feed_forward.conv_2"), (k - 1) / 2, 1, 1, true)?,
                ln2: ln(&format!("{p}.final_layer_norm"))?,
            });
        }
        let dk = cfg.duration_predictor_kernel_size;
        let dds = |p: &str| -> Result<Dds> {
            let mut d = Dds { dil: vec![], pw: vec![], n1: vec![], n2: vec![] };
            for i in 0..cfg.depth_separable_num_layers {
                let dil = dk.pow(i as u32);
                d.dil.push(conv(&format!("{p}.convs_dilated.{i}"), (dk * dil - dil) / 2, dil, cfg.hidden_size, true)?);
                d.pw.push(conv(&format!("{p}.convs_pointwise.{i}"), 0, 1, 1, true)?);
                d.n1.push(ln(&format!("{p}.norms_1.{i}"))?);
                d.n2.push(ln(&format!("{p}.norms_2.{i}"))?);
            }
            Ok(d)
        };
        let host2 = |n: &str| -> Result<[f32; 2]> { let v: Vec<f32> = ws.get::<2>(n)?.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?; Ok([v[0], v[1]]) };
        let mut dp_flows = vec![];
        for i in 1..=cfg.duration_predictor_num_flows {
            let p = format!("duration_predictor.flows.{i}");
            dp_flows.push(ConvFlow { pre: conv(&format!("{p}.conv_pre"), 0, 1, 1, true)?, dds: dds(&format!("{p}.conv_dds"))?, proj: conv(&format!("{p}.conv_proj"), 0, 1, 1, true)? });
        }
        let mut flows = vec![];
        let wk = cfg.wavenet_kernel_size;
        for i in 0..cfg.prior_encoder_num_flows {
            let p = format!("flow.flows.{i}");
            let mut wn = WaveNet { inl: vec![], rs: vec![], h: cfg.hidden_size };
            for j in 0..cfg.prior_encoder_num_wavenet_layers {
                wn.inl.push(conv(&format!("{p}.wavenet.in_layers.{j}"), (wk - 1) / 2, 1, 1, true)?);
                wn.rs.push(conv(&format!("{p}.wavenet.res_skip_layers.{j}"), 0, 1, 1, true)?);
            }
            flows.push(Coupling { pre: conv(&format!("{p}.conv_pre"), 0, 1, 1, true)?, wn, post: conv(&format!("{p}.conv_post"), 0, 1, 1, true)? });
        }
        let mut ups = vec![];
        for (i, (&r, &kk)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            ups.push((ws.get::<3>(&format!("decoder.upsampler.{i}.weight"))?, t1(&format!("decoder.upsampler.{i}.bias"))?, r, (kk - r) / 2));
        }
        let mut res = vec![];
        let nk = cfg.resblock_kernel_sizes.len();
        for i in 0..cfg.upsample_rates.len() * nk {
            let (kk, dl) = (cfg.resblock_kernel_sizes[i % nk], &cfg.resblock_dilation_sizes[i % nk]);
            let p = format!("decoder.resblocks.{i}");
            let mut rb = ResBlock { c1: vec![], c2: vec![] };
            for (j, &d) in dl.iter().enumerate() {
                rb.c1.push(conv(&format!("{p}.convs1.{j}"), (kk * d - d) / 2, d, 1, true)?);
                rb.c2.push(conv(&format!("{p}.convs2.{j}"), (kk - 1) / 2, 1, 1, true)?);
            }
            res.push(rb);
        }
        Ok(Self {
            embed: ws.get("text_encoder.embed_tokens.weight")?,
            proj: conv("text_encoder.project", 0, 1, 1, true)?,
            dp_pre: conv("duration_predictor.conv_pre", 0, 1, 1, true)?, dp_dds: dds("duration_predictor.conv_dds")?,
            dp_proj: conv("duration_predictor.conv_proj", 0, 1, 1, true)?,
            dp_affine: Affine { t: host2("duration_predictor.flows.0.translate")?, ls: host2("duration_predictor.flows.0.log_scale")? },
            dp_flows, flows,
            dec_pre: conv("decoder.conv_pre", 3, 1, 1, true)?, ups, res, dec_post: conv("decoder.conv_post", 3, 1, 1, false)?,
            enc, vocab, device, cfg,
        })
    }

    /// VitsTokenizer: lowercase, keep vocab characters, collapse spaces, intersperse blank (id 0).
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        let lower = text.to_lowercase();
        let mut s = String::new();
        for c in lower.chars() {
            let c = if c.is_whitespace() { ' ' } else { c };
            if c == ' ' && s.ends_with(' ') { continue; }
            if c == ' ' || self.vocab.contains_key(&c) { s.push(c); }
        }
        let ids: Vec<i32> = s.trim().chars().filter_map(|c| self.vocab.get(&c).copied()).collect();
        let mut out = vec![0; ids.len() * 2 + 1];
        for (i, id) in ids.iter().enumerate() { out[2 * i + 1] = *id; }
        out
    }

    fn dds(&self, d: &Dds, mut x: Tensor<3>) -> Tensor<3> {
        for i in 0..d.dil.len() {
            let h = gelu(d.n1[i].f(d.dil[i].f(x.clone()), false));
            let h = gelu(d.n2[i].f(d.pw[i].f(h), false));
            x = x.add(h);
        }
        x
    }

    /// Relative-position self-attention (window `w`) for one layer, x: [1, L, C].
    fn attn(&self, l: &EncLayer, x: Tensor<3>) -> Tensor<3> {
        let [_, n, c] = x.dims();
        let h = self.cfg.num_attention_heads;
        let hd = c / h;
        let w = self.cfg.window_size;
        let r = 2 * w + 1;
        let q = l.q.f(x.clone()).mul_scalar((hd as f64).powf(-0.5)).reshape([n, h, hd]).swap_dims(0, 1); // [H,L,hd]
        let k = l.k.f(x.clone()).reshape([n, h, hd]).swap_dims(0, 1);
        let v = l.v.f(x).reshape([n, h, hd]).swap_dims(0, 1);
        let mut s = q.clone().matmul(k.swap_dims(1, 2)); // [H,L,L]
        // one-hot M[i, j, r] = 1 iff j - i == r - w
        let mut m = vec![0f32; n * n * r];
        for i in 0..n { for j in 0..n { let d = j as isize - i as isize; if d.unsigned_abs() <= w { m[(i * n + j) * r + (d + w as isize) as usize] = 1.0; } } }
        let m = Tensor::<3>::from_data(TensorData::new(m, [n, n, r]), opts(&self.device)).unsqueeze::<4>(); // [1,L,L,R]
        let relk = q.matmul(l.rel_k.clone().transpose().unsqueeze::<3>()); // [H,L,R]
        s = s.add(relk.unsqueeze_dim::<4>(2).mul(m.clone()).sum_dim(3).squeeze_dim(3));
        let p = softmax(s, 2);
        let mut o = p.clone().matmul(v); // [H,L,hd]
        let relw = p.unsqueeze_dim::<4>(3).mul(m).sum_dim(2).squeeze_dim(2); // [H,L,R]
        o = o.add(relw.matmul(l.rel_v.clone().unsqueeze::<3>()));
        l.o.f(o.swap_dims(0, 1).reshape([1, n, c]))
    }

    /// Synthesize mono f32 samples at `cfg.sampling_rate`. `noise`: scale noise (1.0 = model
    /// defaults, 0.0 = deterministic). `seed` seeds the host RNG.
    pub fn synth(&self, text: &str, speed: f64, noise: f64, seed: u64) -> Result<Vec<f32>> {
        let ids = self.tokenize(text);
        if ids.len() <= 1 { anyhow::bail!("no speakable characters in input"); }
        let n = ids.len();
        let dev = &self.device;
        let mut rng = Rng(seed.max(1));
        let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), dev);
        let mut x = embedding(self.embed.clone(), idx).mul_scalar((self.cfg.hidden_size as f64).sqrt()); // [1,L,C]
        for l in &self.enc {
            let a = self.attn(l, x.clone());
            x = l.ln1.f(x.add(a), true);
            let f = l.ff2.f(relu(l.ff1.f(x.clone().swap_dims(1, 2)))).swap_dims(1, 2);
            x = l.ln2.f(x.add(f), true);
        }
        let hid = x.swap_dims(1, 2); // [1,C,L]
        let stats = self.proj.f(hid.clone()); // [1,2F,L]
        let fs = self.cfg.flow_size;
        let means = stats.clone().narrow(1, 0, fs);
        let logs = stats.narrow(1, fs, fs);
        // --- stochastic duration predictor (reverse) ---
        if !self.cfg.use_stochastic_duration_prediction { anyhow::bail!("deterministic duration predictor not ported"); }
        let cond = self.dp_proj.f(self.dds(&self.dp_dds, self.dp_pre.f(hid)));
        let ns = self.cfg.noise_scale_duration * noise;
        let mut lat: Vec<[f32; 2]> = (0..n).map(|_| [(rng.normal() * ns) as f32, (rng.normal() * ns) as f32]).collect();
        let nb = self.cfg.duration_predictor_flow_bins;
        let fc = (self.cfg.hidden_size as f32).sqrt();
        // HF reverse order: flows[4], [3], [2] (flows[1] is the skipped useless vflow), then the affine flow.
        for fl in self.dp_flows[1..].iter().rev() {
            for p in lat.iter_mut() { p.swap(0, 1); }
            let first = Tensor::<3>::from_data(TensorData::new(lat.iter().map(|p| p[0]).collect::<Vec<_>>(), [1, 1, n]), opts(dev));
            let h = fl.proj.f(self.dds(&fl.dds, fl.pre.f(first).add(cond.clone())));
            let hv: Vec<f32> = h.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?; // [29, L]
            for t in 0..n {
                let col: Vec<f32> = (0..3 * nb - 1).map(|c| hv[c * n + t]).collect();
                let wv: Vec<f32> = col[..nb].iter().map(|v| v / fc).collect();
                let hh: Vec<f32> = col[nb..2 * nb].iter().map(|v| v / fc).collect();
                lat[t][1] = rq_spline_inverse(lat[t][1], &wv, &hh, &col[2 * nb..], self.cfg.duration_predictor_tail_bound as f32);
            }
        }
        // (the second-to-last flow is HF's "useless vflow" and is skipped), then the affine flow
        for p in lat.iter_mut() { p.swap(0, 1); }
        let a = &self.dp_affine;
        let logw: Vec<f32> = lat.iter().map(|p| (p[0] - a.t[0]) * (-a.ls[0]).exp()).collect();
        let scale = 1.0 / (self.cfg.speaking_rate * speed.max(0.1));
        let dur: Vec<usize> = logw.iter().map(|l| ((*l as f64).exp() * scale).ceil().max(0.0) as usize).collect();
        let frames: usize = dur.iter().sum::<usize>().max(1);
        let mut map = Vec::with_capacity(frames);
        for (i, d) in dur.iter().enumerate() { for _ in 0..*d { map.push(i as i32); } }
        if map.is_empty() { map.push(0); }
        let t = map.len();
        let fi = Tensor::<1, Int>::from_data(TensorData::new(map, [t]), dev);
        let m = means.select(2, fi.clone());
        let lv = logs.select(2, fi);
        let eps: Vec<f32> = (0..fs * t).map(|_| (rng.normal() * self.cfg.noise_scale * noise) as f32).collect();
        let eps = Tensor::<3>::from_data(TensorData::new(eps, [1, fs, t]), opts(dev));
        let mut z = m.add(eps.mul(lv.exp()));
        // --- flow (reverse) ---
        let half = fs / 2;
        for c in self.flows.iter().rev() {
            z = z.flip([1]);
            let a = z.clone().narrow(1, 0, half);
            let b = z.narrow(1, half, half);
            let mut hcur = c.pre.f(a.clone());
            let mut out: Option<Tensor<3>> = None;
            let nl = c.wn.inl.len();
            for i in 0..nl {
                let hs = c.wn.inl[i].f(hcur.clone());
                let acts = hs.clone().narrow(1, 0, c.wn.h).tanh().mul(sigmoid(hs.narrow(1, c.wn.h, c.wn.h)));
                let rs = c.wn.rs[i].f(acts);
                let skip = if i < nl - 1 {
                    hcur = hcur.add(rs.clone().narrow(1, 0, c.wn.h));
                    rs.narrow(1, c.wn.h, c.wn.h)
                } else { rs };
                out = Some(match out { Some(o) => o.add(skip), None => skip });
            }
            let mean = c.post.f(out.unwrap());
            z = Tensor::cat(vec![a, b.sub(mean)], 1);
        }
        // --- HiFi-GAN ---
        let mut h = self.dec_pre.f(z);
        let nk = self.cfg.resblock_kernel_sizes.len();
        let slope = self.cfg.leaky_relu_slope;
        for (i, (w, b, r, p)) in self.ups.iter().enumerate() {
            h = leaky_relu(h, slope);
            h = conv_transpose1d(h, w.clone(), Some(b.clone()), ConvTransposeOptions::new([*r], [*p], [0], [1], 1));
            let mut acc: Option<Tensor<3>> = None;
            for j in 0..nk {
                let rb = &self.res[i * nk + j];
                let mut y = h.clone();
                for (c1, c2) in rb.c1.iter().zip(&rb.c2) {
                    let r0 = y.clone();
                    y = c2.f(leaky_relu(c1.f(leaky_relu(y, slope)), slope)).add(r0);
                }
                acc = Some(match acc { Some(a) => a.add(y), None => y });
            }
            h = acc.unwrap().div_scalar(nk as f64);
        }
        let wav = self.dec_post.f(leaky_relu(h, 0.01)).tanh();
        wav.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

/// Inverse piecewise rational-quadratic spline (HF `_unconstrained_rational_quadratic_spline`,
/// reverse=True) for one scalar; identity outside `[-B, B]`.
fn rq_spline_inverse(x: f32, uw: &[f32], uh: &[f32], ud: &[f32], bound: f32) -> f32 {
    if x < -bound || x > bound { return x; }
    let (minw, minh, mind) = (1e-3f32, 1e-3f32, 1e-3f32);
    let nb = uw.len();
    let sm = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect(); let s: f32 = e.iter().sum(); e.into_iter().map(|x| x / s).collect::<Vec<_>>() };
    let cum = |v: Vec<f32>, minv: f32| {
        let v: Vec<f32> = v.iter().map(|x| minv + (1.0 - minv * nb as f32) * x).collect();
        let mut c = vec![0f32; nb + 1];
        for i in 0..nb { c[i + 1] = c[i] + v[i]; }
        for x in c.iter_mut() { *x = 2.0 * bound * *x - bound; }
        c[0] = -bound; c[nb] = bound;
        c
    };
    let cw = cum(sm(uw), minw);
    let ch = cum(sm(uh), minh);
    let constant = ((1.0f64 - mind as f64).exp() - 1.0).ln() as f32;
    let mut d = Vec::with_capacity(nb + 1);
    d.push(constant); d.extend_from_slice(ud); d.push(constant);
    let deriv: Vec<f32> = d.iter().map(|u| mind + if *u > 20.0 { *u } else { u.exp().ln_1p() }).collect();
    let mut loc = ch.clone();
    loc[nb] += 1e-6;
    let bin = (loc.iter().filter(|&&l| x >= l).count() as isize - 1).clamp(0, nb as isize - 1) as usize;
    let (in_cw, in_w) = (cw[bin], cw[bin + 1] - cw[bin]);
    let (in_ch, in_h) = (ch[bin], ch[bin + 1] - ch[bin]);
    let delta = in_h / in_w;
    let (d0, d1) = (deriv[bin], deriv[bin + 1]);
    let i1 = d0 + d1 - 2.0 * delta;
    let i2 = x - in_ch;
    let i3 = i2 * i1;
    let a = in_h * (delta - d0) + i3;
    let b = in_h * d0 - i3;
    let c = -delta * i2;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let root = (2.0 * c) / (-b - disc.sqrt());
    root * in_w + in_cw
}

/// xorshift64* + Box-Muller.
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 { self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27; ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos() }
}

/// 16-bit PCM WAV bytes.
pub fn wav_bytes(s: &[f32], rate: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(44 + s.len() * 2);
    let data = (s.len() * 2) as u32;
    v.extend_from_slice(b"RIFF"); v.extend_from_slice(&(36 + data).to_le_bytes()); v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes()); v.extend_from_slice(&1u16.to_le_bytes()); v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes()); v.extend_from_slice(&(rate * 2).to_le_bytes()); v.extend_from_slice(&2u16.to_le_bytes()); v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data"); v.extend_from_slice(&data.to_le_bytes());
    for x in s { v.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()); }
    v
}

pub struct TtsModel { pub vits: Mutex<Vits>, pub busy: AtomicBool, pub queued: AtomicU64, pub backend: String }
impl TtsModel {
    pub fn load(dir: &Path, backend: &str) -> Result<Self> {
        let (dev, _, label) = crate::backend::select(backend, "f32", None)?;
        let t = std::time::Instant::now();
        let v = Vits::load(dir, dev)?;
        tracing::info!("VITS loaded from {} on {label} in {:.1}s", dir.display(), t.elapsed().as_secs_f64());
        Ok(Self { vits: Mutex::new(v), busy: AtomicBool::new(false), queued: AtomicU64::new(0), backend: label.into() })
    }
    pub fn describe(&self) -> String { "VITS (Burn port; MMS-TTS character vocab, no phonemizer)".into() }
}
