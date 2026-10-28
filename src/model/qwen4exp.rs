//! Qwen3.8-Flash-Next (HF `qwen4_exp` / `Qwen4ExpForConditionalGeneration`, text part) on Burn.
//!
//! Architecture (from transformers `models/qwen4_exp/modeling_qwen4_exp.py`, Apache-2.0):
//! - `hc_count` hyper-connection residual streams (`[T, hc*H]`), mixed per block by a gated
//!   low-rank `GatedResidual` and re-injected with per-stream weights; final mixer, no final norm.
//! - 3 of 4 layers: Gated DeltaNet linear attention (causal depthwise conv + delta-rule recurrent
//!   state `[Hv, Dk, Dv]`, gated RMSNorm with sigmoid output gate).
//! - every 4th layer: gated full attention (q/gate interleaved per head, q/k RMSNorm(1+w), partial
//!   RoPE over 25% of head_dim, GQA). The QSA indexer (block top-k sparse attention) only changes
//!   the result once more than `indexer_budget` (2048) tokens are visible; it is not implemented
//!   here, so contexts are capped at that budget for exact results.
//! - MoE: softmax router, top-k (renormalised), SwiGLU experts + sigmoid-gated shared expert.
//! - PLE: hashed 2/3-gram embedding table injected at `ple_layer_ids` via a gated, dilated conv.
//! Experts and the n-gram table can live on a second device (CPU) so the dense part fits a 12 GB
//! GPU - the same split Strata (github.com/Niko1221/Strata, MIT) uses: dense/attention/DeltaNet/
//! router/shared expert on the GPU, routed experts in RAM computed by the CPU.
//! Batch size 1; prefill runs the recurrent DeltaNet form token by token (exact, not fast).
use super::loader::Weights;
use anyhow::{bail, Context, Result};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorCreationOptions, TensorData};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Cfg {
    pub hidden: usize, pub n_layers: usize, pub linear: Vec<bool>,
    pub heads: usize, pub kv_heads: usize, pub head_dim: usize, pub rot_dim: usize, pub rope_theta: f64,
    pub eps: f64,
    pub lk_heads: usize, pub lv_heads: usize, pub lk_dim: usize, pub lv_dim: usize, pub conv_k: usize,
    pub gate_sigmoid: bool,
    pub n_exp: usize, pub top_k: usize, pub moe_i: usize, pub shared_i: usize, pub norm_topk: bool,
    pub hc: usize, pub hc_lowrank: usize, pub vocab: usize, pub eos: u32,
    pub ngram_size: usize, pub heads_per_ngram: usize, pub ngram_base: u64, pub ngram_div: u64,
    pub ple_dim: usize, pub ple_conv_k: usize, pub ple_layer_ids: Vec<usize>, pub seed: u64,
    pub indexer_budget: usize, pub max_pos: usize,
}

impl Cfg {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
        let t = v.get("text_config").cloned().unwrap_or(v.clone());
        let u = |k: &str| -> Result<usize> { t.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).with_context(|| format!("config: {k}")) };
        let uo = |k: &str, d: usize| t.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d);
        let n_layers = u("num_hidden_layers")?;
        let linear: Vec<bool> = match t.get("layer_types").and_then(|x| x.as_array()) {
            Some(a) => a.iter().map(|s| s.as_str() == Some("linear_attention")).collect(),
            None => { let iv = uo("full_attention_interval", 4); (0..n_layers).map(|i| (i + 1) % iv != 0).collect() }
        };
        let head_dim = u("head_dim")?;
        let rp = t.get("rope_parameters").cloned().unwrap_or(Value::Null);
        let prf = rp.get("partial_rotary_factor").or(t.get("partial_rotary_factor")).and_then(|x| x.as_f64()).unwrap_or(1.0);
        let theta = rp.get("rope_theta").or(t.get("rope_theta")).and_then(|x| x.as_f64()).unwrap_or(1e7);
        let eos = match t.get("eos_token_id") { Some(Value::Array(a)) => a[0].as_u64().unwrap_or(0), Some(x) => x.as_u64().unwrap_or(0), None => 0 } as u32;
        let gate = t.get("output_gate_type").and_then(|x| x.as_str()).or(t.get("hidden_act").and_then(|x| x.as_str())).unwrap_or("silu");
        let ple_layer_ids = t.get("ple_layer_ids").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
        Ok(Self {
            hidden: u("hidden_size")?, n_layers, linear, heads: u("num_attention_heads")?, kv_heads: u("num_key_value_heads")?,
            head_dim, rot_dim: (head_dim as f64 * prf) as usize, rope_theta: theta, eps: t.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6),
            lk_heads: u("linear_num_key_heads")?, lv_heads: u("linear_num_value_heads")?, lk_dim: u("linear_key_head_dim")?, lv_dim: u("linear_value_head_dim")?,
            conv_k: uo("linear_conv_kernel_dim", 4), gate_sigmoid: gate == "sigmoid",
            n_exp: u("num_experts")?, top_k: u("num_experts_per_tok")?, moe_i: u("moe_intermediate_size")?, shared_i: u("shared_expert_intermediate_size")?,
            norm_topk: t.get("norm_topk_prob").and_then(|x| x.as_bool()).unwrap_or(true),
            hc: uo("hc_count", 4), hc_lowrank: u("hc_lowrank")?, vocab: u("vocab_size")?, eos,
            ngram_size: uo("ngram_size", 3), heads_per_ngram: uo("heads_per_ngram", 8), ngram_base: uo("ngram_vocab_size_base", 20_000_000) as u64,
            ngram_div: uo("make_ngram_vocab_size_divisible_by", 128) as u64, ple_dim: uo("ple_embed_dim", 0), ple_conv_k: uo("ple_conv_kernel_size", 4),
            ple_layer_ids, seed: uo("seed", 1234) as u64, indexer_budget: uo("indexer_budget", 2048), max_pos: uo("max_position_embeddings", 262144),
        })
    }
}

// ---------- PLE n-gram hashing (bit-exact port of the Python helpers) ----------
const GAMMA: u64 = 0x9E3779B97F4A7C15;
fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
    v ^ (v >> 31)
}
fn layer_multipliers(vocab: u64, n: usize, ple_index: u64, seed: u64) -> Vec<i64> {
    let max_long = (1u64 << 63) - 1;
    let half = (max_long / vocab.max(1) / 2).max(1);
    let base = seed + 10007 * ple_index;
    (0..n).map(|i| { let v = base.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1)); (2 * (splitmix64(v) % half) + 1) as i64 }).collect()
}
fn is_prime(v: u64) -> bool {
    if v < 2 { return false; }
    if v % 2 == 0 { return v == 2; }
    let mut d = 3; while d * d <= v { if v % d == 0 { return false; } d += 2; } true
}
fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut p = start;
    for _ in 0..count { p += 1; while !is_prime(p) { p += 1; } }
    p
}

struct Lin { w: Tensor<2> }
impl Lin {
    fn load(ws: &Weights, n: &str) -> Result<Self> { let w: Tensor<2> = ws.get(&format!("{n}.weight"))?; Ok(Self { w: w.transpose() }) }
    fn fwd(&self, x: Tensor<2>) -> Tensor<2> { x.matmul(self.w.clone()) }
}

/// Qwen RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w), optionally per group of `group` features.
struct Norm { w1: Tensor<1>, group: Option<usize> }
impl Norm {
    fn load(ws: &Weights, n: &str, group: Option<usize>) -> Result<Self> { let w: Tensor<1> = ws.get(&format!("{n}.weight"))?; Ok(Self { w1: w.add_scalar(1.0), group }) }
    fn fwd(&self, x: Tensor<2>, eps: f64) -> Tensor<2> {
        let [t, d] = x.dims();
        let y = match self.group {
            Some(g) => { let x3: Tensor<3> = x.reshape([t, d / g, g]); rms3(x3, eps).reshape([t, d]) }
            None => rms2(x, eps),
        };
        y.mul(self.w1.clone().unsqueeze::<2>())
    }
}
fn rms2(x: Tensor<2>, eps: f64) -> Tensor<2> { let ms = x.clone().powi_scalar(2).mean_dim(1); x.mul(ms.add_scalar(eps).sqrt().recip()) }
fn rms3(x: Tensor<3>, eps: f64) -> Tensor<3> { let ms = x.clone().powi_scalar(2).mean_dim(2); x.mul(ms.add_scalar(eps).sqrt().recip()) }

struct Hyper { norm: Norm, down: Lin, up: Lin, inject: Option<Lin> }
impl Hyper {
    fn load(ws: &Weights, p: &str, h: usize, combine: bool) -> Result<Self> {
        Ok(Self { norm: Norm::load(ws, &format!("{p}.hc_norm"), Some(h))?, down: Lin::load(ws, &format!("{p}.input_mix_weight_down"))?,
            up: Lin::load(ws, &format!("{p}.input_mix_weight_up"))?, inject: if combine { Some(Lin::load(ws, &format!("{p}.block_inject_weight"))?) } else { None } })
    }
    /// -> (mixed [T,H], injection weights [T,hc])
    fn fwd(&self, x: Tensor<2>, hc: usize, eps: f64) -> (Tensor<2>, Option<Tensor<2>>) {
        let [t, d] = x.dims();
        let h = d / hc;
        let xn = self.norm.fwd(x, eps);
        let mw = silu(self.down.fwd(xn.clone()).div_scalar(hc as f64));
        let mw = sigmoid(self.up.fwd(mw));
        let mixed: Tensor<2> = mw.mul(xn.clone()).reshape([t, hc, h]).mean_dim(1).reshape([t, h]);
        let inj = self.inject.as_ref().map(|l| sigmoid(l.fwd(xn).div_scalar(hc as f64)).mul_scalar(2.0));
        (mixed, inj)
    }
}
fn inject(x: Tensor<2>, out: Tensor<2>, w: Tensor<2>) -> Tensor<2> {
    let [t, h] = out.dims();
    let hc = w.dims()[1];
    let o3: Tensor<3> = out.reshape([t, 1, h]);
    let w3: Tensor<3> = w.reshape([t, hc, 1]);
    x.add(o3.mul(w3).reshape([t, hc * h]))
}

/// Causal depthwise conv over [T, C] with `state` = the previous `(k-1)*dil` rows; weight [C, k].
fn dconv(state: Tensor<2>, x: Tensor<2>, w: &Tensor<2>, dil: usize) -> (Tensor<2>, Tensor<2>) {
    let [t, c] = x.dims();
    let s = state.dims()[0];
    let xp = Tensor::cat(vec![state, x], 0);
    let k = w.dims()[1];
    let mut out: Option<Tensor<2>> = None;
    for j in 0..k {
        let wj: Tensor<2> = w.clone().narrow(1, j, 1).reshape([1, c]);
        let term = xp.clone().narrow(0, j * dil, t).mul(wj);
        out = Some(match out { Some(o) => o.add(term), None => term });
    }
    let new_state = xp.narrow(0, t, s);
    (out.unwrap(), new_state)
}

struct DeltaNet { qkv: Lin, z: Lin, b: Lin, a: Lin, conv_w: Tensor<2>, neg_a: Tensor<1>, dt_bias: Tensor<1>, norm_w: Tensor<1>, out: Lin }
struct Attn { q: Lin, k: Lin, v: Lin, o: Lin, qn: Norm, kn: Norm }
enum Mixer { Linear(DeltaNet), Full(Attn) }

struct Moe { router: Lin, gate_up: Tensor<3>, down: Tensor<3>, sh_gate: Lin, sh_up: Lin, sh_down: Lin, sh_g: Lin }

struct Ple { emb: Tensor<2>, sizes: Vec<i64>, offsets: Vec<i64>, mult: Vec<i64>, key: Lin, value: Lin, nk: Norm, nq: Norm, nc: Norm, conv_w: Tensor<2> }

struct Layer { mixer: Mixer, moe: Moe, ple: Option<Ple>, ahc: Hyper, mhc: Hyper }

pub struct State {
    pub pos: usize,
    conv: Vec<Option<Tensor<2>>>, rec: Vec<Option<Tensor<3>>>,
    k: Vec<Option<Tensor<3>>>, v: Vec<Option<Tensor<3>>>,
    ple_ctx: Vec<u32>, ple_conv: Vec<Option<Tensor<2>>>,
}

pub struct Qwen4Exp {
    pub cfg: Cfg, device: Device, xdev: Device,
    embed: Tensor<2>, layers: Vec<Layer>, mixer: Hyper, lm_head: Lin,
    inv_freq: Vec<f32>,
}

fn find_prefix(ws: &Weights) -> &'static str {
    if ws.has("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" }
}

impl Qwen4Exp {
    /// `xdev`: device for routed experts and the PLE n-gram table (e.g. CPU); `device`: everything else.
    pub fn load(dir: &Path, device: Device, xdev: Device) -> Result<Self> {
        let cfg = Cfg::from_dir(dir)?;
        let ws = Weights::open(dir, device.clone(), FloatDType::F32.into())?;
        let wx = Weights::open(dir, xdev.clone(), FloatDType::F32.into())?;
        let p = find_prefix(&ws);
        let h = cfg.hidden;
        let mut layers = vec![];
        let ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
        for i in 0..cfg.n_layers {
            let lp = format!("{p}.layers.{i}");
            let mixer = if cfg.linear[i] {
                let a = format!("{lp}.linear_attn");
                let cw: Tensor<3> = ws.get(&format!("{a}.conv1d.weight"))?;
                let [c, _, k] = cw.dims();
                let alog: Tensor<1> = ws.get(&format!("{a}.A_log"))?;
                Mixer::Linear(DeltaNet {
                    qkv: Lin::load(&ws, &format!("{a}.in_proj_qkv"))?, z: Lin::load(&ws, &format!("{a}.in_proj_z"))?,
                    b: Lin::load(&ws, &format!("{a}.in_proj_b"))?, a: Lin::load(&ws, &format!("{a}.in_proj_a"))?,
                    conv_w: cw.reshape([c, k]), neg_a: alog.exp().neg(), dt_bias: ws.get(&format!("{a}.dt_bias"))?,
                    norm_w: ws.get(&format!("{a}.norm.weight"))?, out: Lin::load(&ws, &format!("{a}.out_proj"))?,
                })
            } else {
                let a = format!("{lp}.self_attn");
                Mixer::Full(Attn { q: Lin::load(&ws, &format!("{a}.q_proj"))?, k: Lin::load(&ws, &format!("{a}.k_proj"))?, v: Lin::load(&ws, &format!("{a}.v_proj"))?,
                    o: Lin::load(&ws, &format!("{a}.o_proj"))?, qn: Norm::load(&ws, &format!("{a}.q_norm"), None)?, kn: Norm::load(&ws, &format!("{a}.k_norm"), None)? })
            };
            let m = format!("{lp}.mlp");
            // fused [E, 2I, H] / [E, H, I] (Qwen's checkpoint) or per-expert gate/up/down (save_pretrained legacy)
            let (gu, dn): (Tensor<3>, Tensor<3>) = if wx.has(&format!("{m}.experts.gate_up_proj")) {
                (wx.get::<3>(&format!("{m}.experts.gate_up_proj"))?.swap_dims(1, 2), wx.get::<3>(&format!("{m}.experts.down_proj"))?.swap_dims(1, 2))
            } else {
                let (mut g, mut d) = (vec![], vec![]);
                for e in 0..cfg.n_exp {
                    let ep = format!("{m}.experts.{e}");
                    let gw: Tensor<2> = wx.get(&format!("{ep}.gate_proj.weight"))?;
                    let uw: Tensor<2> = wx.get(&format!("{ep}.up_proj.weight"))?;
                    let dw: Tensor<2> = wx.get(&format!("{ep}.down_proj.weight"))?;
                    g.push(Tensor::cat(vec![gw, uw], 0).transpose().unsqueeze::<3>());
                    d.push(dw.transpose().unsqueeze::<3>());
                }
                (Tensor::cat(g, 0), Tensor::cat(d, 0))
            };
            let moe = Moe { router: Lin::load(&ws, &format!("{m}.gate"))?, gate_up: gu, down: dn,
                sh_gate: Lin::load(&ws, &format!("{m}.shared_expert.gate_proj"))?, sh_up: Lin::load(&ws, &format!("{m}.shared_expert.up_proj"))?,
                sh_down: Lin::load(&ws, &format!("{m}.shared_expert.down_proj"))?, sh_g: Lin::load(&ws, &format!("{m}.shared_expert_gate"))? };
            let ple = match cfg.ple_layer_ids.iter().position(|&x| x == i + 1) {
                None => None,
                Some(pi) => {
                    let pp = format!("{lp}.ple");
                    let en = format!("{pp}.ple_embedding.ngram_embedding");
                    let emb: Tensor<2> = if wx.has(&format!("{en}.weight")) { wx.get(&format!("{en}.weight"))? } else {
                        let mut parts = vec![]; let mut s = 0;
                        while wx.has(&format!("{en}.shard_{s}.weight")) { parts.push(wx.get::<2>(&format!("{en}.shard_{s}.weight"))?); s += 1; }
                        if parts.is_empty() { bail!("missing {en}"); }
                        Tensor::cat(parts, 0)
                    };
                    let (mut sizes, mut offsets, mut tot) = (vec![], vec![], 0i64);
                    for hi in 0..ngram_heads {
                        let g = (pi * ngram_heads + hi) as u64;
                        let sz = nth_prime_after(cfg.ngram_base - 1, g + 1) as i64;
                        sizes.push(sz); offsets.push(tot); tot += sz;
                    }
                    let cw: Tensor<3> = ws.get(&format!("{pp}.conv1d.weight"))?;
                    let [c, _, k] = cw.dims();
                    let hch = h * cfg.hc;
                    Some(Ple { emb, sizes, offsets, mult: layer_multipliers(cfg.vocab as u64, cfg.ngram_size, pi as u64, cfg.seed),
                        key: Lin::load(&ws, &format!("{pp}.key_proj"))?, value: Lin::load(&ws, &format!("{pp}.value_proj"))?,
                        nk: Norm::load(&ws, &format!("{pp}.norm_key"), Some(h))?, nq: Norm::load(&ws, &format!("{pp}.norm_query"), Some(h))?,
                        nc: Norm::load(&ws, &format!("{pp}.norm_conv"), Some(h))?, conv_w: { let _ = hch; cw.reshape([c, k]) } })
                }
            };
            layers.push(Layer { mixer, moe, ple, ahc: Hyper::load(&ws, &format!("{lp}.attn_hyper_connection"), h, true)?, mhc: Hyper::load(&ws, &format!("{lp}.mlp_hyper_connection"), h, true)? });
        }
        let lm = if ws.has("lm_head.weight") { Lin::load(&ws, "lm_head")? } else { Lin::load(&ws, &format!("{p}.embed_tokens"))? };
        let half = cfg.rot_dim / 2;
        let inv_freq = (0..half).map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / cfg.rot_dim as f64)) as f32).collect();
        Ok(Self { embed: ws.get(&format!("{p}.embed_tokens.weight"))?, layers, mixer: Hyper::load(&ws, &format!("{p}.hyper_connection_mixer"), h, false)?,
            lm_head: lm, inv_freq, cfg, device, xdev })
    }

    pub fn new_state(&self) -> State {
        let n = self.cfg.n_layers;
        State { pos: 0, conv: vec![None; n], rec: vec![None; n], k: vec![None; n], v: vec![None; n], ple_ctx: vec![self.cfg.eos; self.cfg.ngram_size - 1], ple_conv: vec![None; n] }
    }
    pub fn max_ctx(&self) -> usize { self.cfg.indexer_budget.min(self.cfg.max_pos) }

    fn f(&self, v: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::from_data(TensorData::new(v, shape), TensorCreationOptions::new(self.device.clone()).with_dtype(DType::F32))
    }

    /// Run `ids` (appended at `st.pos`); returns logits for every position `[T, vocab]`.
    pub fn forward(&self, ids: &[u32], st: &mut State) -> Tensor<2> {
        let c = &self.cfg;
        let t = ids.len();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids.iter().map(|&x| x as i32).collect::<Vec<_>>(), [t]), &self.device);
        let e: Tensor<2> = self.embed.clone().select(0, idx);
        let mut x = Tensor::cat(vec![e; c.hc], 1);
        // rope tables for positions pos..pos+t (text-only mrope == plain rope)
        let half = c.rot_dim / 2;
        let (mut cs, mut sn) = (Vec::with_capacity(t * c.rot_dim), Vec::with_capacity(t * c.rot_dim));
        for i in 0..t {
            let p = (st.pos + i) as f32;
            for r in 0..2 { let _ = r; for j in 0..half { let a = p * self.inv_freq[j]; cs.push(a.cos()); sn.push(a.sin()); } }
        }
        let cos = self.f(cs, [t, c.rot_dim]);
        let sin = self.f(sn, [t, c.rot_dim]);
        let ple_ctx = st.ple_ctx.clone();
        for (li, l) in self.layers.iter().enumerate() {
            if let Some(p) = &l.ple { x = x.clone().add(self.ple(p, li, x, ids, &ple_ctx, st)); }
            let (hin, w) = l.ahc.fwd(x.clone(), c.hc, c.eps);
            let o = match &l.mixer { Mixer::Linear(d) => self.deltanet(d, li, hin, st), Mixer::Full(a) => self.attn(a, li, hin, &cos, &sin, st) };
            x = inject(x, o, w.unwrap());
            let (hin, w) = l.mhc.fwd(x.clone(), c.hc, c.eps);
            let o = self.moe(&l.moe, hin);
            x = inject(x, o, w.unwrap());
            if std::env::var("PYRE_QN_DEBUG").is_ok() { let v: Vec<f32> = x.clone().narrow(0, 0, 1).narrow(1, 0, 4).into_data().convert::<f32>().to_vec::<f32>().unwrap(); eprintln!("layer {li} x[0,:4] = {v:?}"); }
        }
        st.pos += t;
        let k = c.ngram_size - 1;
        let all: Vec<u32> = ple_ctx.iter().chain(ids.iter()).copied().collect();
        st.ple_ctx = all[all.len() - k..].to_vec();
        let (h, _) = self.mixer.fwd(x, c.hc, c.eps);
        self.lm_head.fwd(h)
    }

    pub fn logits_last(&self, logits: Tensor<2>) -> Vec<f32> {
        let [t, v] = logits.dims();
        logits.narrow(0, t - 1, 1).reshape([v]).into_data().convert::<f32>().to_vec::<f32>().expect("logits")
    }

    fn deltanet(&self, d: &DeltaNet, li: usize, x: Tensor<2>, st: &mut State) -> Tensor<2> {
        let c = &self.cfg;
        let t = x.dims()[0];
        let (kd, vd, hk, hv) = (c.lk_dim, c.lv_dim, c.lk_heads, c.lv_heads);
        let key_dim = kd * hk;
        let val_dim = vd * hv;
        let conv_dim = 2 * key_dim + val_dim;
        let qkv = d.qkv.fwd(x.clone());
        let z = d.z.fwd(x.clone());
        let b = d.b.fwd(x.clone());
        let a = d.a.fwd(x);
        let state = st.conv[li].take().unwrap_or_else(|| Tensor::zeros([c.conv_k - 1, conv_dim], TensorCreationOptions::new(self.device.clone()).with_dtype(DType::F32)));
        let (y, ns) = dconv(state, qkv, &d.conv_w, 1);
        st.conv[li] = Some(ns);
        let y = silu(y);
        let rep = hv / hk;
        let l2 = |t3: Tensor<3>| { let n = t3.clone().powi_scalar(2).sum_dim(2).add_scalar(1e-6).sqrt().recip(); t3.mul(n) };
        let expand = |t3: Tensor<3>| -> Tensor<3> { // [T, hk, kd] -> [T, hv, kd] (repeat_interleave)
            let t4: Tensor<4> = t3.reshape([t, hk, 1, kd]);
            t4.expand([t, hk, rep, kd]).reshape([t, hv, kd])
        };
        let q = expand(l2(y.clone().narrow(1, 0, key_dim).reshape([t, hk, kd]))).mul_scalar(1.0 / (kd as f64).sqrt());
        let k = expand(l2(y.clone().narrow(1, key_dim, key_dim).reshape([t, hk, kd])));
        let v: Tensor<3> = y.narrow(1, 2 * key_dim, val_dim).reshape([t, hv, vd]);
        let beta = sigmoid(b); // [T, hv]
        let sp = { let u = a.add(d.dt_bias.clone().unsqueeze::<2>()); softplus(u) };
        let g = sp.mul(d.neg_a.clone().unsqueeze::<2>()).exp(); // decay factor [T, hv]
        let mut s = st.rec[li].take().unwrap_or_else(|| Tensor::zeros([hv, kd, vd], TensorCreationOptions::new(self.device.clone()).with_dtype(DType::F32)));
        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            let qi: Tensor<3> = q.clone().narrow(0, i, 1).reshape([hv, kd, 1]);
            let ki: Tensor<3> = k.clone().narrow(0, i, 1).reshape([hv, kd, 1]);
            let vi: Tensor<3> = v.clone().narrow(0, i, 1).reshape([hv, 1, vd]);
            let gi: Tensor<3> = g.clone().narrow(0, i, 1).reshape([hv, 1, 1]);
            let bi: Tensor<3> = beta.clone().narrow(0, i, 1).reshape([hv, 1, 1]);
            s = s.mul(gi);
            let kv_mem = s.clone().mul(ki.clone()).sum_dim(1); // [hv,1,vd]
            let delta = vi.sub(kv_mem).mul(bi);
            s = s.add(ki.mul(delta));
            outs.push(s.clone().mul(qi).sum_dim(1).reshape([1, hv * vd]));
        }
        st.rec[li] = Some(s);
        let o: Tensor<2> = Tensor::cat(outs, 0);
        // gated RMSNorm per value head
        let o3: Tensor<3> = o.reshape([t, hv, vd]);
        let o3 = rms3(o3, c.eps).mul(d.norm_w.clone().unsqueeze::<3>());
        let z3: Tensor<3> = z.reshape([t, hv, vd]);
        let gz = if c.gate_sigmoid { sigmoid(z3) } else { silu(z3) };
        d.out.fwd(o3.mul(gz).reshape([t, val_dim]))
    }

    fn attn(&self, a: &Attn, li: usize, x: Tensor<2>, cos: &Tensor<2>, sin: &Tensor<2>, st: &mut State) -> Tensor<2> {
        let c = &self.cfg;
        let t = x.dims()[0];
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let qg: Tensor<3> = a.q.fwd(x.clone()).reshape([t, nh, 2 * hd]);
        let q = qg.clone().narrow(2, 0, hd);
        let gate: Tensor<2> = qg.narrow(2, hd, hd).reshape([t, nh * hd]);
        let q = a.qn.fwd(q.reshape([t * nh, hd]), c.eps).reshape([t, nh, hd]);
        let k = a.kn.fwd(a.k.fwd(x.clone()).reshape([t * nkv, hd]), c.eps).reshape([t, nkv, hd]);
        let v: Tensor<3> = a.v.fwd(x).reshape([t, nkv, hd]);
        let q = self.rope(q, cos, sin).swap_dims(0, 1); // [nh, t, hd]
        let k = self.rope(k, cos, sin).swap_dims(0, 1); // [nkv, t, hd]
        let v = v.swap_dims(0, 1);
        let k = match st.k[li].take() { Some(p) => Tensor::cat(vec![p, k], 1), None => k };
        let v = match st.v[li].take() { Some(p) => Tensor::cat(vec![p, v], 1), None => v };
        st.k[li] = Some(k.clone());
        st.v[li] = Some(v.clone());
        let tk = k.dims()[1];
        let g = nh / nkv;
        let kx: Tensor<3> = k.reshape([nkv, 1, tk, hd]).expand([nkv, g, tk, hd]).reshape([nh, tk, hd]);
        let vx: Tensor<3> = v.reshape([nkv, 1, tk, hd]).expand([nkv, g, tk, hd]).reshape([nh, tk, hd]);
        let mut mask = vec![0f32; t * tk];
        for i in 0..t { for j in 0..tk { if j > st.pos + i { mask[i * tk + j] = -1e30; } } }
        let m: Tensor<3> = self.f(mask, [t, tk]).unsqueeze::<3>();
        let sc = q.matmul(kx.swap_dims(1, 2)).mul_scalar(1.0 / (hd as f64).sqrt()).add(m);
        let o = softmax(sc, 2).matmul(vx).swap_dims(0, 1).reshape([t, nh * hd]);
        a.o.fwd(o.mul(sigmoid(gate)))
    }

    fn rope(&self, x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
        let [t, h, d] = x.dims();
        let r = self.cfg.rot_dim;
        let half = r / 2;
        let xr = x.clone().narrow(2, 0, r);
        let x1 = xr.clone().narrow(2, 0, half);
        let x2 = xr.clone().narrow(2, half, half);
        let rot = Tensor::cat(vec![x2.neg(), x1], 2);
        let c3: Tensor<3> = cos.clone().reshape([t, 1, r]);
        let s3: Tensor<3> = sin.clone().reshape([t, 1, r]);
        let y = xr.mul(c3).add(rot.mul(s3));
        if r == d { y } else { Tensor::cat(vec![y, x.narrow(2, r, d - r)], 2) }.reshape([t, h, d])
    }

    fn moe(&self, m: &Moe, x: Tensor<2>) -> Tensor<2> {
        let c = &self.cfg;
        let [t, h] = x.dims();
        let shared = m.sh_down.fwd(silu(m.sh_gate.fwd(x.clone())).mul(m.sh_up.fwd(x.clone())));
        let shared = shared.mul(sigmoid(m.sh_g.fwd(x.clone())));
        let probs = softmax(m.router.fwd(x.clone()), 1);
        let pv: Vec<f32> = probs.into_data().convert::<f32>().to_vec::<f32>().expect("router");
        let e = c.n_exp;
        let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; e];
        for r in 0..t {
            let p = &pv[r * e..(r + 1) * e];
            let mut ix: Vec<usize> = (0..e).collect();
            ix.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
            let top = &ix[..c.top_k];
            let s: f32 = if c.norm_topk { top.iter().map(|&i| p[i]).sum() } else { 1.0 };
            for &i in top { per[i].push((r, p[i] / s)); }
        }
        let xx = x.to_device(&self.xdev);
        let mut acc: Tensor<2> = Tensor::zeros([t, h], TensorCreationOptions::new(self.xdev.clone()).with_dtype(DType::F32));
        let xo = TensorCreationOptions::new(self.xdev.clone()).with_dtype(DType::F32);
        for (ei, rows) in per.iter().enumerate() {
            if rows.is_empty() { continue; }
            let n = rows.len();
            let ridx = Tensor::<1, Int>::from_data(TensorData::new(rows.iter().map(|r| r.0 as i32).collect::<Vec<_>>(), [n]), &self.xdev);
            let wts = Tensor::<2>::from_data(TensorData::new(rows.iter().map(|r| r.1).collect::<Vec<_>>(), [n, 1]), xo.clone());
            let xs = xx.clone().select(0, ridx.clone());
            let gu: Tensor<2> = m.gate_up.clone().narrow(0, ei, 1).reshape([h, 2 * c.moe_i]);
            let dn: Tensor<2> = m.down.clone().narrow(0, ei, 1).reshape([c.moe_i, h]);
            let y = xs.matmul(gu);
            let act = silu(y.clone().narrow(1, 0, c.moe_i)).mul(y.narrow(1, c.moe_i, c.moe_i));
            let o = act.matmul(dn).mul(wts);
            acc = acc.select_assign(0, ridx, o, burn::tensor::IndexingUpdateOp::Add);
        }
        acc.to_device(&self.device).add(shared)
    }

    fn ple(&self, p: &Ple, li: usize, x: Tensor<2>, ids: &[u32], ple_ctx: &[u32], st: &mut State) -> Tensor<2> {
        let c = &self.cfg;
        let t = ids.len();
        let h = c.hidden;
        let ctx = c.ngram_size - 1;
        let hist: Vec<i64> = ple_ctx.iter().chain(ids.iter()).map(|&v| v as i64).collect();
        let eos = c.eos as i64;
        let n = hist.len();
        // previous eos (exclusive) for each position
        let mut seg_start = vec![0i64; n];
        let mut last = -1i64;
        for i in 0..n { seg_start[i] = last + 1; if hist[i] == eos { last = i as i64; } }
        let shifted = |s: usize, i: usize| -> i64 {
            let src = i as i64 - s as i64;
            if s == 0 { return hist[i]; }
            if (i as i64 - seg_start[i]) >= s as i64 && src >= 0 { hist[src as usize] } else { eos }
        };
        let mut idv: Vec<i32> = vec![];
        for i in ctx..n {
            for ng in 2..=c.ngram_size {
                let mut mixed = shifted(0, i).wrapping_mul(p.mult[0]);
                for q in 1..ng { mixed ^= shifted(q, i).wrapping_mul(p.mult[q]); }
                for hh in 0..c.heads_per_ngram {
                    let hi = (ng - 2) * c.heads_per_ngram + hh;
                    idv.push((mixed.rem_euclid(p.sizes[hi]) + p.offsets[hi]) as i32);
                }
            }
        }
        let nh = idv.len() / t;
        let idx = Tensor::<1, Int>::from_data(TensorData::new(idv, [t * nh]), &self.xdev);
        let emb: Tensor<2> = p.emb.clone().select(0, idx).reshape([t, c.ple_dim]).to_device(&self.device);
        let key: Tensor<3> = p.nk.fwd(p.key.fwd(emb.clone()), c.eps).reshape([t, c.hc, h]);
        let value: Tensor<3> = p.value.fwd(emb).reshape([t, 1, h]);
        let qn: Tensor<3> = p.nq.fwd(x, c.eps).reshape([t, c.hc, h]);
        let gate = key.mul(qn).sum_dim(2).div_scalar((h as f64).sqrt()); // [t,hc,1]
        let gate = gate.clone().abs().clamp_min(1e-6).sqrt().mul(gate.sign());
        let gv: Tensor<2> = sigmoid(gate).mul(value).reshape([t, c.hc * h]);
        let gvn = p.nc.fwd(gv.clone(), c.eps);
        let k = p.conv_w.dims()[1];
        let slen = (k - 1) * c.ngram_size;
        let state = st.ple_conv[li].take().unwrap_or_else(|| Tensor::zeros([slen, c.hc * h], TensorCreationOptions::new(self.device.clone()).with_dtype(DType::F32)));
        let (y, ns) = dconv(state, gvn, &p.conv_w, c.ngram_size);
        st.ple_conv[li] = Some(ns);
        gv.add(silu(y))
    }
}

fn softplus(x: Tensor<2>) -> Tensor<2> {
    // log(1 + e^x), stable: max(x,0) + log1p(e^-|x|); torch uses threshold 20 -> x
    let pos = x.clone().clamp_min(0.0);
    pos.add(x.abs().neg().exp().log1p())
}
