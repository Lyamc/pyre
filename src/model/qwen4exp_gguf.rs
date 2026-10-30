//! Qwen3.8-Flash-Next (GGUF arch `qwen4exp`) on the CPU, straight from llama.cpp-style GGUF
//! quants (Q2_0, IQ2_XXS/IQ2_XS/IQ2_S/IQ1_M/IQ3_S experts, K/IQ4 dense, BF16 hyper-connections).
//!
//! The forward pass follows the GGUF tensor conventions of llama.cpp's
//! `src/models/qwen4exp.cpp` + `conversion/qwen4exp.py` (MIT): norm gammas are pre-folded to
//! (1+w) except the gated DeltaNet norm, `ssm_a` = -exp(A_log), DeltaNet V heads are stored in
//! tiled order (v head h reads k head h % n_k), q/gate are interleaved per attention head.
//! The architecture itself is transformers' `qwen4_exp` (Apache-2.0); the CPU-experts/mmapped
//! n-gram table layout follows Strata (MIT, github.com/Niko1221/Strata).
//!
//! Everything is computed on the host with rayon; weights stay quantized in the mmap. Prefill is
//! batched (every projection dequantizes each weight row once for all prompt tokens).
use super::gguf_q::{dot, Gguf, QMat};
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;

pub struct Cfg {
    pub n_layers: usize, pub d: usize, pub hc: usize, pub hc_lr: usize, pub n_head: usize, pub n_kv: usize, pub hd: usize,
    pub n_rot: usize, pub rope_base: f32, pub eps: f32, pub n_exp: usize, pub top_k: usize, pub ff_exp: usize,
    pub lin_nk: usize, pub lin_nv: usize, pub lin_dk: usize, pub lin_dv: usize, pub conv_k: usize, pub full_every: usize,
    pub ple_layers: Vec<usize>, pub ple_ngram: usize, pub ple_per_gram: usize, pub ple_conv_k: usize, pub ple_eos: u32, pub ple_dim: usize,
    pub ple_mult: Vec<u64>, pub ple_off: Vec<u64>, pub ple_vocab: Vec<u64>, pub vocab: usize, pub max_ctx: usize,
    /// QSA indexer: token budget (dense attention up to budget + ratio - 1 visible tokens), block size, heads, head dim; trained context
    pub idx_budget: usize, pub idx_ratio: usize, pub idx_nh: usize, pub idx_hd: usize, pub ctx_train: usize,
}

struct Hc { norm: Vec<f32>, down: QMat, up: QMat, inject: Option<QMat>, q8: Option<(super::qdot::Q8Mat, super::qdot::Q8Mat)> }
enum Mixer {
    Attn { q: QMat, k: QMat, v: QMat, o: QMat, qn: Vec<f32>, kn: Vec<f32>, idx: Option<(QMat, QMat, Vec<f32>, Vec<f32>)> },
    Lin { qkv: QMat, z: QMat, beta: QMat, alpha: QMat, a: Vec<f32>, dt: Vec<f32>, conv: Vec<f32>, norm: Vec<f32>, out: QMat },
}
struct Ple { key: QMat, value: QMat, nk: Vec<f32>, nq: Vec<f32>, nc: Vec<f32>, conv: Vec<f32> }
struct Layer { hc_a: Hc, hc_f: Hc, mix: Mixer, ple: Option<Ple>, router: QMat, gate_e: QMat, up_e: QMat, down_e: QMat, sh_gate_inp: Vec<f32>, sh_gate: QMat, sh_up: QMat, sh_down: QMat }

/// GPU copy of one attention layer's KV cache: `len` positions valid, room for `cap`.
pub struct KvG { k: burn_cubecl::cubecl::server::Handle, v: burn_cubecl::cubecl::server::Handle, ik: burn_cubecl::cubecl::server::Handle, cap: usize, len: usize }
/// Pre-packed host copies of missed experts in the GPU arena layout (Strata's host expert arena, expert_source.cpp:
/// experts kept in the device layout in RAM so a PCIe upload is a plain copy). Filled on first use up to
/// PYRE_QN_XARENA_GB, and never while less than PYRE_QN_RAM_RESERVE_GB (default 16) of RAM would stay free.
#[derive(Default)]
pub struct XPk { map: std::collections::HashMap<(usize, usize), super::qn_cube::Pk>, bytes: usize, full: bool, next_check: usize }
#[cfg(windows)]
fn avail_ram() -> usize { use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX}; let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() }; m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32; if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 { return 0; } m.ullAvailPhys as usize }
#[cfg(not(windows))]
fn avail_ram() -> usize { 0 }
/// The drafter's CPU matmuls on their own rayon pool (PYRE_MTP_THREADS, default 8; 0 = the global pool): ~20 vs
/// ~23.5 ms per 3-draft round against the 16-thread global pool. Only CPU work goes in here: a CubeCL call from a pool
/// thread would open another stream and memory pool on the GPU.
fn mtp_cpu<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static P: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    match P.get_or_init(|| { let n = env_usize("PYRE_MTP_THREADS").unwrap_or(8); if n == 0 { None } else { rayon::ThreadPoolBuilder::new().num_threads(n).build().ok() } }) { Some(p) => p.install(f), None => f() }
}
fn xarena_pin() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XARENA_PIN").map(|v| v == "1").unwrap_or(false)) }
fn xshare() -> f32 { static K: std::sync::OnceLock<f32> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XSHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)) }
pub struct XCache { tabs: super::qn_cube::XTabs, arena: Vec<burn_cubecl::cubecl::server::Handle>, ar_bytes: usize, per: usize, slot: usize, free: Vec<usize>, map: std::collections::HashMap<(usize, usize), super::qn_cube::XEnt>, cnt: Vec<Vec<u32>>, stream: Vec<usize>, used: usize, budget: usize, calls: usize, pub hits: u64, pub total: u64, scr: super::qn_cube::XScr }

/// Expert-ranking profile (PYRE_QN_XPROFILE=<file>): per-layer routing counts saved from earlier runs
/// (PYRE_QN_XPROFILE_SAVE=<file>; u32 n_layers, u32 n_exp, then n_layers*n_exp u32 counts, little endian).
/// Strata ships a similar profile; this one is pyre's own, built from our routing on our prompts. It only
/// changes which experts are cached (and so where they run), never which experts are used.
fn xprofile() -> &'static Option<Vec<Vec<f64>>> {
    static P: std::sync::OnceLock<Option<Vec<Vec<f64>>>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let pth = env_var("PYRE_QN_XPROFILE").ok()?; let b = std::fs::read(&pth).ok()?;
        let u = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        if b.len() < 8 { return None; } let (nl, ne) = (u(0) as usize, u(1) as usize); if b.len() != 8 + 4 * nl * ne { return None; }
        // scaled so the whole profile weighs as much as PYRE_QN_XPRIOR_TOK (default 256) tokens of top-k routing
        let tot: f64 = (0..nl * ne).map(|i| u(2 + i) as f64).sum::<f64>().max(1.0);
        let k = env_usize("PYRE_QN_XPRIOR_TOK").unwrap_or(256) as f64 * 10.0 * nl as f64 / tot;
        eprintln!("xcache: expert profile {pth} ({nl} layers x {ne} experts, {:.0} routings)", tot);
        Some((0..nl).map(|l| (0..ne).map(|e| u(2 + l * ne + e) as f64 * k).collect()).collect())
    })
}
/// Strata's adaptive tier (PYRE_QN_ADAPT=1, opt-in): see `Model::xadapt`.
fn adapt_on() -> bool { static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *A.get_or_init(|| env_var("PYRE_QN_ADAPT").map(|v| v == "1").unwrap_or(false)) }
/// tokens between adaptive rounds (PYRE_QN_ADAPT_EVERY, default 12: Strata's 4 verify rounds at ~3 tokens a round)
fn adapt_every() -> usize { env_usize("PYRE_QN_ADAPT_EVERY").unwrap_or(12).max(1) }
const XSCALE: u32 = 16;
/// one routing's count step: x16 with the adaptive tier (its 0.7 decay needs fractions), else 1
fn xinc() -> u32 { if adapt_on() { XSCALE } else { 1 } }
fn env_f32(k: &str) -> Option<f32> { env_var(k).ok().and_then(|v| v.parse().ok()) }
pub static ADAPT_SWAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Cache rank of an expert: routing count in this conversation plus the profile prior (x1000, integer
/// so ties break the same way every run).
fn xscore(xc: &XCache, il: usize, e: usize) -> u64 {
    let pr = xprofile().as_ref().and_then(|p| p.get(il)).and_then(|r| r.get(e)).copied().unwrap_or(0.0);
    xc.cnt[il][e] as u64 * 1000 + (pr * 1000.0) as u64
}
fn save_xprofile(pth: &str, cnt: &[Vec<u32>]) {
    let (nl, ne) = (cnt.len(), cnt.first().map_or(0, |r| r.len()));
    let mut b = Vec::with_capacity(8 + 4 * nl * ne); b.extend((nl as u32).to_le_bytes()); b.extend((ne as u32).to_le_bytes());
    for r in cnt { for &c in r { b.extend(c.to_le_bytes()); } }
    let _ = std::fs::write(pth, b);
}

/// Per-layer persistent buffers and the captured CUDA graph of a DeltaNet layer's dense part (PYRE_QN_GRAPH=1).
/// One captured verify-window graph of a DeltaNet layer's dense half (PYRE_QN_VGRAPH=1): its own buffers, kept for
/// the session, one per (layer, window size), so a replay never needs a recapture (Strata, MIT: per-window-size
/// verify graphs, src/core/verify.cpp). Recaptured only when the state buffers change (a new conversation).
pub struct VGraph { xn: H, da: H, mixed: H, proj: H, core: H, co: H, o: H, df: H, rd: H, g: Option<burn_cubecl::cubecl::client::Graph>, key: String, keep: Vec<H> }
static VGRAPH_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static VG_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VG_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn vgraph_on() -> bool { static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *G.get_or_init(|| env_var("PYRE_QN_VGRAPH").map(|v| v == "1").unwrap_or(false)) && !VGRAPH_OFF.load(std::sync::atomic::Ordering::Relaxed) }
pub struct LGraph { xn: H, da: H, mixed: H, proj: H, o: H, co: H, ob: H, df: H, rd: H, g: Option<burn_cubecl::cubecl::client::Graph>, key: String, keep: Vec<H> }
type H = burn_cubecl::cubecl::server::Handle;
static GRAPH_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn graph_on() -> bool { static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *G.get_or_init(|| env_var("PYRE_QN_GRAPH").map(|v| v == "1").unwrap_or(false)) && !GRAPH_OFF.load(std::sync::atomic::Ordering::Relaxed) }
pub struct Model { vg: std::sync::Mutex<std::collections::HashMap<(usize, usize), VGraph>>, vxs: std::sync::Mutex<std::collections::HashMap<usize, Vec<H>>>, mtp_headc: std::sync::OnceLock<Option<super::qn_cube::CMat>>, mtp_dense: std::sync::OnceLock<std::collections::HashMap<usize, super::qn_cube::CMat>>, xpk: std::sync::Mutex<XPk>, xper: std::sync::Mutex<Option<H>>, lg: std::sync::Mutex<std::collections::HashMap<usize, LGraph>>, headc: Option<super::qn_cube::CMat>, xc: Option<std::sync::Mutex<XCache>>, cube: Option<super::qn_cube::Cube>, cl: Vec<Option<super::qn_cube::CLayer>>, pub gpu: Option<super::qn_gpu::Gpu>, gmats: std::collections::HashMap<usize, super::qn_gpu::GMat>, gl: Vec<Option<super::qn_gpu::GLayer>>, pub cfg: Cfg, pub g: Gguf, embd: QMat, out: QMat, head: Hc, ple_tab: Option<QMat>, layers: Vec<Layer>, rope_inv: Vec<f32>, mtp: Option<super::mtp::Mtp> }

pub struct State {
    pub tokens: Vec<u32>,
    conv: Vec<Vec<f32>>,  // per linear layer [(conv_k-1) * channels], oldest first
    ssm: Vec<Vec<f32>>,   // per linear layer [nv][dk][dv]
    /// GPU copies of (conv, ssm) per linear layer while decode runs DeltaNet on the GPU; the host
    /// Vecs above are stale while `gdn_gpu` is set.
    gdn: Vec<(burn_cubecl::cubecl::server::Handle, burn_cubecl::cubecl::server::Handle)>,
    gdn_gpu: bool,
    kvg: Vec<Option<KvG>>,
    k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, // per attn layer [pos][n_kv*hd]
    /// QSA indexer per attn layer: raw index keys [pos][idx_hd], pooled+normed+roped block keys [block][idx_hd]
    ik: Vec<Vec<f32>>, ib: Vec<Vec<f32>>,
    ple_hist: Vec<Vec<f32>>, // per PLE layer: ring of past normalized values, [(conv_k-1)*dil][hc*d]
    /// speculative verify: DeltaNet (conv, ssm) after each of the first t-1 tokens of the last multi-token
    /// forward, per linear layer, on the GPU; PLE ring after each token. Used by `spec_keep` to roll back.
    ck: Vec<Vec<(burn_cubecl::cubecl::server::Handle, burn_cubecl::cubecl::server::Handle)>>,
    ple_ck: Vec<Vec<Vec<f32>>>,
    /// (layer, expert) picks of each token of the last `forward_multi`, so `spec_keep` can uncount rejected drafts
    spec_sel: Vec<Vec<(usize, usize)>>,
    /// set while a prompt runs through the verify path (PYRE_QN_PF_MULTI): no rollback checkpoints
    pf: bool,
    /// MTP drafter: main-model final residuals by position (recent window) and the drafter's own K/V
    mtp_r: std::collections::VecDeque<(usize, Vec<f32>)>, mtp_kv: super::mtp::Kv,
}

fn rms(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { y[i] = x[i] * s * w[i]; } }
fn rms_inplace(x: &mut [f32], w: &[f32], eps: f32) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { x[i] *= s * w[i]; } }
/// Submit queued GPU work before the CPU expert pass (WDDM otherwise holds it until the next read). PYRE_QN_KICK=1 on (no measurable effect on HYTE, so off).
/// One-arena low-overhead expert path (PYRE_QN_XFAST=0 off).
fn xfast() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XFAST").map(|v| v != "0").unwrap_or(true)) }
fn kick() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_KICK").map(|v| v != "0").unwrap_or(true)) }
fn attn_gpu(c: &Cfg) -> bool { c.hd % 32 == 0 && c.n_rot <= c.hd && c.n_head % c.n_kv == 0 && env_var("PYRE_QN_ATTN_GPU").map(|v| v != "0").unwrap_or(true) }
#[inline] fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline] fn silu(x: f32) -> f32 { x * sigm(x) }
#[inline] fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { x.exp().ln_1p() } }

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        anyhow::ensure!(g.arch() == "qwen4exp", "{}: arch '{}' is not qwen4exp", path.display(), g.arch());
        let a = "qwen4exp";
        let k = |s: &str| g.u(&format!("{a}.{s}"));
        let n_layers = k("block_count")? as usize;
        let ple_layers: Vec<usize> = g.arr_u64(&format!("{a}.ple.layers")).unwrap_or_default().into_iter().map(|v| v as usize).collect();
        let vocab = g.shape("token_embd.weight").context("token_embd")?[1];
        let cfg = Cfg {
            n_layers, d: k("embedding_length")? as usize, hc: k("hyper_connection.count")? as usize, hc_lr: k("hyper_connection.low_rank")? as usize,
            n_head: k("attention.head_count")? as usize, n_kv: k("attention.head_count_kv")? as usize, hd: k("attention.key_length")? as usize,
            n_rot: k("rope.dimension_count")? as usize, rope_base: g.f(&format!("{a}.rope.freq_base"))? as f32, eps: g.f(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f32,
            n_exp: k("expert_count")? as usize, top_k: k("expert_used_count")? as usize, ff_exp: k("expert_feed_forward_length")? as usize,
            lin_nk: k("ssm.group_count")? as usize, lin_nv: k("ssm.time_step_rank")? as usize, lin_dk: k("ssm.state_size")? as usize, lin_dv: k("ssm.state_size")? as usize,
            conv_k: k("ssm.conv_kernel")? as usize, full_every: k("full_attention_interval").unwrap_or(4) as usize,
            ple_ngram: k("ple.ngram_size").unwrap_or(0) as usize, ple_per_gram: k("ple.heads_per_ngram").unwrap_or(0) as usize, ple_conv_k: k("ple.conv_kernel").unwrap_or(0) as usize,
            ple_eos: k("ple.eos_token_id").unwrap_or(0) as u32, ple_dim: k("embedding_length_per_layer_input").unwrap_or(0) as usize,
            ple_mult: g.arr_u64(&format!("{a}.ple.layer_multipliers")).unwrap_or_default(), ple_off: g.arr_u64(&format!("{a}.ple.head_offsets")).unwrap_or_default(),
            ple_vocab: g.arr_u64(&format!("{a}.ple.head_vocab_sizes")).unwrap_or_default(), ple_layers, vocab,
            // the QSA sparse indexer only kicks in beyond top_k visible tokens; below that attention is dense
            max_ctx: 0,
            idx_budget: k("attention.indexer.top_k").unwrap_or(2048) as usize, idx_nh: k("attention.indexer.head_count").unwrap_or(0) as usize,
            idx_hd: k("attention.indexer.key_length").unwrap_or(128) as usize,
            idx_ratio: g.arr_u64(&format!("{a}.attention.compress_ratios")).unwrap_or_default().into_iter().max().unwrap_or(4).max(1) as usize,
            ctx_train: k("context_length").unwrap_or(262144) as usize,
        };
        let mut cfg = cfg;
        let has_idx = cfg.idx_nh > 0 && g.has("blk.3.indexer.q_proj.weight");
        // 128k by default (PYRE_QN_CTX overrides); without indexer weights only the dense range is exact
        cfg.max_ctx = if has_idx { env_usize("PYRE_QN_CTX").unwrap_or(131072).min(cfg.ctx_train).max(16) } else { cfg.idx_budget };
        let hc = |p: &str, inj: bool| -> Result<Hc> { Ok(Hc { norm: g.vec(&format!("{p}_norm.weight"))?, down: g.mat(&format!("{p}_down.weight"))?, up: g.mat(&format!("{p}_up.weight"))?, inject: if inj { Some(g.mat(&format!("{p}_inject.weight"))?) } else { None }, q8: None }) };
        let mut layers = vec![];
        for il in 0..n_layers {
            let p = format!("blk.{il}");
            let m = |s: &str| g.mat(&format!("{p}.{s}"));
            let v = |s: &str| g.vec(&format!("{p}.{s}"));
            let full = (il + 1) % cfg.full_every == 0;
            let mix = if full {
                Mixer::Attn { q: m("attn_q.weight")?, k: m("attn_k.weight")?, v: m("attn_v.weight")?, o: m("attn_output.weight")?, qn: v("attn_q_norm.weight")?, kn: v("attn_k_norm.weight")?,
                    idx: if has_idx { Some((m("indexer.q_proj.weight")?, m("indexer.k_proj.weight")?, v("indexer.q_norm.weight")?, v("indexer.k_norm.weight")?)) } else { None } }
            } else {
                Mixer::Lin { qkv: m("attn_qkv.weight")?, z: m("attn_gate.weight")?, beta: m("ssm_beta.weight")?, alpha: m("ssm_alpha.weight")?, a: v("ssm_a")?, dt: v("ssm_dt.bias")?, conv: v("ssm_conv1d.weight")?, norm: v("ssm_norm.weight")?, out: m("ssm_out.weight")? }
            };
            let ple = if cfg.ple_layers.contains(&il) { Some(Ple { key: m("ple_key.weight")?, value: m("ple_value.weight")?, nk: v("ple_norm_key.weight")?, nq: v("ple_norm_query.weight")?, nc: v("ple_norm_conv.weight")?, conv: v("ple_conv1d.weight")? }) } else { None };
            layers.push(Layer { hc_a: hc(&format!("{p}.hc_attn"), true)?, hc_f: hc(&format!("{p}.hc_ffn"), true)?, mix, ple,
                router: m("ffn_gate_inp.weight")?, gate_e: m("ffn_gate_exps.weight")?, up_e: m("ffn_up_exps.weight")?, down_e: m("ffn_down_exps.weight")?,
                sh_gate_inp: v("ffn_gate_inp_shexp.weight")?, sh_gate: m("ffn_gate_shexp.weight")?, sh_up: m("ffn_up_shexp.weight")?, sh_down: m("ffn_down_shexp.weight")? });
        }
        let head = hc("output_hc", false)?;
        let out = if g.has("output.weight") { g.mat("output.weight")? } else { g.mat("token_embd.weight")? };
        let ple_tab = if cfg.ple_layers.is_empty() { None } else { Some(g.mat("per_layer_token_embd.weight")?) };
        let half = cfg.n_rot / 2;
        let rope_inv = (0..half).map(|i| cfg.rope_base.powf(-(2.0 * i as f32) / cfg.n_rot as f32)).collect();
        let mut layers = layers; let mut head = head;
        if env_var("PYRE_QN_HC_Q8").map(|v| v == "1").unwrap_or(false) {
            let mk = |h: &mut Hc| { let mut a = vec![&h.down]; if let Some(i) = &h.inject { a.push(i); } let di = super::qdot::Q8Mat::from_rows(&a); let up = super::qdot::Q8Mat::from_rows(&[&h.up]); h.q8 = Some((di, up)); };
            for l in layers.iter_mut() { mk(&mut l.hc_a); mk(&mut l.hc_f); } mk(&mut head);
        }
        Ok(Model { vg: Default::default(), vxs: Default::default(), mtp_headc: Default::default(), mtp_dense: Default::default(), xpk: Default::default(), xper: Default::default(), lg: Default::default(), headc: None, xc: None, cube: None, cl: vec![], gpu: None, gmats: Default::default(), gl: vec![], embd: g.mat("token_embd.weight")?, out, head, ple_tab, layers, rope_inv, mtp: None, cfg, g })
    }

    pub fn max_ctx(&self) -> usize { self.cfg.max_ctx }

    /// Bring every routed-expert page into this process's working set once, up front (the idea of
    /// Strata's resident expert arena, without a copy: the pages stay the OS file cache's). Without
    /// it, each decode step soft-faults on ~10k first-touched 4 KiB pages. Skipped when free RAM is
    /// short (keeps 16 GiB spare); `PYRE_QN_PREFAULT=0` turns it off. Returns GiB touched.
    pub fn prefault_experts(&self) -> f64 {
        if env_var("PYRE_QN_PREFAULT").map(|v| v == "0").unwrap_or(false) { return 0.0; }
        let mats: Vec<&QMat> = self.layers.iter().flat_map(|l| [&l.gate_e, &l.up_e, &l.down_e]).collect();
        let total: usize = mats.iter().map(|m| m.bytes().len()).sum();
        let mut sys = sysinfo::System::new(); sys.refresh_memory();
        if (sys.available_memory() as i64) < total as i64 + (16i64 << 30) {
            tracing::warn!("qwen4exp: {:.1} GiB of experts not prefaulted: only {:.1} GiB RAM available", total as f64 / (1u64 << 30) as f64, sys.available_memory() as f64 / (1u64 << 30) as f64);
            return 0.0;
        }
        let sum: u64 = mats.par_iter().map(|m| m.bytes().par_chunks(1 << 22).map(|c| { let mut a = 0u64; let mut i = 0; while i < c.len() { a = a.wrapping_add(unsafe { std::ptr::read_volatile(c.as_ptr().add(i)) } as u64); i += 4096; } a }).sum::<u64>()).sum();
        std::hint::black_box(sum);
        total as f64 / (1u64 << 30) as f64
    }

    /// Strata's split: each layer's dense half (HC mixers, mixer projections, router, shared expert)
    /// resident on the GPU as Q8, as far as VRAM above pyre's reserve allows; routed experts stay on the CPU.
    pub fn attach_gpu(&mut self, g: super::qn_gpu::Gpu) -> Result<(usize, usize, f64)> {
        use super::qn_gpu::{GHc, GLayer};
        let (d, hc) = (self.cfg.d, self.cfg.hc);
        let ghc = |h: &Hc| -> Result<Option<GHc>> {
            let mut a = vec![&h.down]; if let Some(i) = &h.inject { a.push(i); }
            let (Some(di), Some(up)) = (g.upload(&a)?, g.upload(&[&h.up])?) else { return Ok(None) };
            Ok(Some(GHc { norm: g.up3(&h.norm, [1, hc, d]), di, up, inj: h.inject.is_some() }))
        };
        crate::limits::vlog("burn gpu up"); let (mut on, mut off) = (0, 0); let b0 = g.budget_left();
        let mut gl = vec![];
        if env_var("PYRE_QN_CUBE").map(|v| v != "0").unwrap_or(true) {
            if let Ok(cb) = super::qn_cube::Cube::new() {
                use super::qn_cube::{CHc, CLayer};
                let cb0 = cb.budget.load(std::sync::atomic::Ordering::Relaxed);
                let chc = |h: &Hc| -> Option<CHc> { let mut a = vec![&h.down]; if let Some(i) = &h.inject { a.push(i); } Some(CHc { norm: cb.buf(&h.norm), di: cb.upload(&a, &[])?, up: cb.upload(&[&h.up], &[])? }) };
                let mut cl = vec![];
                for l in &self.layers {
                    let min = match &l.mix { Mixer::Attn { q, k, v, idx, .. } => { let mut a = vec![q, k, v]; if let Some((iq, ik, _, _)) = idx { a.push(iq); a.push(ik); } a }, Mixer::Lin { qkv, z, beta, alpha, .. } => vec![qkv, z, beta, alpha] };
                    let mout = match &l.mix { Mixer::Attn { o, .. } => o, Mixer::Lin { out, .. } => out };
                    let gdn = match &l.mix { Mixer::Lin { conv, a, dt, norm, .. } if self.cfg.lin_dk == self.cfg.lin_dv && self.cfg.lin_dk % 32 == 0 =>
                        Some(super::qn_cube::GdnW { conv: cb.buf(conv), a: cb.buf(a), dt: cb.buf(dt), norm: cb.buf(norm) }), _ => None };
                    let an = match &l.mix { Mixer::Attn { qn, kn, .. } => Some((cb.buf(qn), cb.buf(kn))), _ => None };
                    let c = (|| Some(CLayer { an, gdn, hca: chc(&l.hc_a)?, min: cb.upload_mix(&min, &[])?, mout: cb.upload_mix(&[mout], &[])?, hcf: chc(&l.hc_f)?,
                        fin: cb.upload_mix(&[&l.router, &l.sh_gate, &l.sh_up], &[&l.sh_gate_inp])?, sdown: cb.upload_mix(&[&l.sh_down], &[])? }))();
                    if c.is_some() { on += 1 } else { off += 1 }
                    cl.push(c); gl.push(None);
                }
                crate::limits::vlog("cube dense layers"); let used = cb0 - cb.budget.load(std::sync::atomic::Ordering::Relaxed);
                g.budget.fetch_sub(used, std::sync::atomic::Ordering::Relaxed);
                if env_var("PYRE_QN_TYPES").is_ok() { for (i, l) in self.layers.iter().enumerate().take(4) {
                    let mut v: Vec<(&str, &QMat)> = vec![("hca.down", &l.hc_a.down), ("hca.up", &l.hc_a.up), ("router", &l.router), ("sh_gate", &l.sh_gate), ("sh_up", &l.sh_up), ("sh_down", &l.sh_down)];
                    match &l.mix { Mixer::Attn { q, k, v: vv, o, .. } => { v.push(("q", q)); v.push(("k", k)); v.push(("v", vv)); v.push(("o", o)); }, Mixer::Lin { qkv, z, beta, alpha, out, .. } => { v.push(("qkv", qkv)); v.push(("z", z)); v.push(("beta", beta)); v.push(("alpha", alpha)); v.push(("out", out)); } }
                    eprintln!("types L{i}: {}", v.iter().map(|(n, m)| format!("{n}={:?}[{}x{}]", m.ty, m.rows, m.cols)).collect::<Vec<_>>().join(" "));
                } eprintln!("types out={:?} embd={:?}", self.out.ty, self.embd.ty); }
                if env_var("PYRE_QN_GPUBENCH").is_ok() { if let Some(Some(c0)) = cl.iter().find(|x| x.is_some()) {
                    let x = cb.buf(&vec![0.01f32; hc * d]); let xn = cb.empty(hc * d); let o = cb.empty(20000); let dn = cb.buf(&vec![0.01f32; 400]);
                    let bench = |name: &str, f: &dyn Fn()| { f(); cb.read(&o); let t0 = std::time::Instant::now(); for _ in 0..50 { f(); } cb.read(&o); eprintln!("cubebench {name}: {:.1} us", t0.elapsed().as_secs_f64() * 1e6 / 50.0); };
                    bench("nothing", &|| {}); let v40 = vec![0f32; 40]; bench("upload 40 floats", &|| { let _ = cb.buf(&v40); }); let pw = cb.buf(&v40); bench("write + rms", &|| { cb.client.write(&pw, burn_cubecl::cubecl::bytes::Bytes::from_elems(v40.clone())); cb.rms(&x, &c0.hca.norm, &xn, hc, d, 1e-6); }); bench("upload + rms", &|| { let _ = cb.buf(&v40); cb.rms(&x, &c0.hca.norm, &xn, hc, d, 1e-6); });
                    bench(&format!("gemv min {}x{} ({:.1} MB)", c0.min.rows, c0.min.cols, (c0.min.rows * c0.min.cols) as f64 / 1e6), &|| cb.gemv(&c0.min, &x, hc * d, 0, &o, 20000, 0));
                    bench(&format!("gemv hc down {}x{}", c0.hca.di.rows, c0.hca.di.cols), &|| cb.gemv(&c0.hca.di, &x, hc * d, 0, &o, 20000, 0));
                    bench(&format!("gemv fin {}x{}", c0.fin.rows, c0.fin.cols), &|| cb.gemv(&c0.fin, &x, hc * d, 0, &o, 20000, 0));
                    bench(&format!("gemv mout {}x{}", c0.mout.rows, c0.mout.cols), &|| cb.gemv(&c0.mout, &x, hc * d, 0, &o, 20000, 0));
                    bench("rms", &|| cb.rms(&x, &c0.hca.norm, &xn, hc, d, 1e-6));
                    { let x3 = cb.buf(&vec![0.01f32; 3 * hc * d]); let o3 = cb.empty(3 * 20000);
                      for (nm, m) in [("min", &c0.min), ("hca.di", &c0.hca.di), ("hca.up", &c0.hca.up), ("mout", &c0.mout), ("fin", &c0.fin), ("sdown", &c0.sdown)] {
                        let gb = (m.rows * m.cols) as f64 * 9.0 / 8.0 / 1e9;
                        cb.gemv_m(m, &x3, 3 * hc * d, 0, m.cols, &o3, 3 * 20000, 0, m.rows.min(20000), 3); cb.read(&o3); let t1 = std::time::Instant::now(); for _ in 0..50 { cb.gemv_m(m, &x3, 3 * hc * d, 0, m.cols, &o3, 3 * 20000, 0, m.rows.min(20000), 3); } cb.read(&o3); let us = t1.elapsed().as_secs_f64() * 1e6 / 50.0;
                        eprintln!("cubebench gemv_m T=3 {nm} {}x{}: {us:.1} us ({:.0} GB/s)", m.rows, m.cols, gb / (us * 1e-6)); } }
                    bench("gate", &|| cb.gate(&x, &dn, &c0.hca.up, &o, 20000, 0, hc, d));
                    bench("comb", &|| cb.comb(&xn, &o, None, None, &dn, 320, hc, d));
                } }
                self.cl = cl;
                // LM head on the GPU (opt-in PYRE_QN_GPUHEAD=1: only ~+0.3 tok/s, and as Q8 it costs VRAM
                if env_var("PYRE_QN_GPUHEAD").map(|v| v == "1").unwrap_or(false) { self.headc = cb.upload_mix(&[&self.out], &[]); if self.headc.is_some() { eprintln!("gpu: lm head on gpu"); } }
                // routed-expert cache (default on, PYRE_QN_XCACHE_GB=<GiB>, 0 = off): up to that much of the VRAM left above the reserve
                let left = g.budget_left().max(0) as usize;
                let cap = env_var("PYRE_QN_XCACHE_GB").ok().and_then(|v| v.parse::<f64>().ok()).map(|v| (v * (1u64 << 30) as f64) as usize).unwrap_or(5usize << 29); // default on, 2.5 GiB: on the RTX 3060 (quiet GPU) 2 GiB gives ~17 vs ~14 tok/s cache off; bigger budgets push VRAM past the point where WDDM pages and get slower. 0 = off
                let budget = left.saturating_sub(env_usize("PYRE_QN_XHEAD_MB").unwrap_or(2048) << 20).min(cap); // 2 GiB more headroom: past that, WDDM starts paging
                if budget > 0 { g.budget.fetch_sub(budget as i64, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("gpu: expert cache budget {:.2} GiB", budget as f64 / (1u64 << 30) as f64);
                    let slot = self.layers.iter().filter_map(|l| Some((super::qn_cube::xtype(l.gate_e.ty)?, super::qn_cube::xtype(l.up_e.ty)?, super::qn_cube::xtype(l.down_e.ty)?))).map(|t| { let w = super::qn_cube::xwords(t, self.cfg.d, self.cfg.ff_exp); 4 * (w.0 + w.1 + w.2) }).max().unwrap_or(4usize).next_multiple_of(256);
                    let nslot = budget / slot; let maxper = (env_usize("PYRE_QN_ARENA_MB").unwrap_or(2560) << 20) / slot; let narena = nslot.div_ceil(maxper.max(1)).max(1); let per = nslot.div_ceil(narena).max(1); let ar_bytes = per * slot; // even arenas, each <= 2.5 GiB (one buffer > ~3.1 GB fails to allocate)
                    self.xc = Some(std::sync::Mutex::new(XCache { tabs: cb.xtabs(), arena: (0..narena).map(|_| cb.xarena(ar_bytes)).collect(), ar_bytes, per, slot, free: (0..nslot).rev().collect(), map: Default::default(), cnt: vec![vec![0; self.cfg.n_exp]; self.layers.len()], stream: vec![], used: 0, budget, calls: 0, hits: 0, total: 0, scr: cb.xscratch(self.cfg.d, self.cfg.ff_exp, self.cfg.top_k) })); self.xpk_prefill(nslot); }
                // PYRE_QN_XSTREAM=K: K cache slots kept free for streaming missed experts to the GPU per layer
                if let Some(xm) = &self.xc { let mut x = xm.lock().unwrap(); let k = xstream().min(x.free.len()); for _ in 0..k { let sl = x.free.remove(0); x.stream.push(sl); } x.budget = x.budget.saturating_sub(k * x.slot); }
                { let ns = super::qn_cube::NATIVE_STAT.load(std::sync::atomic::Ordering::Relaxed); if ns > 0 { eprintln!("gpu: dense native IQ4_XS {:.0} M weights, Q8_0 {:.0} M (mixed mats)", (ns & ((1u64 << 40) - 1)) as f64 / 1e6, (ns >> 40) as f64 / 1e6); } }
                crate::limits::vlog("expert cache"); self.cube = Some(cb);
            }
        }
        if self.cube.is_none() { for l in &self.layers {
            let min = match &l.mix { Mixer::Attn { q, k, v, idx, .. } => { let mut a = vec![q, k, v]; if let Some((iq, ik, _, _)) = idx { a.push(iq); a.push(ik); } a }, Mixer::Lin { qkv, z, beta, alpha, .. } => vec![qkv, z, beta, alpha] };
            let mout = match &l.mix { Mixer::Attn { o, .. } => o, Mixer::Lin { out, .. } => out };
            let parts = (|| -> Result<Option<GLayer>> { Ok(Some(GLayer {
                hca: match ghc(&l.hc_a)? { Some(x) => x, None => return Ok(None) },
                min: match g.upload(&min)? { Some(x) => x, None => return Ok(None) },
                mout: match g.upload(&[mout])? { Some(x) => x, None => return Ok(None) },
                hcf: match ghc(&l.hc_f)? { Some(x) => x, None => return Ok(None) },
                fin: match g.upload(&[&l.router, &l.sh_gate, &l.sh_up])? { Some(x) => x, None => return Ok(None) },
                sdown: match g.upload(&[&l.sh_down])? { Some(x) => x, None => return Ok(None) },
                sgw: g.up(&l.sh_gate_inp, [d, 1]),
            })) })()?;
            if parts.is_some() { on += 1 } else { off += 1 }
            gl.push(parts);
        } }
        let mut m = std::collections::HashMap::new();
        let mut extra: Vec<Vec<&QMat>> = vec![];
        for l in &self.layers { if let Some(p) = &l.ple { extra.push(vec![&p.key, &p.value]); } }
        extra.push(vec![&self.head.down]); extra.push(vec![&self.head.up]); if self.headc.is_none() { extra.push(vec![&self.out]); } // the cube head (PYRE_QN_GPUHEAD=1) makes the burn copy dead weight
        for grp in extra { if let Some(gm) = g.upload(&grp)? { m.insert(grp[0].key(), gm); } }
        crate::limits::vlog("burn extras (ple, head hc, lm head)"); let gb = (b0 - g.budget_left()) as f64 / (1u64 << 30) as f64;
        if env_var("PYRE_QN_GPUBENCH").is_ok() { if let Some(Some(l0)) = gl.iter().find(|x| x.is_some()) {
            let x1 = vec![0.01f32; d]; let xh = vec![0.01f32; hc * d];
            let b = |name: &str, f: &dyn Fn() -> usize| { for _ in 0..3 { f(); } let t0 = std::time::Instant::now(); let mut n = 0; for _ in 0..20 { n = f(); } eprintln!("gpubench {name}: {:.3} ms ({n} out)", t0.elapsed().as_secs_f64() * 50.0); };
            b("readback only (upload+read 2560)", &|| g.read(g.up(&x1, [1, d])).len());
            b(&format!("min matmul {}x{} Q8 + read", l0.min.cols, l0.min.rows), &|| g.read(l0.min.fwd(g.up(&x1, [1, d]))).len());
            b(&format!("hc up {}x{} + read", l0.hca.up.cols, l0.hca.up.rows), &|| g.read(l0.hca.up.fwd(g.up(&vec![0.01f32; l0.hca.up.cols], [1, l0.hca.up.cols]))).len());
            b("hc mix (no read of x) + read mixed", &|| g.read(l0.hca.mix(g.up(&xh, [1, hc * d]), 1, hc, d, self.cfg.eps).0).len());
            { let mut te = 0f64; let mut tt = 0f64; for _ in 0..20 { let t0 = std::time::Instant::now(); let (m_, _) = l0.hca.mix(g.up(&xh, [1, hc * d]), 1, hc, d, self.cfg.eps); let m2 = l0.hcf.mix(g.up(&xh, [1, hc * d]), 1, hc, d, self.cfg.eps).0; let s_ = m_ + m2; te += t0.elapsed().as_secs_f64(); let _ = g.read(s_); tt += t0.elapsed().as_secs_f64(); }
              eprintln!("gpubench 2 hc mixes: enqueue {:.3} ms, total {:.3} ms", te * 50.0, tt * 50.0); }
            b(&format!("hc down {}x{} + read", l0.hca.di.cols, l0.hca.di.rows), &|| g.read(l0.hca.di.fwd(g.up(&xh, [1, hc * d]))).len());
            b("rms only + read", &|| { let x3 = g.up(&xh, [1, hc * d]).reshape([1, hc, d]); let ms = x3.clone().powi_scalar(2).mean_dim(2); let xn = x3 * ms.add_scalar(1e-6).sqrt().recip() * l0.hca.norm.clone(); g.read(xn.reshape([1, hc * d])).len() });
            b("sigmoid-mean only + read", &|| { let x3 = g.up(&xh, [1, hc * d]).reshape([1, hc, d]); g.read((x3.clone() * burn::tensor::activation::sigmoid(x3)).mean_dim(1).reshape([1, d])).len() });
            b("10x min matmul then read", &|| { let mut y = g.up(&x1, [1, d]); let mut acc = None; for _ in 0..10 { let o = l0.min.fwd(y.clone()); y = o.clone().slice([0..1, 0..d]); acc = Some(o); } g.read(acc.unwrap()).len() });
        } }
        self.gmats = m; self.gl = gl; self.gpu = Some(g);
        Ok((on, off, gb))
    }

    /// The layer with its dense half on the GPU; the residual `xd` [t, hc*d] stays on the device.
    /// Two device->host copies: the mixer inputs, then the MoE input + router logits. The shared expert
    /// is queued after the second copy so the GPU computes it while the CPU runs the routed experts.
    fn layer_gpu(&self, l: &Layer, gl: &super::qn_gpu::GLayer, xd: burn::tensor::Tensor<2>, st: &mut State, ai: &mut usize, li: &mut usize, start: usize, t: usize) -> burn::tensor::Tensor<2> {
        use burn::tensor::activation::{sigmoid, silu};
        let g = self.gpu.as_ref().unwrap(); let c = &self.cfg; let (d, hc, ne) = (c.d, c.hc, c.n_exp);
        let (mixed, inj_a) = gl.hca.mix(xd.clone(), t, hc, d, c.eps);
        let mi = g.read(gl.min.fwd(mixed));
        let proj = gl.min.split_host(&mi, t);
        let core = match l.mix { Mixer::Attn { .. } => { let o = self.attn_core(l, *ai, st, proj, start, t); *ai += 1; o }, Mixer::Lin { .. } => { let o = self.linear_core(l, *li, st, proj, t); *li += 1; o } };
        let o = gl.mout.fwd(g.up(&core, [t, core.len() / t]));
        let x2 = super::qn_gpu::combine(xd, o, inj_a.unwrap(), t, hc, d);
        let (mf, inj_f) = gl.hcf.mix(x2.clone(), t, hc, d, c.eps);
        let fi = gl.fin.fwd(mf.clone()); let sf = l.sh_gate.rows;
        let both = g.read(burn::tensor::Tensor::cat(vec![mf.clone(), fi.clone().slice([0..t, 0..ne])], 1));
        let sh = gl.sdown.fwd(silu(fi.clone().slice([0..t, ne..ne + sf])) * fi.slice([0..t, ne + sf..ne + 2 * sf]))
            * sigmoid(mf.matmul(gl.sgw.clone()));
        let w = d + ne; let (mut mfc, mut lgc) = (Vec::with_capacity(t * d), Vec::with_capacity(t * ne));
        for ti in 0..t { mfc.extend_from_slice(&both[ti * w..ti * w + d]); lgc.extend_from_slice(&both[ti * w + d..(ti + 1) * w]); }
        let mut yr = vec![0f32; t * d];
        self.routed(l, &mfc, &lgc, t, &mut yr, false);
        super::qn_gpu::combine(x2, sh + g.up(&yr, [t, d]), inj_f.unwrap(), t, hc, d)
    }

    /// One decode token through a layer with the dense half in pyre's CubeCL kernels (x stays on the GPU).
    fn layer_cube(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, x: &burn_cubecl::cubecl::server::Handle, st: &mut State, ai: &mut usize, li: &mut usize, start: usize) {
        let cb = self.cube.as_ref().unwrap(); let c = &self.cfg; let (d, hc, ne) = (c.d, c.hc, c.n_exp); let w = hc * d;
        let q0 = std::time::Instant::now();
        let gfast = graph_on() && st.gdn_gpu && cl.gdn.is_some() && matches!(super::qn_cube::native_mode(), 0 | 1 | 3); // mode 2 allocates scratch per GEMV
        let (rd, df, rn, lrf, q1, q2, q3) = if gfast { self.gdn_graph(il, cl, x, st, li) } else {
        let xn = cb.empty(w);
        let lra = cl.hca.up.cols; let da = cb.empty(cl.hca.di.rows);
        cb.rms(x, &cl.hca.norm, &xn, hc, d, c.eps);
        cb.gemv(&cl.hca.di, &xn, w, 0, &da, cl.hca.di.rows, 0);
        let mixed = cb.empty(d);
        cb.gate(&xn, &da, &cl.hca.up, &mixed, d, 0, hc, d);
        let proj = cb.empty(cl.min.rows);
        cb.gemv(&cl.min, &mixed, d, 0, &proj, cl.min.rows, 0);
        let q1 = std::time::Instant::now();
        let o = cb.empty(d);
        let (q2, q3);
        if let (Some(gw), true) = (&cl.gdn, st.gdn_gpu) {
            let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let ch = cl.min.splits[1]; let sp = &cl.min.splits;
            let (cs, ss) = &st.gdn[*li]; *li += 1;
            let co = cb.empty(ch); let ob = cb.empty(nv * dk);
            cb.conv(&proj, cl.min.rows, &gw.conv, cs, &co, ch, c.conv_k);
            cb.delta(&co, ch, &proj, cl.min.rows, gw, ss, &ob, nk, nv, dk, sp[1], sp[2], sp[3], c.eps);
            cb.gemv(&cl.mout, &ob, nv * dk, 0, &o, d, 0);
            q2 = std::time::Instant::now(); q3 = q2;
        } else if let (Mixer::Attn { idx, .. }, Some((qnh, knh)), true) = (&l.mix, &cl.an, attn_gpu(c) && start + 1 <= self.dense_n()) {
            let (nh, nkv, hd, half) = (c.n_head, c.n_kv, c.hd, c.n_rot / 2); let pos = start; let sp = &cl.min.splits;
            self.kv_push(cb, *ai, st, pos + 1);
            let g = st.kvg[*ai].as_mut().unwrap(); assert_eq!(g.len, pos, "gpu kv out of step");
            let cs: Vec<f32> = (0..half).map(|i| (pos as f32 * self.rope_inv[i]).cos()).chain((0..half).map(|i| (pos as f32 * self.rope_inv[i]).sin())).collect();
            let csb = cb.buf(&cs); let att = cb.empty(nh * hd);
            cb.attn(&proj, cl.min.rows, sp[1], sp[2], qnh, knh, &csb, &g.k, &g.v, g.cap, &att, nh, nkv, hd, half, pos, c.eps);
            if idx.is_some() && sp.len() > 5 { let ihd = c.idx_hd; cb.copy(&proj, cl.min.rows, sp[4], &g.ik, g.cap * ihd, pos * ihd, ihd); }
            g.len = pos + 1; *ai += 1;
            cb.gemv(&cl.mout, &att, nh * hd, 0, &o, d, 0);
            q2 = std::time::Instant::now(); q3 = q2;
        } else {
            let pv = cb.read(&proj);
            q2 = std::time::Instant::now();
            let parts: Vec<Vec<f32>> = (0..cl.min.splits.len() - 1).map(|i| pv[cl.min.splits[i]..cl.min.splits[i + 1]].to_vec()).collect();
            let core = match l.mix { Mixer::Attn { .. } => { let o = self.attn_core(l, *ai, st, parts, start, 1); *ai += 1; o }, Mixer::Lin { .. } => { let o = self.linear_core(l, *li, st, parts, 1); *li += 1; o } };
            q3 = std::time::Instant::now();
            let cbuf = cb.buf(&core);
            cb.gemv(&cl.mout, &cbuf, core.len(), 0, &o, d, 0);
        }
        cb.comb(x, &o, None, None, &da, lra, hc, d);
        let lrf = cl.hcf.up.cols; let df = cb.empty(cl.hcf.di.rows);
        cb.rms(x, &cl.hcf.norm, &xn, hc, d, c.eps);
        cb.gemv(&cl.hcf.di, &xn, w, 0, &df, cl.hcf.di.rows, 0);
        let rn = d + cl.fin.rows; let rd = cb.empty(rn);
        cb.gate(&xn, &df, &cl.hcf.up, &rd, rn, 0, hc, d);
        cb.gemv(&cl.fin, &rd, rn, 0, &rd, rn, d);
        (rd, df, rn, lrf, q1, q2, q3) };
        let q4 = std::time::Instant::now();
        if prewake() { qn_pool().prewake(); }
        let rv = cb.read(&rd);
        let q5 = std::time::Instant::now();
        let sf = l.sh_gate.rows; let hb = cb.empty(sf); let sh = cb.empty(d);
        let mut yr = vec![0f32; d];
        let mut yg = None;
        if let Some(xm) = &self.xc {
            // route on the host, run the cached experts on the GPU while the CPU pool runs the rest
            let mut xcg = xm.lock().unwrap();
            let sel = self.route_sel(il, &rv[d..d + ne], Some(&xcg));
            for &(e, _) in &sel { xcg.cnt[il][e] += xinc(); }
            xcg.total += sel.len() as u64;
            let (on, mut off): (Vec<(usize, f32)>, Vec<(usize, f32)>) = sel.into_iter().partition(|(e, _)| xcg.map.contains_key(&(il, *e)));
            tail_skip(&mut off, &on);
            // Strata-style PCIe share: the first few misses are copied into the stream slots and run on the GPU
            // with the cached ones, while the CPU pool takes the rest
            let tys0 = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
            let nst = if xfast() && tys0.0 == tys0.1 && tys0.0 <= 1 && tys0.2 == 3 && xcg.arena.len() <= 2 { let sh = xshare(); xcg.stream.len().min(if sh > 0.0 { (off.len() as f32 * sh).round() as usize } else { off.len() }) } else { 0 };
            // with PYRE_QN_XSHARE the last (lowest-weight) misses go to the GPU, as Strata's dispatch takes the tail
            let s0 = if xshare() > 0.0 { off.len() - nst } else { 0 };
            let strm: Vec<(usize, f32)> = if xarena_pin() {
                // pinned arena: the last nst misses already packed there (the rest stay CPU misses, as in Strata)
                let pk = self.xpk.lock().unwrap(); let mut pick: Vec<usize> = (0..off.len()).rev().filter(|&i| pk.map.contains_key(&(il, off[i].0))).take(nst).collect(); drop(pk);
                pick.sort_unstable(); let v: Vec<(usize, f32)> = pick.iter().map(|&i| off[i]).collect(); for &i in pick.iter().rev() { off.remove(i); } v
            } else { off.drain(s0..s0 + nst).collect() };
            xcg.hits += on.len() as u64; crate::metrics::expert_cache(on.len() as u64, off.len() as u64);
            // CPU workers start on the missed experts while this thread enqueues the GPU side
            let mut gpu = || {
                let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                let sents: Vec<(super::qn_cube::XEnt, f32)> = strm.iter().enumerate().map(|(j, &(e, w))| { let sl = xcg.stream[j]; let ff = c.ff_exp;
                    let _ = ff; let pk = self.xpacked(il, e, tys); let mut ent = cb.xupload_pk(&xcg.arena[sl / xcg.per], xcg.ar_bytes, sl % xcg.per, xcg.slot, &pk.0, pk.1, pk.2);
                    ent.arena = sl / xcg.per; ent.slot = sl; (ent, w) }).collect();
                let fast = !(on.is_empty() && sents.is_empty()) && xfast() && tys.0 == tys.1 && tys.0 <= 1 && tys.2 == 3 && {
                    let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcg.map[&(il, *e)], *w)).chain(sents.iter().map(|(e, w)| (e, *w))).collect();
                    if xcg.arena.len() <= 2 { cb.xrun_fast_to(&xcg.tabs, &xcg.arena, xcg.ar_bytes, &ents, &rd, rn, d, c.ff_exp, tys, &xcg.scr, false, &xcg.scr.y) } else {
                    let mut first = true; let mut ok = true;
                    for ai in 0..xcg.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ai).cloned().collect(); if grp.is_empty() { continue; }
                        ok &= cb.xrun_fast(&xcg.tabs, &xcg.arena[ai], xcg.ar_bytes, &grp, &rd, rn, d, c.ff_exp, tys, &xcg.scr, !first); first = false; }
                    ok } };
                if fast { yg = Some(xcg.scr.y.clone()); }
                if !on.is_empty() && !fast {
                    let xb = cb.empty(d); let y0 = cb.empty(d);
                    cb.q8copy(&rd, rn, 0, &xb, &y0, d);
                    let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
                    let rbs = (l.gate_e.row_bytes(), l.up_e.row_bytes(), l.down_e.row_bytes());
                    let _ = rbs; let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcg.map[&(il, *e)], *w)).collect();
                    for ai in 0..xcg.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ai).cloned().collect(); if grp.is_empty() { continue; }
                        let hx = cb.empty(c.ff_exp * grp.len()); cb.xrun_b(&xcg.tabs, &xcg.arena[ai], xcg.ar_bytes, &grp, &xb, d, c.ff_exp, tys, &hx, &y0); }
                    yg = Some(y0);
                }
                cb.silu_mul(&rd, rn, d + ne, d + ne + sf, &hb, sf);
                cb.gemv(&cl.sdown, &hb, sf, 0, &sh, d, 0);
                if kick() { let _ = cb.client.flush(); }
            };
            MISS[3].fetch_add(off.len() as u64, std::sync::atomic::Ordering::Relaxed); MISS[4].fetch_add(on.len() as u64, std::sync::atomic::Ordering::Relaxed);
            if !off.is_empty() {
                let lists: Vec<(usize, Vec<(usize, f32)>)> = off.iter().map(|&(e, w)| (e, vec![(0usize, w)])).collect();
                let work: Vec<(usize, &Vec<(usize, f32)>)> = lists.iter().map(|(e, v)| (*e, v)).collect();
                let tr = std::time::Instant::now();
                self.routed_pool(l, &rv[..d], &work, 1, &mut yr, &mut gpu);
                { let rns = tr.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
            } else { gpu(); }
            drop(xcg);
        } else {
            cb.silu_mul(&rd, rn, d + ne, d + ne + sf, &hb, sf);
            cb.gemv(&cl.sdown, &hb, sf, 0, &sh, d, 0);
            if kick() { let _ = cb.client.flush(); }
            self.routed(l, &rv[..d], &rv[d..d + ne], 1, &mut yr, false);
        }
        let q6 = std::time::Instant::now();
        let yb = cb.buf(&yr);
        cb.comb3(x, &sh, Some(&yb), yg.as_ref(), Some((&rd, rn, d + ne + 2 * sf)), &df, lrf, hc, d);
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
        HY[0].fetch_add(ns(q0, q1) + ns(q3, q4) + ns(q6, std::time::Instant::now()), Relaxed); HY[1].fetch_add(ns(q1, q2) + ns(q4, q5), Relaxed); HY[2].fetch_add(ns(q2, q3), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed); crate::metrics::add_gpu_ns(ns(q1, q2) + ns(q4, q5));
    }

    fn gdn_bufs(&self, st: &State) -> Vec<(H, H)> {
        let cb = self.cube.as_ref().unwrap();
        let mk = |_: ()| st.conv.iter().zip(&st.ssm).map(|(a, b)| (cb.buf(a), cb.buf(b))).collect::<Vec<_>>();
        if graph_on() { cb.client.memory_persistent_allocation((), mk) } else { mk(()) }
    }
    /// DeltaNet layer dense part (hc mix, in-proj, conv, delta, out-proj, hc combine, ffn hc mix, router/shared
    /// projections) on persistent buffers: captured once into a CUDA graph and replayed on later tokens, so its
    /// ~12 launches cost one. Recaptured when the state or residual buffers change (a new request).
    #[allow(clippy::type_complexity)]
    fn gdn_graph(&self, il: usize, cl: &super::qn_cube::CLayer, x: &H, st: &mut State, li: &mut usize) -> (H, H, usize, usize, std::time::Instant, std::time::Instant, std::time::Instant) {
        let cb = self.cube.as_ref().unwrap(); let c = &self.cfg; let (d, hc) = (c.d, c.hc); let w = hc * d;
        let gw = cl.gdn.as_ref().unwrap();
        let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let ch = cl.min.splits[1]; let sp = &cl.min.splits;
        let (cs, ss) = st.gdn[*li].clone(); *li += 1;
        let rn = d + cl.fin.rows; let lra = cl.hca.up.cols; let lrf = cl.hcf.up.cols;
        let mut gm = self.lg.lock().unwrap();
        // persistent pool: a sealed graph guards the pages it touched, so its buffers must not share pages with per-token temporaries
        let b = gm.entry(il).or_insert_with(|| cb.client.memory_persistent_allocation((), |_| LGraph { xn: cb.empty(w), da: cb.empty(cl.hca.di.rows), mixed: cb.empty(d), proj: cb.empty(cl.min.rows), o: cb.empty(d), co: cb.empty(ch), ob: cb.empty(nv * dk),
            df: cb.empty(cl.hcf.di.rows), rd: cb.empty(rn), g: None, key: String::new(), keep: vec![] }));
        let seg = |b: &LGraph| {
            cb.rms(x, &cl.hca.norm, &b.xn, hc, d, c.eps);
            cb.gemv(&cl.hca.di, &b.xn, w, 0, &b.da, cl.hca.di.rows, 0);
            cb.gate(&b.xn, &b.da, &cl.hca.up, &b.mixed, d, 0, hc, d);
            cb.gemv(&cl.min, &b.mixed, d, 0, &b.proj, cl.min.rows, 0);
            cb.conv(&b.proj, cl.min.rows, &gw.conv, &cs, &b.co, ch, c.conv_k);
            cb.delta(&b.co, ch, &b.proj, cl.min.rows, gw, &ss, &b.ob, nk, nv, dk, sp[1], sp[2], sp[3], c.eps);
            cb.gemv(&cl.mout, &b.ob, nv * dk, 0, &b.o, d, 0);
            cb.comb(x, &b.o, None, None, &b.da, lra, hc, d);
            cb.rms(x, &cl.hcf.norm, &b.xn, hc, d, c.eps);
            cb.gemv(&cl.hcf.di, &b.xn, w, 0, &b.df, cl.hcf.di.rows, 0);
            cb.gate(&b.xn, &b.df, &cl.hcf.up, &b.rd, rn, 0, hc, d);
            cb.gemv(&cl.fin, &b.rd, rn, 0, &b.rd, rn, d);
        };
        let key = format!("{:?}{:?}{:?}", cs, ss, x);
        if b.g.is_some() && b.key == key { unsafe { b.g.as_ref().unwrap().replay() }.expect("graph replay"); }
        else {
            b.g = None;
            // warmup run executes this token; the recorded run that follows does not execute
            if cb.client.graph_prepare().is_ok() { seg(b);
                match cb.client.start_capture() {
                    Ok(()) => { seg(b); match cb.client.stop_capture() { Ok(g) => { if env_var("PYRE_QN_XDEBUG").is_ok() { eprintln!("graph: captured layer {il}"); } b.g = Some(g); b.key = key; b.keep = vec![cs.clone(), ss.clone(), x.clone()]; }
                        Err(e) => { eprintln!("graph: layer {il} capture failed, graphs off: {e:?}"); GRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); } } }
                    Err(e) => { eprintln!("graph: layer {il} start failed, graphs off: {e:?}"); GRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); let _ = cb.client.stop_capture(); } }
            } else { seg(b); }
        }
        let t = std::time::Instant::now();
        (b.rd.clone(), b.df.clone(), rn, lrf, t, t, t)
    }

    /// Keep the most-routed experts (by count so far) resident on the GPU, within the cache budget.
    /// Expert (il, e) repacked for the GPU arena, from the host arena when it is there.
    fn xpacked(&self, il: usize, e: usize, tys: (u32, u32, u32)) -> super::qn_cube::Pk {
        if let Some(v) = self.xpk.lock().unwrap().map.get(&(il, e)) { return v.clone(); }
        let (l, ff, d) = (&self.layers[il], self.cfg.ff_exp, self.cfg.d);
        let t0 = std::time::Instant::now();
        let mut v = super::qn_cube::repack(tys.0, l.gate_e.raw_rows(e * ff, ff)); let uo = v.len(); v.extend(super::qn_cube::repack(tys.1, l.up_e.raw_rows(e * ff, ff))); let dof = v.len(); v.extend(super::qn_cube::repack(tys.2, l.down_e.raw_rows(e * d, d)));
        super::qn_cube::XUP_NS[0].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // shared (refcounted) bytes: an upload hands CubeCL a cheap clone instead of copying the expert again
        // PYRE_QN_XARENA_PIN=1: the packed expert lives in CubeCL's page-locked pool, so a PCIe-share upload of it is a
        // zero-copy async DMA (CubeCL never restages pinned bytes), as Strata streams from its pinned host blobs
        let mut by = burn_cubecl::cubecl::bytes::Bytes::from_elems(v);
        let cap0 = (env_var("PYRE_QN_XARENA_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) * (1u64 << 30) as f64) as usize;
        if xarena_pin() && { let pk = self.xpk.lock().unwrap(); !pk.full && pk.bytes + by.len() <= cap0 } { if let Some(cb) = self.cube.as_ref() { cb.client.staging(std::iter::once(&mut by), false); } }
        let a = std::sync::Arc::new((by.shared(), uo, dof));
        let cap = (env_var("PYRE_QN_XARENA_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) * (1u64 << 30) as f64) as usize;
        let mut pk = self.xpk.lock().unwrap();
        let len = a.0.len();
        if cap > 0 && !pk.full && pk.bytes + len <= cap {
            if pk.bytes >= pk.next_check { let rsv = env_usize("PYRE_QN_RAM_RESERVE_GB").unwrap_or(16) << 30;
                if avail_ram() < rsv + (1 << 30) { pk.full = true; eprintln!("host expert arena: stop at {:.2} GiB (RAM reserve)", pk.bytes as f64 / (1u64 << 30) as f64); } else { pk.next_check = pk.bytes + (1 << 30); } }
            if !pk.full { pk.bytes += len; pk.map.insert((il, e), a.clone()); }
        }
        a
    }
    /// Fill the host expert arena at load, hottest first by the routing profile, skipping the `skip` hottest
    /// (those live in the GPU cache), in parallel; stops at PYRE_QN_XARENA_GB or the RAM reserve.
    fn xpk_prefill(&self, skip: usize) {
        if env_var("PYRE_QN_XARENA_GB").map(|v| v.parse::<f64>().unwrap_or(0.0) <= 0.0).unwrap_or(true) { return; }
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let mut v: Vec<(u64, usize, usize)> = vec![];
        for (il, l) in self.layers.iter().enumerate() {
            let tys = (super::qn_cube::xtype(l.gate_e.ty), super::qn_cube::xtype(l.up_e.ty), super::qn_cube::xtype(l.down_e.ty));
            if tys.0.is_none() || tys.1.is_none() || tys.2.is_none() { continue; }
            for e in 0..self.cfg.n_exp { let pr = xprofile().as_ref().and_then(|p| p.get(il)).and_then(|r| r.get(e)).copied().unwrap_or(0.0); v.push(((pr * 1e6) as u64, il, e)); }
        }
        v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let v = &v[skip.min(v.len())..];
        for ch in v.chunks(512) {
            if self.xpk.lock().unwrap().full { break; }
            ch.par_iter().for_each(|&(_, il, e)| { let l = &self.layers[il]; let _ = self.xpacked(il, e, (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap())); });
            let cap = (env_var("PYRE_QN_XARENA_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) * (1u64 << 30) as f64) as usize;
            if self.xpk.lock().unwrap().bytes + (1 << 30) > cap { break; }
        }
        let pk = self.xpk.lock().unwrap();
        eprintln!("host expert arena: {} experts, {:.2} GiB pre-packed in {:.1}s, {:.1} GiB RAM free", pk.map.len(), pk.bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64(), avail_ram() as f64 / (1u64 << 30) as f64);
        super::qn_cube::XUP_NS[0].store(0, std::sync::atomic::Ordering::Relaxed);
    }
    /// PYRE_QN_ADAPT=1 (opt-in): Strata's adaptive VRAM tier (Strata, MIT, src/program/generate.cpp `adapt`):
    /// per layer, the most-routed missing experts (decayed count >= 2) replace the least-routed resident ones
    /// when they beat them by more than 1.5 (`--adapt-min-gain`); at most 96 swaps a round, biggest gains first;
    /// then every count is multiplied by 0.7 (`--adapt-decay`). Counts are kept x16 (xinc) so the decay works on integers.
    fn xadapt(&self, cb: &super::qn_cube::Cube, xc: &mut XCache) {
        let s = XSCALE; let floor = 2 * s; let gain = (env_f32("PYRE_QN_ADAPT_GAIN").unwrap_or(1.5) * s as f32) as u32;
        let maxs = env_usize("PYRE_QN_ADAPT_SWAPS").unwrap_or(96);
        let mut swaps: Vec<(u32, usize, usize, usize)> = vec![]; // gain, layer, in, out
        for (il, l) in self.layers.iter().enumerate() {
            if super::qn_cube::xtype(l.gate_e.ty).is_none() || super::qn_cube::xtype(l.up_e.ty).is_none() || super::qn_cube::xtype(l.down_e.ty).is_none() { continue; }
            let u = &xc.cnt[il];
            let mut cand: Vec<(u32, usize)> = vec![]; let mut vict: Vec<(u32, usize)> = vec![];
            for e in 0..self.cfg.n_exp { if xc.map.contains_key(&(il, e)) { vict.push((u[e], e)); } else if u[e] >= floor { cand.push((u[e], e)); } }
            if cand.is_empty() || vict.is_empty() { continue; }
            cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1))); vict.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            for i in 0..cand.len().min(vict.len()) { if cand[i].0 < vict[i].0 + gain { break; } swaps.push((cand[i].0 - vict[i].0, il, cand[i].1, vict[i].1)); }
        }
        swaps.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))); swaps.truncate(maxs);
        let mut jobs: Vec<(usize, usize, usize, (u32, u32, u32))> = vec![];
        for &(_, il, ein, eout) in &swaps {
            let Some(v) = xc.map.remove(&(il, eout)) else { continue };
            let l = &self.layers[il];
            let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
            jobs.push((il, ein, v.slot, tys));
        }
        { use rayon::prelude::*;
            let pks: Vec<_> = jobs.par_iter().map(|&(il, e, _, tys)| self.xpacked(il, e, tys)).collect();
            for (&(il, e, sl, _), pk) in jobs.iter().zip(pks.iter()) {
                let mut ent = cb.xupload_pk(&xc.arena[sl / xc.per], xc.ar_bytes, sl % xc.per, xc.slot, &pk.0, pk.1, pk.2);
                ent.arena = sl / xc.per; ent.slot = sl; xc.map.insert((il, e), ent); } }
        ADAPT_SWAPS.fetch_add(jobs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let dec = env_f32("PYRE_QN_ADAPT_DECAY").unwrap_or(0.7);
        for row in xc.cnt.iter_mut() { for c in row.iter_mut() { *c = (*c as f32 * dec) as u32; } }
    }
    fn xrefill(&self, cb: &super::qn_cube::Cube, xc: &mut XCache) {
        xc.calls += 1; if xc.calls <= 3 || xc.calls % 32 == 0 { crate::limits::vlog(&format!("refill call {} (cache {} experts)", xc.calls, xc.map.len())); }
        if adapt_on() && xc.calls != 1 && !xc.map.is_empty() { if xc.calls % adapt_every() == 0 { self.xadapt(cb, xc); } return; }
        let every = env_usize("PYRE_QN_XREFILL").unwrap_or(16);
        // an empty cache fills right away (the call before prefill has no routing counts yet, so the first
        // real fill used to wait until call `every` and then add only XSWAP experts)
        if xc.calls != 1 && xc.calls % every != 0 && !xc.map.is_empty() { return; }
        let ff = self.cfg.ff_exp; let d = self.cfg.d;
        let slot = xc.slot; let size = |_: &Layer| slot;
        if let Ok(pth) = env_var("PYRE_QN_XPROFILE_SAVE") { save_xprofile(&pth, &xc.cnt); }
        // PYRE_QN_XDECAY=k (opt-in): before ranking, every count loses 1/2^k, so recent routing weighs more (LFU with aging)
        if let Some(k) = env_usize("PYRE_QN_XDECAY").filter(|&k| k > 0 && k < 32) { for row in xc.cnt.iter_mut() { for c in row.iter_mut() { *c -= *c >> k; } } }
        let mut cand: Vec<(u64, usize, usize)> = vec![];
        for (il, l) in self.layers.iter().enumerate() {
            if super::qn_cube::xtype(l.gate_e.ty).is_none() || super::qn_cube::xtype(l.up_e.ty).is_none() || super::qn_cube::xtype(l.down_e.ty).is_none() { continue; }
            for e in 0..self.cfg.n_exp { let sc = xscore(xc, il, e); if sc > 0 { cand.push((sc, il, e)); } }
        }
        cand.sort_by(|a, b| b.0.cmp(&a.0));
        let mut want = std::collections::HashSet::new(); let mut tot = 0usize;
        for &(_, il, e) in &cand { let s = size(&self.layers[il]); if tot + s > xc.budget { break; } tot += s; want.insert((il, e)); }
        // eviction order: resident experts that fell out of the wanted set, least used first
        let mut evict: Vec<(u64, usize, usize)> = xc.map.keys().filter(|k| !want.contains(k)).map(|&(il, e)| (xscore(xc, il, e), il, e)).collect();
        // full tie-break: HashMap key order is random per process, and the slot an expert lands in decides the GPU summation order (run-to-run identical output)
        evict.sort_by(|a, b| b.0.cmp(&a.0).then((b.1, b.2).cmp(&(a.1, a.2))));
        let max_new = if xc.calls == 1 || xc.map.is_empty() { usize::MAX } else { env_usize("PYRE_QN_XSWAP").unwrap_or(64) };
        let t0 = std::time::Instant::now(); let mut n = 0; let mut jobs: Vec<(usize, usize, usize, (u32, u32, u32))> = vec![];
        for &(_, il, e) in &cand {
            if n >= max_new { break; }
            if !want.contains(&(il, e)) || xc.map.contains_key(&(il, e)) { continue; }
            let l = &self.layers[il];
            while xc.used + size(l) > xc.budget { match evict.pop() { Some((_, a, b)) => { if let Some(v) = xc.map.remove(&(a, b)) { xc.used -= v.bytes; xc.free.push(v.slot); } }, None => break } }
            if xc.used + size(l) > xc.budget || xc.free.is_empty() { break; }
            let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
            let sl = xc.free.pop().unwrap(); xc.used += xc.slot; jobs.push((il, e, sl, tys)); n += 1;
        }
        // the repacks of a refill run in parallel (they were ~1 ms each on this thread, up to PYRE_QN_XSWAP per refill);
        // the slots and the experts are the ones the serial loop picked, so the cache contents are unchanged
        { use rayon::prelude::*; let _ = (ff, d);
            let pks: Vec<_> = jobs.par_iter().map(|&(il, e, _, tys)| self.xpacked(il, e, tys)).collect();
            for (&(il, e, sl, _), pk) in jobs.iter().zip(pks.iter()) {
                let mut ent = cb.xupload_pk(&xc.arena[sl / xc.per], xc.ar_bytes, sl % xc.per, xc.slot, &pk.0, pk.1, pk.2);
                ent.arena = sl / xc.per; ent.slot = sl; xc.map.insert((il, e), ent); } }
        if env_var("PYRE_QN_GPUBENCH").is_ok() && xc.calls == 1 { if let Some(((il, e), ent)) = xc.map.iter().next() {
            let l = &self.layers[*il]; let _ = e;
            let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
            let rbs = (l.gate_e.row_bytes(), l.up_e.row_bytes(), l.down_e.row_bytes());
            let xb = cb.buf(&vec![0.01f32; d]); let y0 = cb.buf(&vec![0f32; d]); let hx = cb.empty(ff);
            let _ = rbs; cb.xrun_b(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &[(ent, 0.1)], &xb, d, ff, tys, &hx, &y0); cb.read(&y0);
            let mh = cb.client.create_from_slice(&[ent.off[0], ent.off[1], ent.off[2], 0.1f32.to_bits()].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); if let Some(Some(c0)) = self.cl.iter().find(|x| x.is_some()) { let xx = cb.buf(&vec![0.01f32; 4 * d]); let oo = cb.empty(20000); cb.gemv(&c0.fin, &xx, 4 * d, 0, &oo, 20000, 0); cb.read(&oo); let t1 = std::time::Instant::now(); for _ in 0..200 { cb.gemv(&c0.fin, &xx, 4 * d, 0, &oo, 20000, 0); } cb.read(&oo); eprintln!("cubebench (late) gemv fin: {:.1} us", t1.elapsed().as_secs_f64() * 1e6 / 200.0); }
            for sk in [0u32, 6, 5, 3] { cb.xrun_m(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &mh, 1, &xb, d, ff, tys, &hx, &y0, sk); cb.read(&y0); let t1 = std::time::Instant::now(); for _ in 0..200 { cb.xrun_m(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &mh, 1, &xb, d, ff, tys, &hx, &y0, sk); } cb.read(&y0); eprintln!("cubebench expert skip {sk}: {:.1} us", t1.elapsed().as_secs_f64() * 1e6 / 200.0); }
            eprintln!("cubebench expert {:?}/{:?}/{:?} ({:.2} MB)", l.gate_e.ty, l.up_e.ty, l.down_e.ty, ent.bytes as f64 / 1e6);
        } }
        if n > 0 && env_var("PYRE_QN_XDEBUG").is_ok() { eprintln!("xcache: +{n} experts in {:.1} ms, {} resident ({:.2} GiB), hit rate so far {:.1}%", t0.elapsed().as_secs_f64() * 1e3, xc.map.len(), xc.used as f64 / (1u64 << 30) as f64, 100.0 * xc.hits as f64 / xc.total.max(1) as f64); }
    }

    /// Hybrid layer: hyper-connection mixes + residual on the CPU (int8), big projections on the GPU.
    /// Three device round trips per layer and no small GPU kernels.
    fn layer_hyb(&self, l: &Layer, gl: &super::qn_gpu::GLayer, x: &mut Vec<f32>, st: &mut State, ai: &mut usize, li: &mut usize, start: usize, t: usize) {
        use burn::tensor::activation::{sigmoid, silu};
        let g = self.gpu.as_ref().unwrap(); let c = &self.cfg; let (d, ne) = (c.d, c.n_exp);
        let q0 = std::time::Instant::now();
        let (mixed, inj_a) = self.hc_mix(&l.hc_a, x, t);
        let q1 = std::time::Instant::now();
        let mi = g.read(gl.min.fwd(g.up(&mixed, [t, d])));
        let q2 = std::time::Instant::now();
        let proj = gl.min.split_host(&mi, t);
        let core = match l.mix { Mixer::Attn { .. } => { let o = self.attn_core(l, *ai, st, proj, start, t); *ai += 1; o }, Mixer::Lin { .. } => { let o = self.linear_core(l, *li, st, proj, t); *li += 1; o } };
        let q3 = std::time::Instant::now();
        let o = g.read(gl.mout.fwd(g.up(&core, [t, core.len() / t])));
        let q4 = std::time::Instant::now();
        self.hc_combine(x, &o, &inj_a, t);
        let (mf, inj_f) = self.hc_mix(&l.hc_f, x, t);
        let q5 = std::time::Instant::now();
        let mfd = g.up(&mf, [t, d]);
        let fi = gl.fin.fwd(mfd.clone()); let sf = l.sh_gate.rows;
        let sh = gl.sdown.fwd(silu(fi.clone().slice([0..t, ne..ne + sf])) * fi.clone().slice([0..t, ne + sf..ne + 2 * sf])) * sigmoid(mfd.matmul(gl.sgw.clone()));
        let both = g.read(burn::tensor::Tensor::cat(vec![fi.slice([0..t, 0..ne]), sh], 1));
        let w = d + ne; let mut lgc = Vec::with_capacity(t * ne); let mut yr = vec![0f32; t * d];
        for ti in 0..t { lgc.extend_from_slice(&both[ti * w..ti * w + ne]); }
        self.routed(l, &mf, &lgc, t, &mut yr, false);
        for ti in 0..t { for i in 0..d { yr[ti * d + i] += both[ti * w + ne + i]; } }
        self.hc_combine(x, &yr, &inj_f, t);
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
        HY[0].fetch_add(ns(q0, q1) + ns(q4, q5), Relaxed); HY[1].fetch_add(ns(q1, q2) + ns(q3, q4), Relaxed); HY[2].fetch_add(ns(q2, q3), Relaxed); HY[3].fetch_add(ns(q5, std::time::Instant::now()), Relaxed);
    }

    /// Run a projection group (GPU when uploaded, else CPU); one output per member, each [t][rows_i].
    fn mm(&self, ms: &[&QMat], x: &[f32], t: usize) -> Vec<Vec<f32>> {
        if let (Some(g), Some(gm)) = (&self.gpu, self.gmats.get(&ms[0].key())) {
            let mut y = vec![0f32; t * gm.rows]; gm.matmul(g, x, t, &mut y);
            return ms.iter().enumerate().map(|(i, m)| { let (a, b) = (gm.splits[i], gm.splits[i + 1]); let mut o = Vec::with_capacity(t * m.rows);
                for ti in 0..t { o.extend_from_slice(&y[ti * gm.rows + a..ti * gm.rows + b]); } o }).collect();
        }
        ms.iter().map(|m| { let mut y = vec![0f32; t * m.rows]; m.matmul(x, t, &mut y); y }).collect()
    }

    pub fn new_state(&self) -> State {
        let c = &self.cfg; let ch = 2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv;
        let mut st = State { kvg: vec![], gdn: vec![], gdn_gpu: false, tokens: vec![], conv: vec![], ssm: vec![], k: vec![], v: vec![], ik: vec![], ib: vec![], ple_hist: vec![], ck: vec![], ple_ck: vec![], spec_sel: vec![], pf: false, mtp_r: Default::default(), mtp_kv: Default::default() };
        for l in &self.layers {
            match l.mix { Mixer::Lin { .. } => { st.conv.push(vec![0.0; (c.conv_k - 1) * ch]); st.ssm.push(vec![0.0; c.lin_nv * c.lin_dk * c.lin_dv]); }, Mixer::Attn { .. } => { st.k.push(vec![]); st.v.push(vec![]); st.ik.push(vec![]); st.ib.push(vec![]); } }
            if l.ple.is_some() { st.ple_hist.push(vec![0.0; (c.ple_conv_k - 1) * c.ple_ngram * c.hc * c.d]); }
        }
        st
    }

    /// Grouped RMSNorm + low-rank sigmoid gate + stream mean (llama.cpp build_hc_mix).
    /// x: [T][hc*d]. Returns (mixed [T][d], inject [T][hc]).
    fn hc_mix(&self, h: &Hc, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let (d, hc) = (self.cfg.d, self.cfg.hc); let w = hc * d;
        let mut xn = vec![0f32; t * w];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let s = i % hc; rms(&x[i * d..(i + 1) * d], &h.norm[s * d..(s + 1) * d], self.cfg.eps, o); });
        let mut grp = vec![&h.down]; if let Some(i) = &h.inject { grp.push(i); }
        if let (Some((di, up)), true) = (&h.q8, t < 16) {
            // int8 path: down+inject in one pass, then up
            let z0 = std::time::Instant::now(); let nr = di.rows; let lr = h.down.rows; let mut o = vec![0f32; t * nr]; di.matmul(&xn, t, &mut o); let z1 = std::time::Instant::now();
            let mut lo = vec![0f32; t * lr]; let mut inj = vec![0f32; t * hc];
            for ti in 0..t { for i in 0..lr { lo[ti * lr + i] = silu(o[ti * nr + i] / hc as f32); } if h.inject.is_some() { for s in 0..hc { inj[ti * hc + s] = o[ti * nr + lr + s] / hc as f32; } } }
            let mut gate = vec![0f32; t * w]; up.matmul(&lo, t, &mut gate); let z2 = std::time::Instant::now();
            let mut mixed = vec![0f32; t * d];
            for ti in 0..t { for s in 0..hc { for i in 0..d { let k = ti * w + s * d + i; mixed[ti * d + i] += xn[k] * sigm(gate[k]); } } for i in 0..d { mixed[ti * d + i] /= hc as f32; } }
            HY[4].fetch_add((z1 - z0).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); HY[5].fetch_add((z2 - z1).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); HY[6].fetch_add(z2.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            return (mixed, inj);
        }
        let mut r = self.mm(&grp, &xn, t).into_iter();
        let mut lo = r.next().unwrap();
        for v in lo.iter_mut() { *v = silu(*v / hc as f32); }
        let mut inj = r.next().unwrap_or_else(|| vec![0f32; t * hc]);
        if h.inject.is_some() { for v in inj.iter_mut() { *v /= hc as f32; } }
        let gate = self.mm(&[&h.up], &lo, t).pop().unwrap();
        let mut mixed = vec![0f32; t * d];
        for ti in 0..t { for s in 0..hc { for i in 0..d { let k = ti * w + s * d + i; mixed[ti * d + i] += xn[k] * sigm(gate[k]); } } for i in 0..d { mixed[ti * d + i] /= hc as f32; } }
        (mixed, inj)
    }

    fn hc_combine(&self, x: &mut [f32], out: &[f32], inj: &[f32], t: usize) {
        let (d, hc) = (self.cfg.d, self.cfg.hc);
        for ti in 0..t { for s in 0..hc { let w = 2.0 * sigm(inj[ti * hc + s]); let r = &mut x[(ti * hc + s) * d..(ti * hc + s + 1) * d]; for i in 0..d { r[i] += out[ti * d + i] * w; } } }
    }

    fn ple_rows(&self, st: &State, start: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg; let tab = self.ple_tab.as_ref().unwrap();
        let nh = (c.ple_ngram - 1) * c.ple_per_gram; let hd = c.ple_dim;
        let mut emb = vec![0f32; t * nh * hd];
        for ti in 0..t {
            let p = start + ti;
            let mut ctx = vec![st.tokens[p] as u64; c.ple_ngram];
            let mut cut = false;
            for s in 1..c.ple_ngram { let tk = if cut || p < s { None } else { Some(st.tokens[p - s]) }; cut = cut || tk.is_none() || tk == Some(c.ple_eos); ctx[s] = if cut { c.ple_eos as u64 } else { tk.unwrap() as u64 }; }
            for n in 2..=c.ple_ngram {
                let mut mixed = ctx[0].wrapping_mul(c.ple_mult[0]);
                for j in 1..n { mixed ^= ctx[j].wrapping_mul(c.ple_mult[j]); }
                for gi in 0..c.ple_per_gram { let h = (n - 2) * c.ple_per_gram + gi; let row = (mixed % c.ple_vocab[h] + c.ple_off[h]) as usize;
                    tab.row(row, &mut emb[(ti * nh + h) * hd..(ti * nh + h + 1) * hd]); }
            }
        }
        emb
    }

    fn ple(&self, p: &Ple, hist: &mut Vec<f32>, emb: &[f32], x: &mut [f32], t: usize) {
        let c = &self.cfg; let (d, hc) = (c.d, c.hc); let w = hc * d; let ein = p.key.cols;
        let mut r = self.mm(&[&p.key, &p.value], emb, t).into_iter(); let key = r.next().unwrap(); let val = r.next().unwrap();
        let dil = c.ple_ngram; let hl = (c.ple_conv_k - 1) * dil; let _ = ein;
        for ti in 0..t {
            let xt = &mut x[ti * w..(ti + 1) * w];
            let mut gated = vec![0f32; w];
            for s in 0..hc {
                let mut kk = key[ti * w + s * d..ti * w + (s + 1) * d].to_vec(); rms_inplace(&mut kk, &p.nk[s * d..(s + 1) * d], c.eps);
                let mut q = vec![0f32; d]; rms(&xt[s * d..(s + 1) * d], &p.nq[s * d..(s + 1) * d], c.eps, &mut q);
                let sc = dot(&kk, &q) / (d as f32).sqrt();
                let g = sigm(sc.signum() * sc.abs().max(1e-6).sqrt());
                for i in 0..d { gated[s * d + i] = val[ti * d + i] * g; }
            }
            let mut nrm = gated.clone();
            for s in 0..hc { rms_inplace(&mut nrm[s * d..(s + 1) * d], &p.nc[s * d..(s + 1) * d], c.eps); }
            // depthwise causal conv: out[c] = sum_k w[c*K+k] * x[c][t - (K-1-k)*dil]; hist is oldest-first, hl rows
            let kk = c.ple_conv_k;
            for ch in 0..w {
                let mut o = p.conv[ch * kk + kk - 1] * nrm[ch];
                for k in 0..kk - 1 { let back = (kk - 1 - k) * dil; o += p.conv[ch * kk + k] * hist[(hl - back) * w + ch]; }
                xt[ch] += gated[ch] + silu(o);
            }
            hist.drain(..w); hist.extend_from_slice(&nrm);
        }
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.cfg.n_rot / 2;
        for i in 0..half { let a = pos as f32 * self.rope_inv[i]; let (s, co) = a.sin_cos(); let (x0, x1) = (v[i], v[i + half]); v[i] = x0 * co - x1 * s; v[i + half] = x1 * co + x0 * s; }
    }

    fn attn(&self, l: &Layer, ai: usize, st: &mut State, x: &[f32], start: usize, t: usize) -> Vec<f32> {
        let Mixer::Attn { q, k, v, o, idx, .. } = &l.mix else { unreachable!() };
        let mut grp = vec![q, k, v]; if let Some((iq, ik, _, _)) = idx { grp.push(iq); grp.push(ik); }
        let att = self.attn_core(l, ai, st, self.mm(&grp, x, t), start, t);
        self.mm(&[o], &att, t).pop().unwrap()
    }

    /// Attention after the q/k/v projections; returns the gated heads before the output projection.
    /// GPU KV ahead of the host copy (tokens decoded on the GPU): append them to st.k / st.v.
    fn kv_pull(&self, ai: usize, st: &mut State) {
        let row = self.cfg.n_kv * self.cfg.hd;
        if let (Some(cb), Some(Some(g))) = (&self.cube, st.kvg.get(ai)) { let cl = st.k[ai].len() / row;
            if g.len > cl { let (kk, vv) = (cb.read(&g.k), cb.read(&g.v)); st.k[ai].extend_from_slice(&kk[cl * row..g.len * row]); st.v[ai].extend_from_slice(&vv[cl * row..g.len * row]);
                let ih = self.cfg.idx_hd; if self.has_idx() { let ii = cb.read(&g.ik); st.ik[ai].extend_from_slice(&ii[cl * ih..g.len * ih]); } } }
    }
    /// Make the GPU KV hold every host position, with room for `need`.
    fn kv_push(&self, cb: &super::qn_cube::Cube, ai: usize, st: &mut State, need: usize) {
        let row = self.cfg.n_kv * self.cfg.hd;
        if st.kvg.len() <= ai { st.kvg.resize_with(ai + 1, || None); }
        let cl = st.k[ai].len() / row;
        if matches!(&st.kvg[ai], Some(g) if g.cap >= need && g.len >= cl) { return; }
        self.kv_pull(ai, st); let cl = st.k[ai].len() / row;
        let cap = need.max(cl).next_power_of_two().max(1024);
        let mut kb = st.k[ai].clone(); kb.resize(cap * row, 0.0); let mut vb = st.v[ai].clone(); vb.resize(cap * row, 0.0);
        let mut ib = st.ik[ai].clone(); ib.resize(cap * self.cfg.idx_hd, 0.0);
        st.kvg[ai] = Some(KvG { k: cb.buf(&kb), v: cb.buf(&vb), ik: cb.buf(&ib), cap, len: cl });
    }

    /// Visible tokens up to which every QSA block is selected (attention stays dense and exact).
    pub fn dense_n(&self) -> usize { if self.has_idx() { self.cfg.idx_budget + self.cfg.idx_ratio - 1 } else { usize::MAX } }
    fn has_idx(&self) -> bool { self.layers.iter().any(|l| matches!(&l.mix, Mixer::Attn { idx: Some(_), .. })) }

    fn attn_core(&self, l: &Layer, ai: usize, st: &mut State, proj: Vec<Vec<f32>>, start: usize, t: usize) -> Vec<f32> {
        let Mixer::Attn { qn, kn, idx, .. } = &l.mix else { unreachable!() };
        self.kv_pull(ai, st);
        let c = &self.cfg; let (nh, nkv, hd) = (c.n_head, c.n_kv, c.hd);
        let mut r = proj.into_iter(); let qf = r.next().unwrap(); let mut kf = r.next().unwrap(); let vf = r.next().unwrap();
        for ti in 0..t { for h in 0..nkv { let kh = &mut kf[ti * nkv * hd + h * hd..ti * nkv * hd + (h + 1) * hd]; rms_inplace(kh, kn, c.eps); self.rope(kh, start + ti); } }
        st.k[ai].extend_from_slice(&kf); st.v[ai].extend_from_slice(&vf);
        // QSA indexer (transformers Qwen4ExpTextQSAIndexer): blocks of `ratio` tokens, each scored by
        // sum over index heads of relu(q_h . rope(knorm(mean raw keys))) ; the top budget/ratio blocks
        // plus the incomplete tail are the only keys a query attends to. Dense while all blocks fit.
        let (ihd, ir) = (c.idx_hd, c.idx_ratio);
        let mut iqs: Vec<f32> = vec![];
        if let Some((_, _, iqn, ikn)) = idx {
            let iqf = r.next().unwrap(); let ikf = r.next().unwrap();
            st.ik[ai].extend_from_slice(&ikf);
            let nt = st.ik[ai].len() / ihd;
            while st.ib[ai].len() / ihd < nt / ir {
                let b = st.ib[ai].len() / ihd; let mut pk = vec![0f32; ihd];
                for p in b * ir..(b + 1) * ir { for i in 0..ihd { pk[i] += st.ik[ai][p * ihd + i]; } }
                for v in pk.iter_mut() { *v /= ir as f32; }
                rms_inplace(&mut pk, ikn, c.eps); self.rope(&mut pk, b * ir);
                st.ib[ai].extend_from_slice(&pk);
            }
            if start + t > self.dense_n() && env_var("PYRE_QN_DENSE").is_err() {
                iqs = iqf; let inh = c.idx_nh;
                for ti in 0..t { for h in 0..inh { let qh = &mut iqs[(ti * inh + h) * ihd..(ti * inh + h + 1) * ihd]; rms_inplace(qh, iqn, c.eps); self.rope(qh, start + ti); } }
            }
        }
        let (kc, vc) = (&st.k[ai], &st.v[ai]);
        let scale = (hd as f32).sqrt().recip(); let grp = nh / nkv;
        // per query: selected key positions (None = all visible)
        let sel: Vec<Option<Vec<usize>>> = (0..t).into_par_iter().map(|ti| {
            let n = start + ti + 1;
            if iqs.is_empty() || n <= self.dense_n() { return None; }
            let nb = n / ir; let inh = c.idx_nh; let ib = &st.ib[ai]; let iscale = (ihd as f32).sqrt().recip();
            let q = &iqs[ti * inh * ihd..(ti + 1) * inh * ihd];
            let mut sc: Vec<(f32, usize)> = (0..nb).map(|b| { let kb = &ib[b * ihd..(b + 1) * ihd]; let mut s = 0f32; for h in 0..inh { s += dot(&q[h * ihd..(h + 1) * ihd], kb).max(0.0); } (s * iscale, b) }).collect();
            let kk = c.idx_budget / ir;
            sc.select_nth_unstable_by(kk - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut ps: Vec<usize> = sc[..kk].iter().flat_map(|&(_, b)| b * ir..(b + 1) * ir).collect();
            ps.extend(nb * ir..n); ps.sort_unstable(); Some(ps)
        }).collect();
        let mut att = vec![0f32; t * nh * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
            let (ti, h) = (idx / nh, idx % nh);
            let base = ti * nh * 2 * hd + h * 2 * hd;
            let mut qh = qf[base..base + hd].to_vec(); rms_inplace(&mut qh, qn, c.eps); self.rope(&mut qh, start + ti);
            let kvh = h / grp; let n = start + ti + 1;
            let all: Vec<usize>; let ps: &[usize] = match &sel[ti] { Some(v) => v, None => { all = (0..n).collect(); &all } };
            let mut sc: Vec<f32> = ps.iter().map(|&p| dot(&qh, &kc[p * nkv * hd + kvh * hd..p * nkv * hd + (kvh + 1) * hd]) * scale).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            let inv = 1.0 / sum; for (&p, s) in ps.iter().zip(sc.iter()) { let w = s * inv; let vv = &vc[p * nkv * hd + kvh * hd..p * nkv * hd + (kvh + 1) * hd]; for (o, v) in out.iter_mut().zip(vv) { *o += w * v; } }
            let gate = &qf[base + hd..base + 2 * hd]; for i in 0..hd { out[i] *= sigm(gate[i]); }
        });
        // past the dense range the GPU copy of this layer's KV is never used again
        if start + t > self.dense_n() && ai < st.kvg.len() { st.kvg[ai] = None; }
        att
    }

    fn linear(&self, l: &Layer, li: usize, st: &mut State, x: &[f32], t: usize) -> Vec<f32> {
        let Mixer::Lin { qkv, z, beta, alpha, out, .. } = &l.mix else { unreachable!() };
        let o_all = self.linear_core(l, li, st, self.mm(&[qkv, z, beta, alpha], x, t), t);
        self.mm(&[out], &o_all, t).pop().unwrap()
    }

    /// Gated DeltaNet after its input projections (conv, L2 norms, delta-rule recurrence, gated norm).
    fn linear_core(&self, l: &Layer, li: usize, st: &mut State, proj: Vec<Vec<f32>>, t: usize) -> Vec<f32> {
        let Mixer::Lin { qkv, a, dt, conv, norm, .. } = &l.mix else { unreachable!() };
        let c = &self.cfg; let (nk, nv, dk, dv) = (c.lin_nk, c.lin_nv, c.lin_dk, c.lin_dv); let ch = qkv.rows; let kk = c.conv_k;
        let mut r = proj.into_iter();
        let (m, zz, bb, aa) = (r.next().unwrap(), r.next().unwrap(), r.next().unwrap(), r.next().unwrap());
        let mut o_all = vec![0f32; t * nv * dv];
        let cs = &mut st.conv[li]; let s = &mut st.ssm[li];
        // 1) causal depthwise conv + SiLU for every token (sequential in time, vectorised over channels)
        let mut co_all = vec![0f32; t * ch];
        for ti in 0..t {
            let cur = &m[ti * ch..(ti + 1) * ch]; let co = &mut co_all[ti * ch..(ti + 1) * ch];
            for cc in 0..ch { let mut o = conv[cc * kk + kk - 1] * cur[cc]; for k in 0..kk - 1 { o += conv[cc * kk + k] * cs[k * ch + cc]; } co[cc] = silu(o); }
            cs.drain(..ch); cs.extend_from_slice(cur);
        }
        // 2) L2-normalise q (and scale) and k per head
        co_all.par_chunks_mut(ch).for_each(|co| { let (qp, rest) = co.split_at_mut(nk * dk); let (kp, _) = rest.split_at_mut(nk * dk);
            for h in 0..nk { for (buf, sc) in [(&mut qp[h * dk..(h + 1) * dk], (dk as f32).sqrt().recip()), (&mut kp[h * dk..(h + 1) * dk], 1.0)] {
                let n = (buf.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip() * sc; for v in buf.iter_mut() { *v *= n; } } } });
        // 3) the delta-rule recurrence: heads are independent, so one task per head runs all tokens
        let co_all = &co_all; let w = nv * dv;
        let heads: Vec<Vec<f32>> = s.par_chunks_mut(dk * dv).enumerate().map(|(h, sh)| {
            let kh = h % nk; let mut oh_all = vec![0f32; t * dv]; let mut mem = vec![0f32; dv]; let mut delta = vec![0f32; dv];
            for ti in 0..t {
                let co = &co_all[ti * ch..(ti + 1) * ch];
                let qv = &co[kh * dk..(kh + 1) * dk]; let kv = &co[nk * dk + kh * dk..nk * dk + (kh + 1) * dk]; let vv = &co[2 * nk * dk + h * dv..2 * nk * dk + (h + 1) * dv];
                let g = (softplus(aa[ti * nv + h] + dt[h]) * a[h]).exp(); let b = sigm(bb[ti * nv + h]);
                for v in sh.iter_mut() { *v *= g; }
                mem.iter_mut().for_each(|v| *v = 0.0);
                for i in 0..dk { let ki = kv[i]; let row = &sh[i * dv..(i + 1) * dv]; for (m, r) in mem.iter_mut().zip(row) { *m += r * ki; } }
                for j in 0..dv { delta[j] = (vv[j] - mem[j]) * b; }
                let oh = &mut oh_all[ti * dv..(ti + 1) * dv];
                for i in 0..dk { let ki = kv[i]; let qi = qv[i]; let row = &mut sh[i * dv..(i + 1) * dv];
                    for ((r, dl), o) in row.iter_mut().zip(&delta).zip(oh.iter_mut()) { *r += ki * dl; *o += *r * qi; } }
                rms_inplace(oh, norm, c.eps); let zh = &zz[ti * w + h * dv..ti * w + (h + 1) * dv]; for j in 0..dv { oh[j] *= sigm(zh[j]); }
            }
            oh_all
        }).collect();
        for (h, oh) in heads.iter().enumerate() { for ti in 0..t { o_all[ti * w + h * dv..ti * w + (h + 1) * dv].copy_from_slice(&oh[ti * dv..(ti + 1) * dv]); } }
        o_all
    }

    fn ffn(&self, l: &Layer, x: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg; let d = c.d; let ff = c.ff_exp;
        let mut r = self.mm(&[&l.router, &l.sh_gate, &l.sh_up], x, t).into_iter();
        let (lg, mut g, u) = (r.next().unwrap(), r.next().unwrap(), r.next().unwrap());
        // shared expert (batched)
        for i in 0..g.len() { g[i] = silu(g[i]) * u[i]; }
        let mut y = self.mm(&[&l.sh_down], &g, t).pop().unwrap();
        self.routed(l, x, &lg, t, &mut y, true);
        y
    }

    /// Prompt chunks: each expert's rows are dequantized once to f32 and multiplied against all of the
    /// tokens routed to it with a cache-blocked SGEMM (matrixmultiply), tasks = (expert, row block).
    fn routed_gemm(&self, l: &Layer, x: &[f32], work: &[(usize, &Vec<(usize, f32)>)], t: usize, y: &mut [f32]) {
        let c = &self.cfg; let (d, ff) = (c.d, c.ff_exp);
        const RB: usize = 128; const DB: usize = 512;
        let gemm = |m: usize, k: usize, n: usize, a: &[f32], w: &[f32], out: &mut [f32]| unsafe {
            // out[m, n] = a[m, k] * w[n, k]^T
            matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, w.as_ptr(), 1, k as isize, 0.0, out.as_mut_ptr(), n as isize, 1);
        };
        let parts: Vec<(usize, usize, Vec<f32>)> = work.par_iter().flat_map(|&(e, toks)| (0..ff.div_ceil(RB)).into_par_iter().map(move |b| {
            let (r0, r1) = (b * RB, ((b + 1) * RB).min(ff)); let nr = r1 - r0; let n = toks.len();
            let mut xe = vec![0f32; n * d]; for (j, &(ti, _)) in toks.iter().enumerate() { xe[j * d..(j + 1) * d].copy_from_slice(&x[ti * d..(ti + 1) * d]); }
            let mut w = vec![0f32; nr * d];
            let gm = l.gate_e.slice_rows(e * ff + r0, nr); for r in 0..nr { gm.row(r, &mut w[r * d..(r + 1) * d]); }
            let mut hg = vec![0f32; n * nr]; gemm(n, d, nr, &xe, &w, &mut hg);
            let um = l.up_e.slice_rows(e * ff + r0, nr); for r in 0..nr { um.row(r, &mut w[r * d..(r + 1) * d]); }
            let mut hu = vec![0f32; n * nr]; gemm(n, d, nr, &xe, &w, &mut hu);
            for i in 0..hg.len() { hg[i] = silu(hg[i]) * hu[i]; }
            (e, r0, hg)
        })).collect();
        let mut hid: std::collections::HashMap<usize, Vec<f32>> = work.iter().map(|&(e, toks)| (e, vec![0f32; toks.len() * ff])).collect();
        for (e, r0, hb) in parts { let h = hid.get_mut(&e).unwrap(); let n = h.len() / ff; let w = hb.len() / n; for j in 0..n { h[j * ff + r0..j * ff + r0 + w].copy_from_slice(&hb[j * w..(j + 1) * w]); } }
        let hid = &hid;
        let outs: Vec<(usize, usize, Vec<f32>)> = work.par_iter().flat_map(|&(e, toks)| (0..d.div_ceil(DB)).into_par_iter().map(move |b| {
            let (r0, r1) = (b * DB, ((b + 1) * DB).min(d)); let nr = r1 - r0; let n = toks.len();
            let mut w = vec![0f32; nr * ff]; let dm = l.down_e.slice_rows(e * d + r0, nr); for r in 0..nr { dm.row(r, &mut w[r * ff..(r + 1) * ff]); }
            let mut o = vec![0f32; n * nr]; gemm(n, ff, nr, &hid[&e], &w, &mut o);
            (e, r0, o)
        })).collect();
        let by: std::collections::HashMap<usize, &Vec<(usize, f32)>> = work.iter().copied().collect();
        for (e, r0, o) in outs { let toks = by[&e]; let nr = o.len() / toks.len();
            for (j, &(ti, wt)) in toks.iter().enumerate() { let yr = &mut y[ti * d + r0..ti * d + r0 + nr]; for i in 0..nr { yr[i] += wt * o[j * nr + i]; } } }
    }

    /// Shared-expert gate (applied to y's shared output) + the routed experts on the CPU, added into y.
    fn routed(&self, l: &Layer, x: &[f32], lg: &[f32], t: usize, y: &mut [f32], shared_gate: bool) {
        let c = &self.cfg; let d = c.d; let ff = c.ff_exp; let t_r = std::time::Instant::now();
        if shared_gate { for ti in 0..t { let sg = sigm(dot(&l.sh_gate_inp, &x[ti * d..(ti + 1) * d])); for v in &mut y[ti * d..(ti + 1) * d] { *v *= sg; } } }
        // routing: softmax -> top-k -> renormalise
        let mut by_exp: Vec<Vec<(usize, f32)>> = vec![vec![]; c.n_exp];
        for ti in 0..t {
            let r = &lg[ti * c.n_exp..(ti + 1) * c.n_exp];
            let mut idx: Vec<usize> = (0..c.n_exp).collect();
            idx.select_nth_unstable_by(c.top_k, |a, b| r[*b].partial_cmp(&r[*a]).unwrap());
            let top = &idx[..c.top_k]; let mx = top.iter().map(|&e| r[e]).fold(f32::NEG_INFINITY, f32::max);
            let ws: Vec<f32> = top.iter().map(|&e| (r[e] - mx).exp()).collect(); let sum: f32 = ws.iter().sum();
            for (&e, w) in top.iter().zip(ws) { by_exp[e].push((ti, w / sum)); }
            if let Some(xm) = &self.xc { if let Some(il) = self.layers.iter().position(|x| std::ptr::eq(x, l)) { let mut g = xm.lock().unwrap(); for &e in top { g.cnt[il][e] += xinc(); } } }
        }
        let work: Vec<(usize, &Vec<(usize, f32)>)> = by_exp.iter().enumerate().filter(|(_, v)| !v.is_empty()).collect();
        // PYRE_QN_PF_POOL=1: prompt chunks also take the int8 pool path below (decode numerics, opt-in: it
        // changes the prompt's rounding against the f32 path, so greedy output can differ)
        let pf_pool = env_var("PYRE_QN_PF_POOL").map(|v| v == "1").unwrap_or(false);
        if !pf_pool && t * c.top_k >= work.len() * env_usize("PYRE_QN_GEMM_MIN").unwrap_or(8) { // >= 8 tokens per expert on average
            self.routed_gemm(l, x, &work, t, y);
            { let rns = t_r.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
            return;
        }
        if (t <= 4 || pf_pool) && env_var("PYRE_QN_SPIN").map(|v| v != "0").unwrap_or(true) && super::qdot::supports(l.gate_e.ty) && super::qdot::supports(l.up_e.ty) && super::qdot::supports(l.down_e.ty)
            && env_var("PYRE_QN_QDOT").map(|v| v != "0").unwrap_or(true) {
            self.routed_pool(l, x, &work, t, y, &mut || {});
            { let rns = t_r.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
            return;
        }
        // one task per (expert, block of rows): keeps all cores busy for decode (10 experts) and prefill alike
        let blk: usize = env_usize("PYRE_QN_BLK").unwrap_or(if t == 1 { 64 } else { 64 });
        use super::qdot;
        let qon = env_var("PYRE_QN_QDOT").map(|v| v != "0").unwrap_or(true);
        let qgu = qon && qdot::supports(l.gate_e.ty) && qdot::supports(l.up_e.ty);
        let qdn = qon && qdot::supports(l.down_e.ty);
        let acts: Vec<qdot::Q8Act> = if qgu { (0..t).map(|ti| qdot::quantize(&x[ti * d..(ti + 1) * d])).collect() } else { vec![] }; let acts = &acts;
        let parts: Vec<(usize, usize, Vec<f32>)> = work.par_iter().flat_map(|&(e, toks)| (0..ff.div_ceil(blk)).into_par_iter().map(move |b| {
            let (r0, r1) = (b * blk, ((b + 1) * blk).min(ff));
            let gm = l.gate_e.slice_rows(e * ff + r0, r1 - r0); let um = l.up_e.slice_rows(e * ff + r0, r1 - r0);
            let mut hbuf = vec![0f32; toks.len() * (r1 - r0)];
            if qgu { let mut sc = qdot::Scratch::default();
                let mut s2 = qdot::Scratch::default();
                let pf = env_usize("PYRE_QN_PREFETCH").unwrap_or(2);
                for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                    if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                    if toks.len() > 2 { qdot::decode(gm.ty, gr, d, &mut sc); qdot::decode(um.ty, ur, d, &mut s2);
                        for (j, &(ti, _)) in toks.iter().enumerate() { hbuf[j * (r1 - r0) + r] = silu(qdot::dot_decoded(&sc, &acts[ti])) * qdot::dot_decoded(&s2, &acts[ti]); }
                        continue; }
                    for (j, &(ti, _)) in toks.iter().enumerate() { hbuf[j * (r1 - r0) + r] = silu(qdot::row_dot(gm.ty, gr, &acts[ti], &mut sc)) * qdot::row_dot(um.ty, ur, &acts[ti], &mut sc); } }
                return (e, r0, hbuf);
            }
            let mut buf = vec![0f32; d];
            for r in 0..r1 - r0 {
                gm.row(r, &mut buf); let gv: Vec<f32> = toks.iter().map(|&(ti, _)| dot(&buf, &x[ti * d..(ti + 1) * d])).collect();
                um.row(r, &mut buf); for (j, &(ti, _)) in toks.iter().enumerate() { hbuf[j * (r1 - r0) + r] = silu(gv[j]) * dot(&buf, &x[ti * d..(ti + 1) * d]); }
            }
            (e, r0, hbuf)
        })).collect();
        // assemble per-expert hidden [ntok][ff]
        let mut hid: Vec<Vec<f32>> = by_exp.iter().map(|v| vec![0f32; v.len() * ff]).collect();
        for (e, r0, hb) in parts { let n = by_exp[e].len(); let w = hb.len() / n; for j in 0..n { hid[e][j * ff + r0..j * ff + r0 + w].copy_from_slice(&hb[j * w..(j + 1) * w]); } }
        // down projections, row blocks of the output in parallel
        let dblk: usize = env_usize("PYRE_QN_DBLK").unwrap_or(if t == 1 { 128 } else { 128 });
        PH[0].fetch_add(t_r.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let hq: Vec<Vec<qdot::Q8Act>> = if qdn { hid.iter().map(|h| h.chunks(ff).map(qdot::quantize).collect()).collect() } else { vec![] };
        let outs: Vec<(usize, Vec<f32>)> = (0..d.div_ceil(dblk)).into_par_iter().map(|b| {
            let (r0, r1) = (b * dblk, ((b + 1) * dblk).min(d)); let mut o = vec![0f32; t * (r1 - r0)]; let mut buf = vec![0f32; ff];
            let mut sc = qdot::Scratch::default();
            for &(e, toks) in &work { let dm = l.down_e.slice_rows(e * d + r0, r1 - r0);
                if qdn { for r in 0..r1 - r0 { let dr = dm.row_raw(r);
                    if toks.len() > 2 { qdot::decode(dm.ty, dr, ff, &mut sc); for (j, &(ti, w)) in toks.iter().enumerate() { o[ti * (r1 - r0) + r] += w * qdot::dot_decoded(&sc, &hq[e][j]); } continue; }
                    for (j, &(ti, w)) in toks.iter().enumerate() { o[ti * (r1 - r0) + r] += w * qdot::row_dot(dm.ty, dr, &hq[e][j], &mut sc); } } continue; }
                for r in 0..r1 - r0 { dm.row(r, &mut buf); for (j, &(ti, w)) in toks.iter().enumerate() { o[ti * (r1 - r0) + r] += w * dot(&buf, &hid[e][j * ff..(j + 1) * ff]); } } }
            (r0, o)
        }).collect();
        for (r0, o) in outs { let w = o.len() / t; for ti in 0..t { for r in 0..w { y[ti * d + r0 + r] += o[ti * w + r]; } } }
        { let rns = t_r.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
    }

    /// Decode-size expert pass on the spinning pool: gate/up tasks (expert x row block) write the
    /// hidden rows in place, then down tasks (output row block) accumulate into y. No per-task Vecs.
    fn routed_pool(&self, l: &Layer, x: &[f32], work: &[(usize, &Vec<(usize, f32)>)], t: usize, y: &mut [f32], side: &mut dyn FnMut()) { self.routed_pool_ord(l, x, work, t, y, side, None) }
    /// `ord`: per token, its (work index, slot) list in that token's own expert order; the down projection then
    /// accumulates each token's experts in exactly the order a one-token call would (bit-identical to decode).
    #[allow(clippy::too_many_arguments)]
    fn routed_pool_ord(&self, l: &Layer, x: &[f32], work: &[(usize, &Vec<(usize, f32)>)], t: usize, y: &mut [f32], side: &mut dyn FnMut(), ord: Option<&[Vec<(usize, usize)>]>) {
        use super::qdot; use super::spin_pool::SyncPtr;
        let (d, ff) = (self.cfg.d, self.cfg.ff_exp);
        let pool = qn_pool(); let z0 = std::time::Instant::now();
        let acts: Vec<qdot::Q8Act> = (0..t).map(|ti| qdot::quantize(&x[ti * d..(ti + 1) * d])).collect();
        let usek = qdot::supports_k(l.gate_e.ty) && qdot::supports_k(l.up_e.ty) && env_var("PYRE_QN_Q8K").map(|v| v != "0").unwrap_or(true);
        let actk: Vec<qdot::Q8K> = if usek { (0..t).map(|ti| { let mut a = qdot::Q8K::default(); qdot::quantize_k_into(&x[ti * d..(ti + 1) * d], &mut a); a }).collect() } else { vec![] };
        let blk = env_usize("PYRE_QN_SBLK").unwrap_or(32); let nb = ff.div_ceil(blk);
        let mut hoff = vec![0usize; work.len() + 1]; for (i, (_, toks)) in work.iter().enumerate() { hoff[i + 1] = hoff[i] + toks.len() * ff; }
        let mut hid = vec![0f32; hoff[work.len()]]; let hp = SyncPtr(hid.as_mut_ptr());
        thread_local! { static SC: std::cell::RefCell<(super::qdot::Scratch, super::qdot::Scratch)> = std::cell::RefCell::new(Default::default()); }
        let (t_first, t_last) = (std::sync::atomic::AtomicU64::new(u64::MAX), std::sync::atomic::AtomicU64::new(0)); let zp = std::time::Instant::now();
        pool.run_side(work.len() * nb, &|k| { let tb = std::time::Instant::now(); t_first.fetch_min((tb - zp).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            struct B<'a>(std::time::Instant, std::time::Instant, &'a std::sync::atomic::AtomicU64); impl Drop for B<'_> { fn drop(&mut self) { BUSY.fetch_add(self.0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); BUSYN.fetch_add(1, std::sync::atomic::Ordering::Relaxed); self.2.fetch_max(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } } let _b = B(tb, zp, &t_last);
            let (wi, b) = (k / nb, k % nb); let (e, toks) = work[wi];
            let (r0, r1) = (b * blk, ((b + 1) * blk).min(ff));
            let gm = l.gate_e.slice_rows(e * ff + r0, r1 - r0); let um = l.up_e.slice_rows(e * ff + r0, r1 - r0);
            SC.with(|c| { let (sc, s2) = &mut *c.borrow_mut();
                let pf = env_usize("PYRE_QN_PREFETCH").unwrap_or(2);
                let multi = usek && toks.len() >= 2 && toks.len() <= 16 && gm.ty == um.ty && matches!(gm.ty, super::gguf_q::QType::IQ2S | super::gguf_q::QType::IQ2XXS);
                if multi { let al: Vec<&qdot::Q8K> = toks.iter().map(|&(ti, _)| &actk[ti]).collect(); let (mut gv, mut uv) = ([0f32; 16], [0f32; 16]);
                    for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                        if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                        qdot::row_dot_k_multi(gm.ty, gr, &al, &mut gv); qdot::row_dot_k_multi(um.ty, ur, &al, &mut uv);
                        for j in 0..toks.len() { unsafe { hp.slice(hoff[wi] + j * ff + r0 + r, 1)[0] = silu(gv[j]) * uv[j]; } } }
                    return; }
                for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                    if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                    for (j, &(ti, _)) in toks.iter().enumerate() {
                        let v = if usek { let pr = if pair_on() && gm.ty == um.ty { qdot::row_dot_k_pair(gm.ty, gr, ur, &actk[ti]) } else { None }; if let Some((g, u)) = pr { silu(g) * u } else { silu(qdot::row_dot_k(gm.ty, gr, &actk[ti])) * qdot::row_dot_k(um.ty, ur, &actk[ti]) } } else { silu(qdot::row_dot(gm.ty, gr, &acts[ti], sc)) * qdot::row_dot(um.ty, ur, &acts[ti], s2) };
                        unsafe { hp.slice(hoff[wi] + j * ff + r0 + r, 1)[0] = v; } } } });
        }, &mut || { let ts = std::time::Instant::now(); side(); SIDE_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); });
        let z1 = std::time::Instant::now(); TL[0].fetch_add(t_first.load(std::sync::atomic::Ordering::Relaxed).min(1 << 40), std::sync::atomic::Ordering::Relaxed); TL[1].fetch_add(t_last.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed); TL[2].fetch_add((z1 - zp).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let used = l.down_e.ty == super::gguf_q::QType::Q2_0 && ff % 64 == 0 && env_var("PYRE_QN_Q8D").map(|v| v != "0").unwrap_or(true);
        let hq: Vec<Vec<qdot::Q8Act>> = if used { vec![] } else { work.iter().enumerate().map(|(wi, (_, toks))| (0..toks.len()).map(|j| qdot::quantize(&hid[hoff[wi] + j * ff..hoff[wi] + (j + 1) * ff])).collect()).collect() };
        let hdq: Vec<Vec<qdot::Q8D>> = if !used { vec![] } else { work.iter().enumerate().map(|(wi, (_, toks))| (0..toks.len()).map(|j| { let mut a = qdot::Q8D::default(); qdot::quantize_d_into(&hid[hoff[wi] + j * ff..hoff[wi] + (j + 1) * ff], &mut a); a }).collect()).collect() };
        let dblk = env_usize("PYRE_QN_SDBLK").unwrap_or(64); let yp = SyncPtr(y.as_mut_ptr()); let z2 = std::time::Instant::now();
        let mut slot0 = vec![0usize; work.len()]; let mut slot_n = 0usize; for (wi, (_, toks)) in work.iter().enumerate() { slot0[wi] = slot_n; slot_n += toks.len(); } let (slot0, slot_n) = (&slot0, slot_n);
        pool.run(d.div_ceil(dblk), &|b| {
            let (r0, r1) = (b * dblk, ((b + 1) * dblk).min(d));
            if let Some(ord) = ord { SC.with(|c| { let (sc, _) = &mut *c.borrow_mut();
                // expert-major (sequential, prefetched) into per-(expert, token) partials, then each token
                // sums its partials in its own expert order: the same additions as a one-token call
                let nr = r1 - r0; let mut part = vec![0f32; slot_n * nr];
                for (wi, &(e, toks)) in work.iter().enumerate() { let dm = l.down_e.slice_rows(e * d + r0, nr);
                    for r in 0..nr { let dr = dm.row_raw(r); if r + 4 < nr { prefetch(dm.row_raw(r + 4)); }
                        for (j, &(_, w)) in toks.iter().enumerate() { part[(slot0[wi] + j) * nr + r] = w * if used { qdot::q2_0_dot_d(dr, &hdq[wi][j]) } else { qdot::row_dot(dm.ty, dr, &hq[wi][j], sc) }; } } }
                for (ti, lst) in ord.iter().enumerate() { let yt = unsafe { yp.slice(ti * d + r0, nr) };
                    for &(wi, j) in lst { let pp = &part[(slot0[wi] + j) * nr..(slot0[wi] + j + 1) * nr]; for r in 0..nr { yt[r] += pp[r]; } } } }); return; }
            SC.with(|c| { let (sc, _) = &mut *c.borrow_mut();
                for (wi, &(e, toks)) in work.iter().enumerate() { let dm = l.down_e.slice_rows(e * d + r0, r1 - r0);
                    for r in 0..r1 - r0 { let dr = dm.row_raw(r); if r + 4 < r1 - r0 { prefetch(dm.row_raw(r + 4)); }
                        for (j, &(ti, w)) in toks.iter().enumerate() { let v = w * if used { qdot::q2_0_dot_d(dr, &hdq[wi][j]) } else { qdot::row_dot(dm.ty, dr, &hq[wi][j], sc) }; unsafe { yp.slice(ti * d + r0 + r, 1)[0] += v; } } } } });
        });
        use std::sync::atomic::Ordering::Relaxed; HY[4].fetch_add((z1 - z0).as_nanos() as u64, Relaxed); HY[5].fetch_add((z2 - z1).as_nanos() as u64, Relaxed); HY[6].fetch_add(z2.elapsed().as_nanos() as u64, Relaxed);
    }

    /// Feed `ids` (appended to `st.tokens`) and return the logits of the last one.
    pub fn forward(&self, ids: &[u32], st: &mut State) -> Vec<f32> {
        // PYRE_QN_PF_MULTI=K (opt-in): prompt chunks go through the decode verify path K tokens at a time (dense
        // on the GPU, decode numerics); the expert cache fills after the first piece instead of after the prompt
        let k = env_usize("PYRE_QN_PF_MULTI").unwrap_or(0);
        if k > 1 && ids.len() > 1 && self.multi_ok() {
            let mut last = vec![];
            for c in ids.chunks(k) {
                last = if c.len() == 1 { self.forward_inner(c, st) } else { st.pf = true; let v = self.forward_multi_x(c, st, true).pop().unwrap(); st.pf = false; v };
                if let (Some(cb), Some(xm)) = (&self.cube, &self.xc) { let mut g = xm.lock().unwrap(); if g.calls == 0 { self.xrefill(cb, &mut g); } }
            }
            return last;
        }
        let lg = self.forward_inner(ids, st);
        // first fill of the expert cache from the prompt's routing counts (part of prefill time)
        if let (Some(cb), Some(xm), true) = (&self.cube, &self.xc, ids.len() > 1) { let mut g = xm.lock().unwrap(); if g.calls == 0 { self.xrefill(cb, &mut g); } }
        lg
    }

    fn forward_inner(&self, ids: &[u32], st: &mut State) -> Vec<f32> {
        let c = &self.cfg; let (d, hc) = (c.d, c.hc); let t = ids.len(); let start = st.tokens.len();
        st.tokens.extend_from_slice(ids);
        let mut x = vec![0f32; t * hc * d];
        let mut e = vec![0f32; d];
        for (ti, &id) in ids.iter().enumerate() { self.embd.row(id as usize, &mut e); for s in 0..hc { x[(ti * hc + s) * d..(ti * hc + s + 1) * d].copy_from_slice(&e); } }
        let emb = if self.ple_tab.is_some() { Some(self.ple_rows(st, start, t)) } else { None };
        let (mut li, mut ai, mut pi) = (0, 0, 0);
        let mut xd: Option<burn::tensor::Tensor<2>> = None;
        let mut xc: Option<burn_cubecl::cubecl::server::Handle> = None;
        let gdn_ok = t == 1 && self.cube.is_some() && env_var("PYRE_QN_GDN_GPU").map(|v| v != "0").unwrap_or(true)
            && self.cl.len() == self.layers.len() && self.cl.iter().zip(&self.layers).all(|(c, l)| c.as_ref().is_some_and(|c| c.gdn.is_some() || !matches!(l.mix, Mixer::Lin { .. })));
        if let (Some(cb), Some(xm), 1) = (&self.cube, &self.xc, t) { let mut g = xm.lock().unwrap(); if !g.map.is_empty() || g.calls > 0 { self.xrefill(cb, &mut g); } }
        if let Some(cb) = &self.cube {
            if gdn_ok && !st.gdn_gpu { st.gdn = self.gdn_bufs(st); st.gdn_gpu = true; }
            if !gdn_ok && st.gdn_gpu { for (i, (a, b)) in st.gdn.iter().enumerate() { st.conv[i] = cb.read(a); st.ssm[i] = cb.read(b); } st.gdn.clear(); st.gdn_gpu = false; }
        }
        let dbg = env_var("PYRE_QN_DEBUG").is_ok();
        let gpu_tmax = env_usize("PYRE_QN_GPU_TMAX").unwrap_or(8);
        let hyb = env_var("PYRE_QN_GPU_HC").map(|v| v == "0").unwrap_or(false);
        let prof = env_var("PYRE_QN_PROFILE").is_ok(); let mut prof_acc = [0f64; 5]; let t_all = std::time::Instant::now();
        for (il, l) in self.layers.iter().enumerate() {
            let tt = std::time::Instant::now();
            if let Some(h) = &xc { if l.ple.is_some() || !matches!(self.cl.get(il), Some(Some(_))) { x = self.cube.as_ref().unwrap().read(h); xc = None; } }
            if xd.is_some() && (l.ple.is_some() || !matches!(self.gl.get(il), Some(Some(_)))) { x = self.gpu.as_ref().unwrap().read(xd.take().unwrap()); }
            if let Some(p) = &l.ple { let mut h = std::mem::take(&mut st.ple_hist[pi]); self.ple(p, &mut h, emb.as_ref().unwrap(), &mut x, t); st.ple_hist[pi] = h; pi += 1; }
            if let (Some(Some(cl)), 1) = (self.cl.get(il), t) {
                let cb = self.cube.as_ref().unwrap();
                if xc.is_none() { xc = Some(if graph_on() {
                    // one persistent residual buffer, rewritten in place, so captured graphs keep pointing at it
                    let mut g = self.xper.lock().unwrap(); let n = x.len();
                    let h = g.get_or_insert_with(|| cb.client.memory_persistent_allocation(n, |n| cb.client.empty(n * 4))).clone();
                    cb.client.write(&h, burn_cubecl::cubecl::bytes::Bytes::from_elems(x.clone())); h
                } else { cb.buf(&x) }); }
                if prof { prof_acc[0] += tt.elapsed().as_secs_f64(); if l.ple.is_some() { prof_acc[1] += 1.0; } }
                self.layer_cube(il, l, cl, xc.as_ref().unwrap(), st, &mut ai, &mut li, start);
                if prof { prof_acc[3] += tt.elapsed().as_secs_f64(); }
                continue;
            }
            if let (Some(Some(gl)), true) = (self.gl.get(il), t <= gpu_tmax) {
                let g = self.gpu.as_ref().unwrap();
                if hyb { self.layer_hyb(l, gl, &mut x, st, &mut ai, &mut li, start, t); if prof { prof_acc[3] += tt.elapsed().as_secs_f64(); } continue; }
                let xin = match xd.take() { Some(v) => v, None => g.up(&x, [t, hc * d]) };
                xd = Some(self.layer_gpu(l, gl, xin, st, &mut ai, &mut li, start, t));
                if prof { prof_acc[3] += tt.elapsed().as_secs_f64(); }
                continue;
            }
            let (m, inj) = self.hc_mix(&l.hc_a, &x, t);
            let t_hc = tt.elapsed();
            let o = match l.mix { Mixer::Attn { .. } => { let o = self.attn(l, ai, st, &m, start, t); ai += 1; o }, Mixer::Lin { .. } => { let o = self.linear(l, li, st, &m, t); li += 1; o } };
            let t_mix = tt.elapsed();
            self.hc_combine(&mut x, &o, &inj, t);
            let (m, inj) = self.hc_mix(&l.hc_f, &x, t);
            let t_hc2 = tt.elapsed();
            let o = self.ffn(l, &m, t);
            self.hc_combine(&mut x, &o, &inj, t);
            if prof { prof_acc[0] += t_hc.as_secs_f64() + (t_hc2 - t_mix).as_secs_f64(); prof_acc[if matches!(l.mix, Mixer::Attn { .. }) { 1 } else { 2 }] += (t_mix - t_hc).as_secs_f64(); prof_acc[3] += (tt.elapsed() - t_hc2).as_secs_f64(); }
            if dbg { eprintln!("layer {il}: x[0,:4] = {:?}", &x[(t - 1) * hc * d..(t - 1) * hc * d + 4]); }
        }
        if let Some(h) = xc.take() { x = self.cube.as_ref().unwrap().read(&h); }
        let gpu_rows = xd.is_some();
        let t0 = if self.mtp.is_some() { t.saturating_sub(self.mtp.as_ref().unwrap().window) } else { t - 1 };
        if let Some(v) = xd.take() { let r = self.gpu.as_ref().unwrap().read(v.slice([t0..t, 0..hc * d])); x[t0 * hc * d..t * hc * d].copy_from_slice(&r); }
        let _ = gpu_rows; if self.mtp.is_some() { for ti in t0..t { self.mtp_keep_r(st, start + ti, &x[ti * hc * d..(ti + 1) * hc * d]); } }
        let last = &x[(t - 1) * hc * d..t * hc * d];
        let (h, _) = self.hc_mix(&self.head, last, 1);
        let th = std::time::Instant::now();
        let lg = self.head_logits(&h);
        if prof { prof_acc[4] = th.elapsed().as_secs_f64(); let (gn, gs) = super::qn_gpu::take_stats(); let rs = ROUTED_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9; { let f = |i: usize| HY[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9; eprint!("[cube: enqueue {:.3} reads {:.3} core {:.3} sh+routed {:.3}] hyb hc {:.3} (down {:.3} up {:.3} mix {:.3}) rt12 {:.3} core {:.3} ffn {:.3} | ", f(0), f(1), f(2), f(3), 0.0, f(4), f(5), f(6), 0.0, 0.0, 0.0); } eprint!("xupload repack {:.4} write {:.4} | ", super::qn_cube::XUP_NS[0].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, super::qn_cube::XUP_NS[1].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9); eprint!("routed {rs:.3}s (gate/up {:.3}) ", PH[0].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9); eprint!("side(gpu enqueue) {:.4} gu-busy {:.4} tasks {} first {:.4} last {:.4} wall {:.4} | ", SIDE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, BUSY.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, BUSYN.swap(0, std::sync::atomic::Ordering::Relaxed), TL[0].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, TL[1].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, TL[2].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9); eprint!("cpu misses {} gpu hits {} | ", MISS[3].swap(0, std::sync::atomic::Ordering::Relaxed), MISS[4].swap(0, std::sync::atomic::Ordering::Relaxed)); eprintln!("prof T={t} total {:.3}s | hc {:.3} attn {:.3} gdn {:.3} ffn {:.3} head {:.3} | gpu {gn} calls {gs:.3}s", t_all.elapsed().as_secs_f64(), prof_acc[0], prof_acc[1], prof_acc[2], prof_acc[3], prof_acc[4]); }
        lg
    }


    /// Expert choice for one token. With a route prior (crate::spec::route_prior() > 0, default off) the
    /// selection uses logits + lambda for experts already resident in the GPU expert cache; the weights stay
    /// the softmax of the unbiased logits over the picked experts (residency-biased routing, idea from
    /// Strata's STRATA_ROUTE_PRIOR / arXiv 2412.00099). Changes outputs, so it is opt-in.
    fn route_sel(&self, il: usize, r: &[f32], xc: Option<&XCache>) -> Vec<(usize, f32)> {
        let k = self.cfg.top_k; let lam = crate::spec::route_prior();
        let Some(xc) = xc.filter(|_| lam > 0.0) else { return top_k(r, k) };
        let b: Vec<f32> = (0..r.len()).map(|e| r[e] + if xc.map.contains_key(&(il, e)) { lam } else { 0.0 }).collect();
        let mut idx: Vec<usize> = (0..r.len()).collect();
        idx.select_nth_unstable_by(k, |x, y| b[*y].partial_cmp(&b[*x]).unwrap());
        let top = &idx[..k]; let mx = top.iter().map(|&e| r[e]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = top.iter().map(|&e| (r[e] - mx).exp()).collect(); let sum: f32 = ws.iter().sum();
        top.iter().zip(ws).map(|(&e, w)| (e, w / sum)).collect()
    }

    /// MTP drafter (see `mtp`); None when no MTP layer is loaded.
    pub fn mtp_draft(&self, st: &mut State, history: &[u32], k: usize, s: Option<&mut crate::sampler::Sampler>) -> Option<Vec<(u32, Option<Vec<(u32, f32)>>)>> {
        self.mtp_draft_in(st, history, k, s)
    }
    fn mtp_draft_in(&self, st: &mut State, history: &[u32], k: usize, _s: Option<&mut crate::sampler::Sampler>) -> Option<Vec<(u32, Option<Vec<(u32, f32)>>)>> {
        let m = self.mtp.as_ref()?; let n = st.tokens.len(); let pending = *history.last()?;
        if n == 0 || k == 0 { return None; }
        let t0 = std::time::Instant::now(); let d = self.cfg.d; let row = m.row(); let w = m.rdim();
        // PYRE_MTP_GPU=1 (opt-in): the drafter's dense matrices (~95 MB Q8) run as GPU GEMVs on this thread; experts stay on the CPU
        let gh: Option<Box<dyn Fn(&super::qdot::Q8Mat, &[f32], usize) -> Option<Vec<f32>> + '_>> = match (&self.cube, env_var("PYRE_MTP_GPU").map(|v| v == "1").unwrap_or(false)) { (Some(cb), true) => {
            let map = self.mtp_dense.get_or_init(|| { let mut mp = std::collections::HashMap::new(); let mut by = 0usize;
                for q in m.dense_mats() { if let Some(c) = cb.upload_q8(q.rows, q.cols, q.blocks()) { by += q.blocks().len(); mp.insert(q as *const _ as usize, c); } }
                eprintln!("mtp: {} dense drafter matrices on gpu ({:.0} MB)", mp.len(), by as f64 / 1e6); mp });
            Some(Box::new(move |q: &super::qdot::Q8Mat, x: &[f32], t: usize| { let g = map.get(&(q as *const _ as usize))?; let xb = cb.buf(x); let o = cb.empty(t * q.rows);
                if t == 1 { cb.gemv(g, &xb, x.len(), 0, &o, q.rows, 0); } else { cb.gemv_m(g, &xb, t * q.cols, 0, q.cols, &o, t * q.rows, 0, q.rows, t); }
                Some(cb.read(&o)) })) }
            _ => None };
        let mut kv = std::mem::take(&mut st.mtp_kv); kv.truncate(n - 1, row);
        // catch-up: cells i < n-1 whose residual is known (token i+1 is committed)
        let first = st.mtp_r.front().map(|r| r.0).unwrap_or(n);
        let lo = if kv.n > 0 && kv.end() >= first { kv.end() } else { first.max((n - 1).saturating_sub(m.window)) };
        if lo < n - 1 {
            let mut rs = Vec::with_capacity((n - 1 - lo) * w); let mut es = vec![0f32; (n - 1 - lo) * d]; let mut ok = true;
            for (j, i) in (lo..n - 1).enumerate() { match self.mtp_r_get(st, i) { Some(r) => rs.extend_from_slice(r), None => { ok = false; break; } } self.embd.row(st.tokens[i + 1] as usize, &mut es[j * d..(j + 1) * d]); }
            if ok { if let Some(h) = &gh { super::mtp::with_mm_hook(h, || m.catch_up(&mut kv, &rs, &es, n - 1 - lo, lo)); } else { mtp_cpu(|| m.catch_up(&mut kv, &rs, &es, n - 1 - lo, lo)); } } else { kv = Default::default(); }
        }
        let t1 = std::time::Instant::now();
        let mut r = self.mtp_r_get(st, n - 1)?.to_vec(); let mut tok = pending; let mut out = vec![];
        // a later draft is kept only when the drafter is this sure of it (Strata gates draft length on the MTP probability too; PYRE_MTP_MIN_P=0 drafts the full length)
        let min_p: f32 = crate::spec::draft_min_p().unwrap_or_else(|| env_var("PYRE_MTP_MIN_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9));
        let mut e = vec![0f32; d];
        for j in 0..k {
            self.embd.row(tok as usize, &mut e);
            let (s, r2) = if let Some(h) = &gh { super::mtp::with_mm_hook(h, || m.step(&mut kv, &r, &e, n - 1 + j)) } else { mtp_cpu(|| m.step(&mut kv, &r, &e, n - 1 + j)) }; let th = std::time::Instant::now();
            // the draft head: the main head on the GPU when it is there (full vocabulary), else the CPU subset
            // the draft-vocabulary head on the GPU (PYRE_MTP_HEAD_GPU=0: off): ~100 MB of VRAM, a fraction of the full head's time
            let hg = if env_var("PYRE_MTP_HEAD_GPU").map(|v| v != "0").unwrap_or(true) { self.cube.as_ref().and_then(|cb| self.mtp_headc.get_or_init(|| { let h = m.head_q8(); if h.rows == 0 { return None; } let r = cb.upload_q8(h.rows, h.cols, h.blocks()); if r.is_some() { eprintln!("mtp: draft head on gpu ({} rows)", h.rows); } r }).as_ref().map(|hm| (cb, hm))) } else { None };
            let (id, p) = if let Some((cb, hm)) = hg { let xb = cb.buf(&s); let o = cb.empty(hm.rows); cb.gemv(hm, &xb, s.len(), 0, &o, hm.rows, 0); let (i, p) = super::mtp::best(&cb.read(&o)); (m.head_ids()[i], p) }
                else if self.headc.is_some() && env_var("PYRE_MTP_HEAD_CPU").is_err() { let (i, p) = super::mtp::best(&self.head_logits(&s)); (i as u32, p) } else { m.head_cpu(&s) };
            HEAD_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            if j > 0 && p < min_p { break; }
            out.push((id, None)); r = r2; tok = id;
        }
        kv.truncate(n, row); st.mtp_kv = kv;
        if env_var("PYRE_MTP_DEBUG").is_ok() { eprintln!("mtp: head {:.1} ms catch-up {:.1} ms, {} drafts {:.1} ms | {:?}", HEAD_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6, (t1 - t0).as_secs_f64() * 1e3, out.len(), t1.elapsed().as_secs_f64() * 1e3, super::mtp::TM.iter().map(|a| (a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e5).round() / 10.0).collect::<Vec<_>>()); }
        Some(out)
    }
    fn mtp_keep_r(&self, st: &mut State, pos: usize, x: &[f32]) {
        let Some(m) = &self.mtp else { return };
        while st.mtp_r.back().is_some_and(|r| r.0 >= pos) { st.mtp_r.pop_back(); }
        if st.mtp_r.back().is_some_and(|r| r.0 + 1 != pos) { st.mtp_r.clear(); }
        st.mtp_r.push_back((pos, x.to_vec()));
        while st.mtp_r.len() > m.window + 16 { st.mtp_r.pop_front(); }
    }
    fn mtp_r_get<'a>(&self, st: &'a State, pos: usize) -> Option<&'a [f32]> { let f = st.mtp_r.front()?.0; st.mtp_r.get(pos.checked_sub(f)?).filter(|r| r.0 == pos).map(|r| r.1.as_slice()) }
    /// Load the MTP drafter (scripts/mtp_fetch.py's file) when speculative MTP drafting is enabled.
    /// Default draft length: 2 (MTP spec2, greedy-identical and measured faster) when the MTP drafter
    /// loaded and multi-token verify can run; otherwise 0 (plain decode).
    pub fn spec_default(&self) -> usize { if self.mtp.is_some() && self.multi_ok() { 2 } else { 0 } }
    pub fn load_mtp(&mut self) -> Result<()> {
        if !crate::spec::mtp_on() { return Ok(()); }
        let Some(p) = super::mtp::find() else { return Ok(()) };
        let c = &self.cfg; let t = std::time::Instant::now();
        let geo = super::mtp::Geo { d: c.d, hc: c.hc, nh: c.n_head, nkv: c.n_kv, hd: c.hd, n_rot: c.n_rot, eps: c.eps, top_k: c.top_k, rope_base: c.rope_base };
        let ids = super::mtp::draft_vocab(self.out.rows); let out = &self.out;
        let ids = if self.headc.is_some() && env_var("PYRE_MTP_HEAD_CPU").is_err() { vec![] } else { ids };
        let m = super::mtp::Mtp::load(&p, &geo, ids, |r, y| out.row(r, y))?;
        if env_var("PYRE_MTP_LAZY").is_err() { m.warm(); }
        crate::limits::vlog("mtp loaded"); eprintln!("mtp: drafter {} loaded in {:.1} s", p.display(), t.elapsed().as_secs_f64());
        self.mtp = Some(m); Ok(())
    }

    // ---- speculative decoding: multi-token verify on the decode (CubeCL) path, with rollback ----

    /// True when `forward_multi` can run: every layer on the CubeCL decode path with DeltaNet on the GPU.
    pub fn multi_ok(&self) -> bool {
        self.cube.is_some() && env_var("PYRE_QN_GDN_GPU").map(|v| v != "0").unwrap_or(true)
            && self.cl.len() == self.layers.len() && self.cl.iter().zip(&self.layers).all(|(c, l)| c.as_ref().is_some_and(|c| c.gdn.is_some() || !matches!(l.mix, Mixer::Lin { .. })))
    }

    /// Feed `ids` (t >= 1 tokens, appended to `st.tokens`) and return the logits after EACH of them.
    /// Every token goes through exactly the kernels of one-token decode (so greedy output is identical);
    /// what is shared is the CPU expert work: each routed expert's weights are read once for all tokens
    /// that picked it. Checkpoints the recurrent state after every token so `spec_keep` can roll back.
    fn head_logits(&self, h: &[f32]) -> Vec<f32> {
        if let (Some(hm), Some(cb)) = (&self.headc, &self.cube) { let xb = cb.buf(h); let o = cb.empty(hm.rows); cb.gemv(hm, &xb, h.len(), 0, &o, hm.rows, 0); return cb.read(&o); }
        self.mm(&[&self.out], h, 1).pop().unwrap()
    }
    pub fn forward_multi(&self, ids: &[u32], st: &mut State) -> Vec<Vec<f32>> { self.forward_multi_x(ids, st, false) }
    /// `last_only`: logits for the last token only (prompt chunks).
    fn forward_multi_x(&self, ids: &[u32], st: &mut State, last_only: bool) -> Vec<Vec<f32>> {
        let t = ids.len();
        if t == 1 || !self.multi_ok() {
            // no verify path: plain sequential decode (no rollback checkpoints; callers check multi_ok)
            return ids.iter().map(|&id| self.forward(&[id], st)).collect();
        }
        let c = &self.cfg; let (d, hc) = (c.d, c.hc); let start = st.tokens.len(); let t_m = std::time::Instant::now();
        let cb = self.cube.as_ref().unwrap();
        st.tokens.extend_from_slice(ids);
        st.spec_sel = vec![vec![]; t];
        if let Some(xm) = &self.xc { let mut g = xm.lock().unwrap(); for _ in 0..t { if !g.map.is_empty() || g.calls > 0 { self.xrefill(cb, &mut g); } } }
        if !st.gdn_gpu { st.gdn = self.gdn_bufs(st); st.gdn_gpu = true; }
        // checkpoint buffers for t-1 positions
        let ch = 2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv; let (cn, sn) = ((c.conv_k - 1) * ch, c.lin_nv * c.lin_dk * c.lin_dv);
        while !st.pf && st.ck.len() < t - 1 { st.ck.push((0..st.gdn.len()).map(|_| (cb.empty(cn), cb.empty(sn))).collect()); }
        let mut xh: Vec<Vec<f32>> = vec![vec![0f32; hc * d]; t];
        let mut e = vec![0f32; d];
        for (ti, &id) in ids.iter().enumerate() { self.embd.row(id as usize, &mut e); for s in 0..hc { xh[ti][s * d..(s + 1) * d].copy_from_slice(&e); } }
        let emb = if self.ple_tab.is_some() { Some(self.ple_rows(st, start, t)) } else { None };
        let ew = (c.ple_ngram.max(1) - 1) * c.ple_per_gram * c.ple_dim;
        st.ple_ck = vec![vec![]; st.ple_hist.len()];
        let (mut li, mut ai, mut pi) = (0, 0, 0);
        let mut xs: Option<Vec<burn_cubecl::cubecl::server::Handle>> = None;
        // PYRE_QN_WAVE=1: the window in two token groups, pipelined so the GPU runs one group's dense part while
        // the CPU runs the other's routed experts (each token's arithmetic is unchanged; Strata overlaps the
        // same two phases across its batch groups)
        if t >= 2 && env_var("PYRE_QN_WAVE").map(|v| v != "0").unwrap_or(false) {
            let ta = t.div_ceil(2); let grp = [(0usize, ta), (ta, t)];
            let mut ctr = vec![]; { let (mut a, mut b) = (0, 0); for l in &self.layers { ctr.push((a, b)); match l.mix { Mixer::Attn { .. } => a += 1, Mixer::Lin { .. } => b += 1 } } }
            let stages: Vec<(usize, usize)> = (0..self.layers.len()).flat_map(|il| [(il, 0usize), (il, 1usize)]).collect();
            let mut dq: [Option<(H, H, H)>; 2] = [None, None];
            for (i, &(il, g)) in stages.iter().enumerate() {
                let l = &self.layers[il]; let cl = self.cl[il].as_ref().unwrap(); let (o0, o1) = grp[g];
                if dq[g].is_none() {
                    if g == 0 { if let Some(p) = &l.ple {
                        if let Some(hs) = xs.take() { xh = hs.iter().map(|h| cb.read(h)).collect(); }
                        let mut h = std::mem::take(&mut st.ple_hist[pi]);
                        for ti in 0..t { self.ple(p, &mut h, &emb.as_ref().unwrap()[ti * ew..(ti + 1) * ew], &mut xh[ti], 1); if ti + 1 < t && !st.pf { st.ple_ck[pi].push(h.clone()); } }
                        st.ple_hist[pi] = h; pi += 1;
                    } }
                    if xs.is_none() { xs = Some(if vgraph_on() { self.vxs_fill(cb, &xh) } else { xh.iter().map(|v| cb.buf(v)).collect() }); }
                    dq[g] = Some(self.lcm_dense(il, l, cl, &xs.as_ref().unwrap()[o0..o1], st, ctr[il].0, ctr[il].1, start + o0, o0, t));
                }
                let dd = dq[g].take().unwrap();
                let q4 = std::time::Instant::now();
                let fut = cb.read_start(&dd.0);
                if let Some(&(nl, ng)) = stages.get(i + 1) { if !(ng == 0 && self.layers[nl].ple.is_some()) {
                    let (n0, n1) = grp[ng];
                    dq[ng] = Some(self.lcm_dense(nl, &self.layers[nl], self.cl[nl].as_ref().unwrap(), &xs.as_ref().unwrap()[n0..n1], st, ctr[nl].0, ctr[nl].1, start + n0, n0, t));
                    let _ = cb.client.flush();
                } }
                let q4b = std::time::Instant::now();
                if prewake() { qn_pool().prewake(); }
                let rall = cb.read_finish(fut);
                HY[1].fetch_add(q4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                WV[g].fetch_add(q4b.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); WV[2].fetch_add((q4b - q4).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                self.lcm_moe(il, l, cl, &xs.as_ref().unwrap()[o0..o1], st, dd, rall, o0);
            }
        } else {
        for (il, l) in self.layers.iter().enumerate() {
            if let Some(p) = &l.ple {
                if let Some(hs) = xs.take() { xh = hs.iter().map(|h| cb.read(h)).collect(); }
                let mut h = std::mem::take(&mut st.ple_hist[pi]);
                for ti in 0..t { self.ple(p, &mut h, &emb.as_ref().unwrap()[ti * ew..(ti + 1) * ew], &mut xh[ti], 1); if ti + 1 < t && !st.pf { st.ple_ck[pi].push(h.clone()); } }
                st.ple_hist[pi] = h; pi += 1;
            }
            let cl = self.cl[il].as_ref().unwrap();
            if xs.is_none() { xs = Some(if vgraph_on() { self.vxs_fill(cb, &xh) } else { xh.iter().map(|v| cb.buf(v)).collect() }); }
            self.layer_cube_multi(il, l, cl, xs.as_ref().unwrap(), st, &mut ai, &mut li, start);
        }
        }
        let hs = xs.unwrap();
        let mut lg: Vec<Vec<f32>> = vec![];
        for (ti, h) in hs.iter().enumerate() { if last_only && ti + 1 < t && self.mtp.is_none() { continue; } let x = cb.read(h); if self.mtp.is_some() { self.mtp_keep_r(st, start + ti, &x); } if last_only && ti + 1 < t { continue; } let (hh, _) = self.hc_mix(&self.head, &x, 1); lg.push(self.head_logits(&hh)); }
        if env_var("PYRE_QN_PROFILE").is_ok() { use std::sync::atomic::Ordering::Relaxed; let f = |i: usize| HY[i].swap(0, Relaxed) as f64 / 1e6;
            eprint!("cpu distinct {} entries {} gpu hits {} | ", MISS[0].swap(0, Relaxed), MISS[1].swap(0, Relaxed), MISS[2].swap(0, Relaxed));
            eprint!("wave wait A {:.1} B {:.1} enqueue-next {:.1} | ", WV[0].swap(0, Relaxed) as f64 / 1e6, WV[1].swap(0, Relaxed) as f64 / 1e6, WV[2].swap(0, Relaxed) as f64 / 1e6);
            eprint!("xrun {:.1} upl {:.1} flush {:.1} | ", SD[0].swap(0, Relaxed) as f64 / 1e6, SD[1].swap(0, Relaxed) as f64 / 1e6, SD[2].swap(0, Relaxed) as f64 / 1e6); eprint!("gu first {:.1} last {:.1} wall {:.1} | ", TL[0].swap(0, Relaxed) as f64 / 1e6, TL[1].swap(0, Relaxed) as f64 / 1e6, TL[2].swap(0, Relaxed) as f64 / 1e6); eprint!("busy {:.1} tasks {} side {:.1} | ", BUSY.swap(0, Relaxed) as f64 / 1e6, BUSYN.swap(0, Relaxed), SIDE_NS.swap(0, Relaxed) as f64 / 1e6);
            eprintln!("prof multi T={t} total {:.1} ms | dense enqueue {:.1} reads {:.1} sh+routed {:.1} (routed {:.1}: gate/up {:.1} quant {:.1} down {:.1}) ms", t_m.elapsed().as_secs_f64() * 1e3, f(0), f(1), f(3), ROUTED_NS.swap(0, Relaxed) as f64 / 1e6, f(4), f(5), f(6)); }
        lg
    }

    /// After `forward_multi` of `t` tokens, keep only the first `keep` (1..=t) of them: restores DeltaNet
    /// and PLE state from the checkpoints and truncates the attention KV (GPU and host) and the token list.
    /// Longest verify window that does not straddle an expert-cache refresh (plain decode refreshes
    /// before call k when k % PYRE_QN_XREFILL == 0; a window may only refresh at its first token).
    pub fn spec_max_window(&self) -> usize {
        let Some(xm) = &self.xc else { return usize::MAX };
        let c = xm.lock().unwrap().calls; let every = env_usize("PYRE_QN_XREFILL").unwrap_or(16).max(1);
        (2..=every + 1).find(|j| (c + j) % every == 0).map(|j| j - 1).unwrap_or(every)
    }

    pub fn spec_keep(&self, st: &mut State, t: usize, keep: usize) {
        if keep >= t || t < 2 { return; }
        let keep = keep.max(1); let c = &self.cfg;
        // rejected drafts must not shift the expert-cache schedule or usage counts (greedy identity)
        if let Some(xm) = &self.xc { let mut g = xm.lock().unwrap(); g.calls = g.calls.saturating_sub(t - keep);
            for ti in keep..t { if let Some(v) = st.spec_sel.get(ti) { for &(il, e) in v { g.cnt[il][e] = g.cnt[il][e].saturating_sub(xinc()); } } } }
        let n = st.tokens.len() - t + keep;
        st.tokens.truncate(n);
        if let Some(cb) = &self.cube { if st.gdn_gpu {
            let ch = 2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv; let (cn, sn) = ((c.conv_k - 1) * ch, c.lin_nv * c.lin_dk * c.lin_dv);
            for li in 0..st.gdn.len() { let (a, b) = &st.ck[keep - 1][li]; let (ca, cbh) = &st.gdn[li]; cb.copy(a, cn, 0, ca, cn, 0, cn); cb.copy(b, sn, 0, cbh, sn, 0, sn); }
        } }
        for pi in 0..st.ple_hist.len() { if let Some(h) = st.ple_ck.get(pi).and_then(|v| v.get(keep - 1)) { st.ple_hist[pi] = h.clone(); } }
        let row = c.n_kv * c.hd; let ihd = c.idx_hd; let ir = c.idx_ratio.max(1);
        for ai in 0..st.k.len() {
            if let Some(Some(g)) = st.kvg.get_mut(ai) { g.len = g.len.min(n); }
            st.k[ai].truncate(n * row); st.v[ai].truncate(n * row);
            if ai < st.ik.len() { st.ik[ai].truncate(n * ihd); st.ib[ai].truncate((n / ir) * ihd); }
        }
    }

    /// `layer_cube` for t tokens: the dense half token by token (same kernels, positions start..start+t),
    /// then one routing pass where the CPU pool reads every missed expert once for all of its tokens.
    #[allow(clippy::too_many_arguments)]
    fn layer_cube_multi(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, ai: &mut usize, li: &mut usize, start: usize) {
        let (ai0, li0) = (*ai, *li);
        let dd = match if vgraph_on() { self.lcm_dense_g(il, l, cl, xs, st, li0) } else { None } { Some(x) => x, None => self.lcm_dense(il, l, cl, xs, st, ai0, li0, start, 0, xs.len()) };
        match l.mix { Mixer::Attn { .. } => *ai = ai0 + 1, Mixer::Lin { .. } => *li = li0 + 1 }
        let q4 = std::time::Instant::now(); if prewake() { qn_pool().prewake(); } let rall = self.cube.as_ref().unwrap().read(&dd.0);
        HY[1].fetch_add(q4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        self.lcm_moe(il, l, cl, xs, st, dd, rall, 0);
    }
    /// `lcm_dense` of a DeltaNet layer for a whole verify window (t = 2..4, the k_delta_m path) as a replayed graph.
    /// Same kernels, same order, same arguments as lcm_dense; only the buffers are the graph's own. None = not eligible.
    #[allow(clippy::too_many_arguments)]
    /// The window's residual buffers for the verify graphs: one persistent set per window size, rewritten each round.
    fn vxs_fill(&self, cb: &super::qn_cube::Cube, xh: &[Vec<f32>]) -> Vec<H> {
        let t = xh.len(); let n = xh[0].len();
        let mut m = self.vxs.lock().unwrap();
        let v = m.entry(t).or_insert_with(|| cb.client.memory_persistent_allocation((), |_| (0..t).map(|_| cb.empty(n)).collect()));
        for (h, x) in v.iter().zip(xh) { cb.client.write(h, burn_cubecl::cubecl::bytes::Bytes::from_elems(x.clone())); }
        v.clone()
    }
    fn lcm_dense_g(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[H], st: &mut State, li0: usize) -> Option<(H, H, H)> {
        let t = xs.len(); let c = &self.cfg;
        let gw = cl.gdn.as_ref()?;
        if !matches!(l.mix, Mixer::Lin { .. }) || !st.gdn_gpu || st.pf || !(2..=4).contains(&t) || c.lin_dk != c.lin_dv || st.ck.len() < t - 1
            || !env_var("PYRE_QN_DELTAM").map(|v| v != "0").unwrap_or(true) { return None; }
        let cb = self.cube.as_ref().unwrap(); let (d, hc) = (c.d, c.hc); let w = hc * d;
        let (dar, dfr) = (cl.hca.di.rows, cl.hcf.di.rows); let pr = cl.min.rows; let rn = d + cl.fin.rows; let lra = cl.hca.up.cols;
        let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let core_n = nv * dk; let ch = cl.min.splits[1]; let sp = cl.min.splits.clone();
        let (cs, ss) = st.gdn[li0].clone(); let cn = (c.conv_k - 1) * (2 * nk * dk + nv * c.lin_dv);
        let cka: Vec<(H, H)> = (0..t - 1).map(|ti| st.ck[ti][li0].clone()).collect();
        let q0 = std::time::Instant::now();
        // key on stable ids + offsets: a handle's Debug includes its slice location, which moves when a page is split
        let hk = |h: &H| format!("{:?}/{:?}/{:?};", h.memory.descriptor().id, h.offset_start, h.offset_end);
        let mut key = String::new(); for h in [&cs, &ss].into_iter().chain(cka.iter().flat_map(|(a, b)| [a, b])).chain(xs.iter()) { key.push_str(&hk(h)); }
        let mut gm = self.vg.lock().unwrap();
        // the window's scratch is shared by every layer's graph of this window size (they run one after another on one
        // stream): per-layer buffers pinned ~50 MiB of pool pages per captured graph
        let bt = gm.entry((usize::MAX, t)).or_insert_with(|| cb.client.memory_persistent_allocation((), |_| VGraph { xn: cb.empty(t * w), da: cb.empty(t * dar), mixed: cb.empty(t * d), proj: cb.empty(t * pr),
            core: cb.empty(t * core_n), co: cb.empty(t * ch), o: cb.empty(t * d), df: cb.empty(t * dfr), rd: cb.empty(t * rn), g: None, key: String::new(), keep: vec![] }));
        let (xn, da, mixed, proj, core, co, o, df, rd) = (bt.xn.clone(), bt.da.clone(), bt.mixed.clone(), bt.proj.clone(), bt.core.clone(), bt.co.clone(), bt.o.clone(), bt.df.clone(), bt.rd.clone());
        let b = gm.entry((il, t)).or_insert_with(|| VGraph { xn, da, mixed, proj, core, co, o, df, rd, g: None, key: String::new(), keep: vec![] });
        let at = |h: &H, off: usize| -> H { h.clone().offset_start((off * 4) as u64) };
        let seg = |b: &VGraph| {
            cb.rms_t(xs, &cl.hca.norm, &b.xn, hc, d, c.eps);
            cb.gemv_m(&cl.hca.di, &b.xn, t * w, 0, w, &b.da, t * dar, 0, dar, t);
            if !(gatem() && cb.gate_m(&b.xn, &b.da, &cl.hca.up, &b.mixed, t * d, 0, hc, d, t, w, dar, d)) { for ti in 0..t { cb.gate(&at(&b.xn, ti * w), &at(&b.da, ti * dar), &cl.hca.up, &at(&b.mixed, ti * d), d, 0, hc, d); } }
            cb.gemv_m(&cl.min, &b.mixed, t * d, 0, d, &b.proj, t * pr, 0, pr, t);
            let (mut cks, mut ckm) = (vec![], 0u32);
            for ti in 0..t { let pj = at(&b.proj, ti * pr); cb.conv(&pj, pr, &gw.conv, &cs, &at(&b.co, ti * ch), ch, c.conv_k);
                if ti + 1 < t { let (a, bb) = &cka[ti]; cb.copy(&cs, cn, 0, a, cn, 0, cn); if ti < 3 { ckm |= 1 << ti; } cks.push(Some(bb.clone())); } else { cks.push(None); } }
            cb.delta_m(&b.co, ch, &b.proj, pr, gw, &ss, &b.core, core_n, &cks, ckm, nk, nv, dk, sp[1], sp[2], sp[3], c.eps, t);
            cb.gemv_m(&cl.mout, &b.core, t * core_n, 0, core_n, &b.o, t * d, 0, d, t);
            for ti in 0..t { cb.comb(&xs[ti], &at(&b.o, ti * d), None, None, &at(&b.da, ti * dar), lra, hc, d); } cb.rms_t(xs, &cl.hcf.norm, &b.xn, hc, d, c.eps);
            cb.gemv_m(&cl.hcf.di, &b.xn, t * w, 0, w, &b.df, t * dfr, 0, dfr, t);
            if !(gatem() && cb.gate_m(&b.xn, &b.df, &cl.hcf.up, &b.rd, t * rn, 0, hc, d, t, w, dfr, rn)) { for ti in 0..t { cb.gate(&at(&b.xn, ti * w), &at(&b.df, ti * dfr), &cl.hcf.up, &at(&b.rd, ti * rn), rn, 0, hc, d); } }
            cb.gemv_m(&cl.fin, &b.rd, t * rn, 0, rn, &b.rd, t * rn, d, rn, t);
        };
        if b.g.is_some() && b.key == key { unsafe { b.g.as_ref().unwrap().replay() }.expect("graph replay"); VG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        else {
            b.g = None; b.keep.clear();
            if cb.client.graph_prepare().is_ok() { cb.client.memory_persistent_allocation((), |_| seg(b));
                match cb.client.start_capture() {
                    Ok(()) => { cb.client.memory_persistent_allocation((), |_| seg(b)); match cb.client.stop_capture() { Ok(g) => { b.g = Some(g); b.key = key; if env_var("PYRE_QN_XDEBUG").is_ok() { let n = VG_CAPTURES.load(std::sync::atomic::Ordering::Relaxed); if n < 4 || n % 12 == 0 { let r = cb.client.memory_report(burn_cubecl::cubecl::MemoryScope::CurrentStream); let s = format!("{r:?}"); crate::limits::vlog(&format!("vgraph capture {n}")); eprintln!("vgraph: captured layer {il} T={t} usage {:?} | {}", r.usage(), &s[..s.len().min(1500)]); } } let mut k = vec![cs.clone(), ss.clone()]; k.extend(xs.iter().cloned()); for (a, bb) in &cka { k.push(a.clone()); k.push(bb.clone()); } b.keep = k; VG_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                        Err(e) => { eprintln!("vgraph: layer {il} T={t} capture failed, verify graphs off: {e:?}"); VGRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); } } }
                    Err(e) => { eprintln!("vgraph: layer {il} start failed, verify graphs off: {e:?}"); VGRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); let _ = cb.client.stop_capture(); } }
            } else { seg(b); }
        }
        HY[0].fetch_add(q0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Some((b.rd.clone(), b.df.clone(), b.da.clone()))
    }
    /// The dense (GPU) half of a verify layer for tokens xs (window offset `off` of `ttot`, at positions start..):
    /// everything up to the router / shared-expert projection `rd`, enqueued only. Returns (rd, df, da).
    fn lcm_dense(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, ai0: usize, li0: usize, start: usize, off: usize, ttot: usize) -> (H, H, H) {
        type H = burn_cubecl::cubecl::server::Handle;
        let t = xs.len(); let _ = il;
        let cb = self.cube.as_ref().unwrap(); let c = &self.cfg; let (d, hc) = (c.d, c.hc); let w = hc * d;
        let lra = cl.hca.up.cols; let rn = d + cl.fin.rows; let pr = cl.min.rows;
        let at = |h: &H, off: usize| -> H { h.clone().offset_start((off * 4) as u64) };
        let q0 = std::time::Instant::now();
        let lra = cl.hca.up.cols; let lrf = cl.hcf.up.cols; let rn = d + cl.fin.rows; let pr = cl.min.rows;
        let at = |h: &H, off: usize| -> H { h.clone().offset_start((off * 4) as u64) };
        // attention hyper-connection read + mixer input projection, the big GEMVs batched over the t tokens
        let (dar, dfr) = (cl.hca.di.rows, cl.hcf.di.rows);
        let xn = cb.empty(t * w); let da = cb.empty(t * dar); let mixed = cb.empty(t * d); let proj = cb.empty(t * pr);
        cb.rms_t(xs, &cl.hca.norm, &xn, hc, d, c.eps);
        cb.gemv_m(&cl.hca.di, &xn, t * w, 0, w, &da, t * dar, 0, dar, t);
        if !(gatem() && cb.gate_m(&xn, &da, &cl.hca.up, &mixed, t * d, 0, hc, d, t, w, dar, d)) { for ti in 0..t { cb.gate(&at(&xn, ti * w), &at(&da, ti * dar), &cl.hca.up, &at(&mixed, ti * d), d, 0, hc, d); } }
        cb.gemv_m(&cl.min, &mixed, t * d, 0, d, &proj, t * pr, 0, pr, t);
        // the mixer, token by token (recurrent / causal state)
        let core_n = match l.mix { Mixer::Attn { .. } => c.n_head * c.hd, Mixer::Lin { .. } => c.lin_nv * c.lin_dk };
        let core = cb.empty(t * core_n);
        // PYRE_QN_DELTAM (default on): the DeltaNet recurrence of all t tokens in one k_delta_m launch (the convs
        // and their state checkpoints stay per token, in the same order)
        let mut dm_done = false;
        if let (Mixer::Lin { .. }, Some(gw), true) = (&l.mix, &cl.gdn, st.gdn_gpu) { if t >= 2 && t <= 4 && c.lin_dk == c.lin_dv && env_var("PYRE_QN_DELTAM").map(|v| v != "0").unwrap_or(true) {
            let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let ch = cl.min.splits[1]; let sp = &cl.min.splits;
            let (cs, ss) = st.gdn[li0].clone(); let co_all = cb.empty(t * ch);
            let cn = (c.conv_k - 1) * (2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv);
            let (mut cks, mut ckm) = (vec![], 0u32);
            for ti in 0..t { let pj = at(&proj, ti * pr); cb.conv(&pj, pr, &gw.conv, &cs, &at(&co_all, ti * ch), ch, c.conv_k);
                if off + ti + 1 < ttot && !st.pf { let (a, b) = &st.ck[off + ti][li0]; cb.copy(&cs, cn, 0, a, cn, 0, cn); if ti < 3 { ckm |= 1 << ti; } cks.push(Some(b.clone())); } else { cks.push(None); } }
            cb.delta_m(&co_all, ch, &proj, pr, gw, &ss, &core, core_n, &cks, ckm, nk, nv, dk, sp[1], sp[2], sp[3], c.eps, t);
            dm_done = true;
        } }
        for ti in 0..t {
            let pos = start + ti; let pj = at(&proj, ti * pr);
            if dm_done { let _ = (pos, pj); }
            else if let (Some(gw), true) = (&cl.gdn, st.gdn_gpu) {
                let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let ch = cl.min.splits[1]; let sp = &cl.min.splits;
                let (cs, ss) = st.gdn[li0].clone();
                let co = cb.empty(ch);
                cb.conv(&pj, pr, &gw.conv, &cs, &co, ch, c.conv_k);
                cb.delta(&co, ch, &pj, pr, gw, &ss, &at(&core, ti * core_n), nk, nv, dk, sp[1], sp[2], sp[3], c.eps);
                if off + ti + 1 < ttot && !st.pf { let cn = (c.conv_k - 1) * (2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv); let sn = nv * dk * c.lin_dv;
                    let (a, b) = &st.ck[off + ti][li0]; cb.copy(&cs, cn, 0, a, cn, 0, cn); cb.copy(&ss, sn, 0, b, sn, 0, sn); }
            } else if let (Mixer::Attn { idx, .. }, Some((qnh, knh)), true) = (&l.mix, &cl.an, attn_gpu(c) && pos + 1 <= self.dense_n()) {
                let (nh, nkv, hd, half) = (c.n_head, c.n_kv, c.hd, c.n_rot / 2); let sp = &cl.min.splits;
                self.kv_push(cb, ai0, st, pos + 1);
                let g = st.kvg[ai0].as_mut().unwrap(); assert_eq!(g.len, pos, "gpu kv out of step");
                let cs: Vec<f32> = (0..half).map(|i| (pos as f32 * self.rope_inv[i]).cos()).chain((0..half).map(|i| (pos as f32 * self.rope_inv[i]).sin())).collect();
                let csb = cb.buf(&cs);
                cb.attn(&pj, pr, sp[1], sp[2], qnh, knh, &csb, &g.k, &g.v, g.cap, &at(&core, ti * core_n), nh, nkv, hd, half, pos, c.eps);
                if idx.is_some() && sp.len() > 5 { let ihd = c.idx_hd; cb.copy(&proj, t * pr, ti * pr + sp[4], &g.ik, g.cap * ihd, pos * ihd, ihd); }
                g.len = pos + 1;
            } else {
                let pv = cb.read(&proj); let pv = &pv[ti * pr..(ti + 1) * pr];
                let parts: Vec<Vec<f32>> = (0..cl.min.splits.len() - 1).map(|i| pv[cl.min.splits[i]..cl.min.splits[i + 1]].to_vec()).collect();
                let cv = match l.mix { Mixer::Attn { .. } => self.attn_core(l, ai0, st, parts, pos, 1), Mixer::Lin { .. } => self.linear_core(l, li0, st, parts, 1) };
                let cbuf = cb.buf(&cv); cb.copy(&cbuf, core_n, 0, &core, t * core_n, ti * core_n, core_n);
            }
        }
        let o = cb.empty(t * d);
        cb.gemv_m(&cl.mout, &core, t * core_n, 0, core_n, &o, t * d, 0, d, t);
        // FFN hyper-connection read + router / shared expert projection, batched
        let df = cb.empty(t * dfr); let rd = cb.empty(t * rn);
        for ti in 0..t { cb.comb(&xs[ti], &at(&o, ti * d), None, None, &at(&da, ti * dar), lra, hc, d); } cb.rms_t(xs, &cl.hcf.norm, &xn, hc, d, c.eps);
        cb.gemv_m(&cl.hcf.di, &xn, t * w, 0, w, &df, t * dfr, 0, dfr, t);
        if !(gatem() && cb.gate_m(&xn, &df, &cl.hcf.up, &rd, t * rn, 0, hc, d, t, w, dfr, rn)) { for ti in 0..t { cb.gate(&at(&xn, ti * w), &at(&df, ti * dfr), &cl.hcf.up, &at(&rd, ti * rn), rn, 0, hc, d); } }
        cb.gemv_m(&cl.fin, &rd, t * rn, 0, rn, &rd, t * rn, d, rn, t);
        HY[0].fetch_add(q0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        (rd, df, da)
    }
    /// The expert half of a verify layer: routing from `rall` (rd read back), cached experts on the GPU beside the
    /// CPU misses, and the residual update of xs.
    fn lcm_moe(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, dd: (H, H, H), rall: Vec<f32>, off: usize) {
        type H = burn_cubecl::cubecl::server::Handle;
        let t = xs.len();
        let cb = self.cube.as_ref().unwrap(); let c = &self.cfg; let (d, hc, ne) = (c.d, c.hc, c.n_exp);
        let lrf = cl.hcf.up.cols; let rn = d + cl.fin.rows; let dfr = cl.hcf.di.rows;
        let at = |h: &H, off: usize| -> H { h.clone().offset_start((off * 4) as u64) };
        let (rd, df, _da) = dd;
        let rvs: Vec<Vec<f32>> = (0..t).map(|ti| rall[ti * rn..(ti + 1) * rn].to_vec()).collect();
        let dfs: Vec<H> = (0..t).map(|ti| at(&df, ti * dfr)).collect();
        let rds: Vec<H> = (0..t).map(|ti| at(&rd, ti * rn)).collect();
        let q5 = std::time::Instant::now();
        let sf = l.sh_gate.rows;
        let mut yr = vec![0f32; t * d];
        let mut ygs: Vec<Option<burn_cubecl::cubecl::server::Handle>> = vec![None; t];
        let mut shs: Vec<burn_cubecl::cubecl::server::Handle> = Vec::with_capacity(t);
        let (hb_all, sh_all) = (cb.empty(t * sf), cb.empty(t * d));
        let xin: Vec<f32> = rvs.iter().flat_map(|r| r[..d].iter().copied()).collect();
        if let Some(xm) = &self.xc {
            let mut xcg = xm.lock().unwrap();
            let (mut ons, mut offs) = (Vec::with_capacity(t), Vec::with_capacity(t));
            for (ti, rv) in rvs.iter().enumerate() {
                let sel = self.route_sel(il, &rv[d..d + ne], Some(&xcg));
                for &(e, _) in &sel { xcg.cnt[il][e] += xinc(); if let Some(v) = st.spec_sel.get_mut(off + ti) { v.push((il, e)); } }
                xcg.total += sel.len() as u64;
                let (on, mut off): (Vec<(usize, f32)>, Vec<(usize, f32)>) = sel.into_iter().partition(|(e, _)| xcg.map.contains_key(&(il, *e)));
            tail_skip(&mut off, &on);
            // Strata-style PCIe share: the first few misses are copied into the stream slots and run on the GPU
            // with the cached ones, while the CPU pool takes the rest
            let tys0 = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
            let _ = tys0;
                xcg.hits += on.len() as u64; crate::metrics::expert_cache(on.len() as u64, off.len() as u64); ons.push(on); offs.push(off);
            }
            // Strata's verify dispatch (expert_pool_dispatch_multi): the distinct misses of the whole window, and a
            // PCIe share of them (here the ones most tokens use) uploaded once and run on the GPU for every token
            let mut sins: Vec<(usize, usize)> = vec![];
            let tyw = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
            if xshare() > 0.0 && !xcg.stream.is_empty() && xfast() && tyw.0 == tyw.1 && tyw.0 <= 1 && tyw.2 == 3 && xcg.arena.len() <= 2 {
                let mut cm: std::collections::HashMap<usize, usize> = Default::default();
                // with the pinned arena only experts already packed there fly (as Strata streams only pinned blobs): any other would
                // be repacked and staged on this thread, which costs more than the CPU pool needs for it
                let ina = |e: usize| !xarena_pin() || self.xpk.lock().unwrap().map.contains_key(&(il, e));
                for o in &offs { for &(e, _) in o { if ina(e) { *cm.entry(e).or_default() += 1; } } }
                let k = ((cm.len() as f32 * xshare()).round() as usize).min(xcg.stream.len());
                if k > 0 {
                    let mut v: Vec<(usize, usize)> = cm.into_iter().collect(); v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    let qu = std::time::Instant::now(); for (j, &(e, _)) in v[..k].iter().enumerate() { let sl = xcg.stream[j]; let pk = self.xpacked(il, e, tyw);
                        let mut ent = cb.xupload_pk(&xcg.arena[sl / xcg.per], xcg.ar_bytes, sl % xcg.per, xcg.slot, &pk.0, pk.1, pk.2); ent.arena = sl / xcg.per; ent.slot = sl;
                        xcg.map.insert((il, e), ent); sins.push((il, e)); }
                    SD[1].fetch_add(qu.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); let set: std::collections::HashSet<usize> = v[..k].iter().map(|x| x.0).collect();
                    for ti in 0..t { let (mv, keep): (Vec<(usize, f32)>, Vec<(usize, f32)>) = offs[ti].drain(..).partition(|(e, _)| set.contains(e)); offs[ti] = keep; ons[ti].extend(mv); }
                }
            }
            let xcr = &*xcg;
            let mut gpu = || {
                for ti in 0..t {
                    let (rd, on) = (&rds[ti], &ons[ti]);
                    let tyf = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                    let fast = !on.is_empty() && xfast() && tyf.0 == tyf.1 && tyf.0 <= 1 && tyf.2 == 3 && {
                        let y0 = cb.empty(d);
                        let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect();
                        let mut first = true; let mut ok = true;
                        if xcr.arena.len() <= 2 { let q = std::time::Instant::now(); ok = cb.xrun_fast_to(&xcr.tabs, &xcr.arena, xcr.ar_bytes, &ents, rd, rn, d, c.ff_exp, tyf, &xcr.scr, false, &y0); SD[0].fetch_add(q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } else {
                        for ar in 0..xcr.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ar).cloned().collect(); if grp.is_empty() { continue; }
                            ok &= cb.xrun_fast_to(&xcr.tabs, std::slice::from_ref(&xcr.arena[ar]), xcr.ar_bytes, &grp, rd, rn, d, c.ff_exp, tyf, &xcr.scr, !first, &y0); first = false; } }
                        if ok { ygs[ti] = Some(y0); } ok };
                    if !on.is_empty() && !fast {
                        let xb = cb.empty(d); let y0 = cb.empty(d);
                        cb.q8copy(rd, rn, 0, &xb, &y0, d);
                        let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
                        let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect();
                        for ar in 0..xcr.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ar).cloned().collect(); if grp.is_empty() { continue; }
                            let hx = cb.empty(c.ff_exp * grp.len()); cb.xrun_b(&xcr.tabs, &xcr.arena[ar], xcr.ar_bytes, &grp, &xb, d, c.ff_exp, tys, &hx, &y0); }
                        ygs[ti] = Some(y0);
                    }
                    cb.silu_mul(rd, rn, d + ne, d + ne + sf, &at(&hb_all, ti * sf), sf);
                }
                cb.gemv_m(&cl.sdown, &hb_all, t * sf, 0, sf, &sh_all, t * d, 0, d, t);
                for ti in 0..t { shs.push(at(&sh_all, ti * d)); }
                let qf = std::time::Instant::now(); if kick() { let _ = cb.client.flush(); } SD[2].fetch_add(qf.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            };
            // union of the missed experts; each token keeps its own order for the accumulation
            let mut lists: Vec<(usize, Vec<(usize, f32)>)> = vec![]; let mut pos_of: std::collections::HashMap<usize, usize> = Default::default();
            let mut ord: Vec<Vec<(usize, usize)>> = vec![vec![]; t];
            for (ti, off) in offs.iter().enumerate() { for &(e, wt) in off {
                let wi = *pos_of.entry(e).or_insert_with(|| { lists.push((e, vec![])); lists.len() - 1 });
                ord[ti].push((wi, lists[wi].1.len())); lists[wi].1.push((ti, wt)); } }
            if !lists.is_empty() {
                let work: Vec<(usize, &Vec<(usize, f32)>)> = lists.iter().map(|(e, v)| (*e, v)).collect();
                MISS[0].fetch_add(lists.len() as u64, std::sync::atomic::Ordering::Relaxed); MISS[1].fetch_add(lists.iter().map(|x| x.1.len() as u64).sum(), std::sync::atomic::Ordering::Relaxed);
                MISS[2].fetch_add(ons.iter().map(|o| o.len() as u64).sum(), std::sync::atomic::Ordering::Relaxed);
                let tr = std::time::Instant::now();
                self.routed_pool_ord(l, &xin, &work, t, &mut yr, &mut gpu, Some(&ord));
                { let rns = tr.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
            } else { gpu(); }
            for k in &sins { xcg.map.remove(k); }
            drop(xcg);
        } else {
            for ti in 0..t {
                let hb = cb.empty(sf); let sh = cb.empty(d);
                cb.silu_mul(&rds[ti], rn, d + ne, d + ne + sf, &hb, sf);
                cb.gemv(&cl.sdown, &hb, sf, 0, &sh, d, 0); shs.push(sh);
                self.routed(l, &rvs[ti][..d], &rvs[ti][d..d + ne], 1, &mut yr[ti * d..(ti + 1) * d], false);
            }
        }
        let q6 = std::time::Instant::now();
        let yball = cb.buf(&yr);
        for ti in 0..t {
            let yb = at(&yball, ti * d);
            cb.comb3(&xs[ti], &shs[ti], Some(&yb), ygs[ti].as_ref(), Some((&rds[ti], rn, d + ne + 2 * sf)), &dfs[ti], lrf, hc, d);
        }
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
        HY[0].fetch_add(ns(q6, std::time::Instant::now()), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed);
    }
}


/// PYRE_QN_TAIL_SKIP=R (opt-in, changes outputs; 0 = off, the default): a routed expert that missed the GPU cache
/// and ranks R or lower (0-based, by routing weight among the token's top-k) is not computed. Per token, so a
/// verify window decides exactly as plain decode does. Idea from Strata's STRATA_ROUTE_TAIL_SKIP (rank 7 there,
/// applied per verify window).
fn tail_skip(off: &mut Vec<(usize, f32)>, on: &[(usize, f32)]) {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let r = *R.get_or_init(|| env_usize("PYRE_QN_TAIL_SKIP").unwrap_or(if crate::spec::fast() { 7 } else { 0 }));
    if r == 0 || off.is_empty() { return; }
    let mut ws: Vec<f32> = on.iter().chain(off.iter()).map(|x| x.1).collect();
    if r >= ws.len() { return; }
    ws.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cut = ws[r - 1]; // keep experts strictly above the rank-(r-1) weight or equal to it
    off.retain(|x| x.1 >= cut);
}
static BUSY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BUSYN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TL: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static SIDE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SD: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static MISS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static HEAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WV: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static HY: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
static PH: [std::sync::atomic::AtomicU64; 1] = [std::sync::atomic::AtomicU64::new(0)];
static ROUTED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// softmax over the top k router logits (same order of operations as `routed`).
/// Software prefetch of one weight row (expert rows are short, so hardware prefetch ramps too late).
#[inline] fn prefetch(b: &[u8]) {
    #[cfg(target_arch = "x86_64")] unsafe { use std::arch::x86_64::*; let mut o = 0; while o < b.len() { _mm_prefetch::<_MM_HINT_T0>(b.as_ptr().add(o) as *const i8); o += 64; } }
}
fn top_k(r: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..r.len()).collect();
    idx.select_nth_unstable_by(k, |a, b| r[*b].partial_cmp(&r[*a]).unwrap());
    let top = &idx[..k]; let mx = top.iter().map(|&e| r[e]).fold(f32::NEG_INFINITY, f32::max);
    let ws: Vec<f32> = top.iter().map(|&e| (r[e] - mx).exp()).collect(); let sum: f32 = ws.iter().sum();
    top.iter().zip(ws).map(|(&e, w)| (e, w / sum)).collect()
}

/// PYRE_QN_PREWAKE (default on; =0 off): wake the expert pool just before each layer's router readback.
/// PYRE_QN_PAIR=1 (opt-in): gate and up rows of a CPU expert in one interleaved IQ2 kernel (same results)
fn pair_on() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_PAIR").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_GATEM=0 turns off the multi-token hyper-connection gate kernel in verify windows
fn gatem() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_GATEM").map(|v| v != "0").unwrap_or(true)) }
fn prewake() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_PREWAKE").map(|v| v != "0").unwrap_or(true)) }
fn qn_pool() -> &'static super::spin_pool::Pool {
    static P: std::sync::OnceLock<super::spin_pool::Pool> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        // default: one worker per fast physical core, pinned (PYRE_QN_PIN="0,2,..." overrides, "none" disables)
        let auto = super::spin_pool::fast_cores();
        let pin: Vec<usize> = match env_var("PYRE_QN_PIN").ok().as_deref() { Some("none") => vec![], Some(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(), None => auto.clone() };
        let n = env_usize("PYRE_QN_SPIN_THREADS").unwrap_or(if pin.len() >= 4 { pin.len() } else { rayon::current_num_threads() });
        eprintln!("expert pool: {n} workers, pinned to {:?}", pin);
        { // PYRE_QN_ETAIL: a job's last N tasks go to P-core workers only (E-cores ~2.3x slower would finish last)
            let nf = if pin == auto { super::spin_pool::n_fast() } else { usize::MAX }; let et = env_usize("PYRE_QN_ETAIL").unwrap_or(0);
            eprintln!("expert pool: {} fast workers, E-core tail reserve {et}", nf.min(n));
            super::spin_pool::Pool::new_hybrid(n, std::time::Duration::from_micros(env_usize("PYRE_QN_SPIN_US").unwrap_or(100) as u64), &pin, nf, et) }
    })
}

fn xstream() -> usize { static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *K.get_or_init(|| env_usize("PYRE_QN_XSTREAM").unwrap_or(if xshare() > 0.0 { 8 } else { 0 })) }
fn env_usize(k: &str) -> Option<usize> { env_var(k).ok().and_then(|v| v.parse().ok()) }
/// `std::env::var` from a snapshot of the PYRE_* variables taken on first use. The decode path reads
/// settings per layer and per pool task; the real call takes the process environment lock (8 pool
/// threads contended on it) and a syscall on Windows.
fn env_var(k: &str) -> Result<String, std::env::VarError> {
    static ENV: std::sync::OnceLock<std::collections::HashMap<String, String>> = std::sync::OnceLock::new();
    if !k.starts_with("PYRE_") { return std::env::var(k); }
    ENV.get_or_init(|| std::env::vars().filter(|(k, _)| k.starts_with("PYRE_")).collect()).get(k).cloned().ok_or(std::env::VarError::NotPresent)
}
