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
/// PYRE_QN_XARENA_PERSIST=1: the expert arenas in CubeCL's persistent pool, so the dynamic pool's page size is not grown to the arena size (else every later new page is ~2.7 GB)
fn xarena_persist() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XARENA_PERSIST").map(|v| v != "0").unwrap_or(true)) }
fn xmeta1() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XMETA1").map(|v| v != "0").unwrap_or(true)) }
fn xcopys() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XCOPYS").map(|v| v == "1").unwrap_or(false)) }
fn xarena_pin() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XARENA_PIN").map(|v| v == "1").unwrap_or(false)) }
static RTN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); static RTN2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn xshare() -> f32 { static K: std::sync::OnceLock<f32> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XSHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)) }
pub struct XCache { tabs: super::qn_cube::XTabs, arena: Vec<burn_cubecl::cubecl::server::Handle>, ar_bytes: usize, per: usize, slot: usize, free: Vec<usize>, map: std::collections::HashMap<(usize, usize), super::qn_cube::XEnt>, cnt: Vec<Vec<u32>>, stream: Vec<usize>, used: usize, budget: usize, calls: usize, pub hits: u64, pub total: u64, scr: super::qn_cube::XScr, sloc: Vec<(usize, usize, usize)> }
impl XCache {
    /// slot -> (arena, byte offset, capacity): uniform slots, or the PYRE_QN_XCLASS per-class table
    fn loc(&self, sl: usize) -> (usize, usize, usize) { if self.sloc.is_empty() { (sl / self.per, (sl % self.per) * self.slot, self.slot) } else { self.sloc[sl] } }
    fn cap(&self, sl: usize) -> usize { self.loc(sl).2 }
    fn up(&self, cb: &super::qn_cube::Cube, sl: usize, v: &burn_cubecl::cubecl::bytes::Bytes, uo: usize, dof: usize) -> super::qn_cube::XEnt { let (a, b, c) = self.loc(sl); cb.xupload_at(&self.arena[a], self.ar_bytes, b, c, v, uo, dof) }
}
/// PYRE_QN_XCLASS=1 (opt-in): cache slots sized per expert-size class (equal slots per layer), instead of every slot the largest expert
fn xclass() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XCLASS").map(|v| v == "1").unwrap_or(false)) }

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
    // PYRE_QN_XPROFILL=1 (opt-in): the profile only fills slots the run's own routing leaves empty (a routed expert
    // always outranks a profile-only one; the profile orders the rest). Strata pre-fills every slot from its ranked
    // profile at load (Strata, MIT: include/strata/core/expert_cache.hpp read_expert_profile; generate.cpp profile fill);
    // here the prompt's own routing stays first because it measured better than a static profile (moe.md g1).
    if xprofill() { return ((xc.cnt[il][e] as u64) << 24) | ((pr * 1000.0) as u64).min((1 << 24) - 1); }
    xc.cnt[il][e] as u64 * 1000 + (pr * 1000.0) as u64
}
fn xprofill() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_XPROFILL").map(|v| v == "1").unwrap_or(false)) }
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
#[allow(clippy::type_complexity)]
static VG_STATE: std::sync::Mutex<(Vec<(burn_cubecl::cubecl::server::Handle, burn_cubecl::cubecl::server::Handle)>, Vec<Vec<(burn_cubecl::cubecl::server::Handle, burn_cubecl::cubecl::server::Handle)>>)> = std::sync::Mutex::new((Vec::new(), Vec::new()));
/// PYRE_QN_VGRAPH attention layers: captured graphs of the pre-mixer and post-mixer halves, per (part, layer, T)
#[allow(clippy::type_complexity)]
static VG2: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (Option<burn_cubecl::cubecl::client::Graph>, String)>>> = std::sync::LazyLock::new(Default::default);
/// PYRE_MTP_GPUSTEP (default on with PYRE_MTP_GPU, =0 off): the drafter's per-token step on the GPU
fn mtp_gpustep() -> bool { env_var("PYRE_MTP_GPUSTEP").map(|v| v != "0").unwrap_or(true) }
fn vg_run(cb: &super::qn_cube::Cube, slot: (usize, usize, usize), key: &str, f: &(dyn Fn() + Sync)) {
    // one graph per distinct input set too: prewarm T=5..7 windows run as 4+rest groups on slices of other buffers
    let lt = (slot.1, slot.2);
    let slot = (slot.0 + 4 * { use std::hash::{Hash, Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); key.hash(&mut h); (h.finish() as usize) & 0xffff_ffff }, slot.1, slot.2);
    let mut m = VG2.lock().unwrap(); let e = m.entry(slot).or_insert((None, String::new()));
    if e.0.is_some() && e.1 == key { unsafe { e.0.as_ref().unwrap().replay() }.expect("graph replay"); VG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); return; }
    if !e.1.is_empty() && env_var("PYRE_QN_XDEBUG").is_ok() { static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0); if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 { eprintln!("vgraph2 recapture {slot:?}: {} -> {}", e.1, key); } }
    if gate_open() || xg_open() { VG_EAGER.fetch_add(1, std::sync::atomic::Ordering::Relaxed); VG_SKIP.lock().unwrap().insert(lt); if env_var("PYRE_QN_XDEBUG").is_ok() { eprintln!("vg eager behind gate: slot {:?} had {}", slot, if e.0.is_some() { "other key" } else { "none" }); } f(); return; }
    e.0 = None; if VGRAPH_OFF.load(std::sync::atomic::Ordering::Relaxed) || cb.client.graph_prepare().is_err() { f(); return; }
    cb.client.memory_persistent_allocation((), |_| f());
    match cb.client.start_capture() { Ok(()) => { cb.client.memory_persistent_allocation((), |_| f()); match cb.client.stop_capture() { Ok(gr) => { e.0 = Some(gr); e.1 = key.to_string(); VG_SKIP.lock().unwrap().remove(&lt); VG_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed); } Err(er) => { eprintln!("vgraph2: capture failed, off: {er:?}"); VGRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); } } }
        Err(er) => { eprintln!("vgraph2: start failed, off: {er:?}"); VGRAPH_OFF.store(true, std::sync::atomic::Ordering::Relaxed); let _ = cb.client.stop_capture(); } }
}
pub static VG_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// PIPE: verify-graph segments run eagerly because a host gate was open and no matching graph existed
pub static VG_EAGER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// PIPE: (layer, T) whose DeltaNet verify graph was missing behind a gate; the continuation skips that layer until a
/// normal (ungated) pass has captured it, so it replays behind later gates instead of always running eagerly
static VG_SKIP: std::sync::Mutex<std::collections::BTreeSet<(usize, usize)>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
// no explicit flush after enqueueing the next wave group: the device thread launches queued work anyway, and the blocking flush cost the main thread ~21 ms per 5-token round (PYRE_QN_WFLUSH=1 restores it)
fn wflush() -> bool { static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *G.get_or_init(|| env_var("PYRE_QN_WFLUSH").map(|v| v == "1").unwrap_or(false)) }
fn vgraph_on() -> bool { static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *G.get_or_init(|| env_var("PYRE_QN_VGRAPH").map(|v| v != "0").unwrap_or(true)) && !VGRAPH_OFF.load(std::sync::atomic::Ordering::Relaxed) }
pub struct LGraph { xn: H, da: H, mixed: H, proj: H, o: H, co: H, ob: H, df: H, rd: H, g: Option<burn_cubecl::cubecl::client::Graph>, key: String, keep: Vec<H> }
type H = burn_cubecl::cubecl::server::Handle;
static GRAPH_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn graph_on() -> bool { static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *G.get_or_init(|| env_var("PYRE_QN_GRAPH").map(|v| v == "1").unwrap_or(false)) && !GRAPH_OFF.load(std::sync::atomic::Ordering::Relaxed) }
pub struct Model { mtp_gkv: std::sync::Mutex<Option<super::mtp::GRing>>, vg: std::sync::Mutex<std::collections::HashMap<(usize, usize), VGraph>>, vxs: std::sync::Mutex<std::collections::HashMap<usize, Vec<H>>>, mtp_headc: std::sync::OnceLock<Option<super::qn_cube::CMat>>, mtp_dense: std::sync::OnceLock<std::collections::HashMap<usize, super::qn_cube::CMat>>, xpk: std::sync::Mutex<XPk>, xper: std::sync::Mutex<Option<H>>, lg: std::sync::Mutex<std::collections::HashMap<usize, LGraph>>, headc: Option<super::qn_cube::CMat>, xc: Option<std::sync::Mutex<XCache>>, cube: Option<super::qn_cube::Cube>, cl: Vec<Option<super::qn_cube::CLayer>>, pub gpu: Option<super::qn_gpu::Gpu>, gmats: std::collections::HashMap<usize, super::qn_gpu::GMat>, gl: Vec<Option<super::qn_gpu::GLayer>>, pub cfg: Cfg, pub g: Gguf, embd: QMat, out: QMat, head: Hc, ple_tab: Option<QMat>, layers: Vec<Layer>, rope_inv: Vec<f32>, mtp: Option<super::mtp::Mtp> }

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
// PYRE_QN_KICKQ=1: the kick hands the queue to the device thread without waiting for it to drain
fn kickq() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_KICKQ").map(|v| v == "1").unwrap_or(false)) }
// PYRE_QN_CONVM=0: per-token k_conv + k_copy in the verify window instead of one k_conv_m
fn convm() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_CONVM").map(|v| v != "0").unwrap_or(true)) }
// PYRE_QN_COMBM=0: per-token k_comb after the verify mixer instead of one k_comb_m
/// PYRE_QN_PIPE=1 (opt-in): Strata-style verify pipeline, stream-ordered version (no device-side spinning).
/// Design after Strata (github.com/Niko1221/Strata, MIT, verify.cpp wait_flag/copy_rows_from_mapped); reimplemented.
/// Per MoE layer the host enqueues the GPU hits, a host gate (cuLaunchHostFunc that waits, bounded, for the CPU
/// expert pool), a pinned->device copy of the CPU rows, the combine and the NEXT layer's dense half, all while the
/// CPU pool runs; the GPU then starts the next layer as soon as the CPU rows land instead of after a host round trip.
fn pipe_on() -> bool { static P: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *P.get_or_init(|| env_usize("PYRE_QN_PIPE").unwrap_or(0) as u32) >= 1 && !PIPE_ERR.load(std::sync::atomic::Ordering::Relaxed) }
thread_local! { static GATE_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// True while this thread has a PIPE host gate queued and not yet opened: graph capture (graph_prepare drains/syncs) must not happen then, so a missing graph runs eagerly instead.
fn gate_open() -> bool { GATE_OPEN.with(|g| g.get()) }
static PIPE_ERR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Host-side latch the stream-ordered gate waits on (bounded: PYRE_QN_PIPE_TMO_MS, default 2000).
struct PGate { m: std::sync::Mutex<bool>, cv: std::sync::Condvar, tgt: std::sync::Mutex<Option<(usize, u32)>> }
/// PYRE_QN_PIPE=2: the gate is a bounded device wait (pyre_wait_flag kernel, __nanosleep backoff, %globaltimer
/// timeout PYRE_QN_PIPE_SPIN_US default 100000 us, capped at 200 ms in cubecl) on a mapped host u32 the host bumps.
fn pipe2() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_usize("PYRE_QN_PIPE").unwrap_or(0) >= 2) }
static PIPE_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// (flag, err) mapped host words; err is checked after each window.
fn pipe_flag(cb: &super::qn_cube::Cube) -> (usize, usize) { static F: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *F.get_or_init(|| { let p = cb.client.pyre_host_pinned(4096).expect("pinned flag"); unsafe { std::ptr::write_bytes(p as *mut u8, 0, 4096); } (p, p + 64) }) }
fn pipe_err_check() { if let Some(&(_, er)) = PIPE_FLAG_ERR.get() { if unsafe { std::ptr::read_volatile(er as *const u32) } != 0 { if !PIPE_ERR.swap(true, std::sync::atomic::Ordering::Relaxed) { eprintln!("PYRE_QN_PIPE=2: device wait timed out; pipeline disabled, a window's output is invalid"); } } } }
static PIPE_FLAG_ERR: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
impl PGate {
    fn open(&self) { if let Some((fl, tg)) = self.tgt.lock().unwrap().take() { unsafe { (*(fl as *const std::sync::atomic::AtomicU32)).store(tg, std::sync::atomic::Ordering::Release); } } *self.m.lock().unwrap() = true; self.cv.notify_all(); }
    fn wait(&self) { let tmo = std::time::Duration::from_millis(env_usize("PYRE_QN_PIPE_TMO_MS").unwrap_or(2000) as u64); let dl = std::time::Instant::now() + tmo;
        let mut g = self.m.lock().unwrap(); while !*g { let now = std::time::Instant::now(); if now >= dl { PIPE_ERR.store(true, std::sync::atomic::Ordering::Relaxed); eprintln!("PYRE_QN_PIPE: host gate timed out ({tmo:?}); pipeline disabled, this window's output is invalid"); return; } g = self.cv.wait_timeout(g, dl - now).unwrap().0; } }
}
/// Opens the gate on drop (also on unwind), so the stream can never stay held by a lost signal.
struct PGateOpen(std::sync::Arc<PGate>);
impl Drop for PGateOpen { fn drop(&mut self) { self.0.open(); } }
/// Ring of page-locked host slots for the CPU rows (8 slots: a slot is rewritten only after a later blocking routing read, i.e. after its copy ran).
fn pipe_slot(cb: &super::qn_cube::Cube, bytes: usize) -> Option<usize> {
    static R: std::sync::Mutex<(Vec<usize>, usize, usize)> = std::sync::Mutex::new((Vec::new(), 0, 0));
    let mut g = R.lock().unwrap();
    if g.0.is_empty() { let sz = bytes.max(1 << 16) * 2; for _ in 0..8 { match cb.client.pyre_host_pinned(sz) { Ok(p) => g.0.push(p), Err(e) => { eprintln!("PYRE_QN_PIPE: pinned alloc failed: {e}"); PIPE_ERR.store(true, std::sync::atomic::Ordering::Relaxed); return None; } } } g.2 = sz; }
    if bytes > g.2 { return None; }
    let i = g.1; g.1 = (i + 1) % g.0.len(); Some(g.0[i])
}
fn ns0(a: std::time::Instant, b: std::time::Instant) -> u64 { (b - a).as_nanos() as u64 }
/// PYRE_QN_XFUSE (default on; =0 off): fewer small launches in the verify window, bit-identical: rsum_ex + comb3_m as one
/// k_rsc_m behind the XGATE wait, and the DeltaNet verify graph's per-token conv/copy and comb as one k_conv_m / k_comb_m
/// PYRE_QN_XZT (XEXACT, default on under XZC bit 1; =0 off): the per-layer expert tables (hit meta, counts, routing table)
/// go to the GPU by a stream-ordered kernel reading a mapped pinned ring slot (pyre_wait_copy_from_mapped without a flag,
/// own never-set err word) instead of cb.hup's H2D copies, so the hit kernels do not wait behind copy-engine switches.
/// Ring of 32 x 16 KiB slots: a slot is rewritten only ~10 layers later, after rd doorbells that follow its consumer.
fn xzt() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| xzc() & 1 != 0 && env_var("PYRE_QN_XZT").map(|v| v != "0").unwrap_or(true)) }
static XZT_ERR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// set by cubecl-pipe's pyre_wait_copy_from_mapped_async on a launch error (that window's output is invalid; XZT off after)
static XZT_FAIL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn xzt_check() -> bool { if XZT_FAIL.load(std::sync::atomic::Ordering::Relaxed) != 0 { if !XZT_ERR.swap(true, std::sync::atomic::Ordering::Relaxed) { eprintln!("PYRE_QN_XZT: async zero-copy launch failed; table/row zero-copy off for the rest of the run (that window's output is invalid)"); } false } else { true } }
fn xt_up(cb: &super::qn_cube::Cube, w: &[u32]) -> H {
    const SLOT: usize = 16 << 10; const NS: usize = 32;
    if xzt() && xzt_check() && !XZT_ERR.load(std::sync::atomic::Ordering::Relaxed) && !w.is_empty() && w.len() * 4 <= SLOT {
        static R: std::sync::Mutex<(Vec<usize>, usize, usize)> = std::sync::Mutex::new((Vec::new(), 0, 0));
        let mut g = R.lock().unwrap();
        if g.0.is_empty() { match (cb.client.pyre_host_pinned(4096), cb.client.pyre_host_pinned(SLOT * NS)) {
            (Ok(e), Ok(b)) => { unsafe { std::ptr::write_bytes(e as *mut u8, 0, 4096); } g.2 = e; for i in 0..NS { g.0.push(b + i * SLOT); } }
            _ => { eprintln!("PYRE_QN_XZT: pinned alloc failed; table zero-copy off"); XZT_ERR.store(true, std::sync::atomic::Ordering::Relaxed); } } }
        if !g.0.is_empty() {
            let i = g.1; g.1 = (i + 1) % NS; let (sl, er) = (g.0[i], g.2); drop(g);
            let n4 = w.len().div_ceil(4) * 4;
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), sl as *mut u32, w.len()); std::ptr::write_bytes((sl as *mut u32).add(w.len()), 0, n4 - w.len()); }
            let h = cb.empty(n4);
            // non-blocking submit (sizes/alignment are fixed here: 16-B multiple, page-aligned ring); a launch error sets XZT_FAIL
            unsafe { cb.client.pyre_wait_copy_from_mapped_async(0, 0, 0, er, sl, n4 * 4, &h, XZT_FAIL.as_ptr() as usize); }
            XZ_N[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed); return h;
        }
    }
    cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(w))
}
/// PYRE_QN_XBAL (XEXACT): per-(layer, T) balance of the CPU pool vs the GPU hit path. The host's wait for layer l+1's rd
/// (queued behind layer l's gate) minus its per-(layer, T) minimum is how late layer l's GPU side was (hits finishing after
/// the pool; when the pool is the critical side the GPU reaches the gate first and the wait is ~the dense chain = the minimum).
/// =1: when that excess is > PYRE_QN_XBAL_HI_US (40) layer l moves one more GPU hit expert (fewest tokens first) to the CPU
/// pool (all experts are in RAM; the exact path is placement-invariant); after PYRE_QN_XBAL_N (4) windows with excess
/// < PYRE_QN_XBAL_LO_US (10) it gives one back. PYRE_QN_XLSTAT=1: per-layer table (pool us, excess us, GPU-critical share).
fn xbal() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XBAL").map(|v| v == "1").unwrap_or(false)) }
fn xlstat() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XLSTAT").map(|v| v == "1").unwrap_or(false)) }
const XLN: usize = 128 * 5;
static XL_MIN: [std::sync::atomic::AtomicU64; XLN] = [const { std::sync::atomic::AtomicU64::new(u64::MAX) }; XLN];
static XL_CNT: [std::sync::atomic::AtomicU64; XLN] = [const { std::sync::atomic::AtomicU64::new(0) }; XLN];
static XL_K: [std::sync::atomic::AtomicU64; XLN] = [const { std::sync::atomic::AtomicU64::new(0) }; XLN];
static XL_LOW: [std::sync::atomic::AtomicU64; XLN] = [const { std::sync::atomic::AtomicU64::new(0) }; XLN];
// per layer stats: 0 n, 1 pool ns, 2 excess ns, 3 GPU-critical n, 4 moved experts, 5 n with excess, 6 hit experts
static XL_ST: [[std::sync::atomic::AtomicU64; 7]; 128] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 7] }; 128];
fn xl_idx(il: usize, t: usize) -> Option<usize> { (il < 128 && t < 5).then_some(il * 5 + t) }
/// layer il's rd wait (behind layer il-1's gate): update il-1's balance
fn xl_rdwait(il: usize, t: usize, ns: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if il == 0 || !(xbal() || xlstat()) { return; }
    let (Some(ib), Some(ip)) = (xl_idx(il, t), xl_idx(il - 1, t)) else { return; };
    let mn = XL_MIN[ib].fetch_min(ns, Relaxed).min(ns); let n = XL_CNT[ib].fetch_add(1, Relaxed);
    if n < 4 { return; }
    let ex = ns - mn; let hi = env_usize("PYRE_QN_XBAL_HI_US").unwrap_or(40) as u64 * 1000; let lo = env_usize("PYRE_QN_XBAL_LO_US").unwrap_or(10) as u64 * 1000;
    let s = &XL_ST[il - 1]; s[2].fetch_add(ex, Relaxed); s[5].fetch_add(1, Relaxed); if ex > hi { s[3].fetch_add(1, Relaxed); }
    if !xbal() { return; }
    if ex > hi { XL_K[ip].fetch_add(1, Relaxed); XL_LOW[ip].store(0, Relaxed); }
    else if ex < lo { let lw = XL_LOW[ip].fetch_add(1, Relaxed) + 1; if lw >= env_usize("PYRE_QN_XBAL_N").unwrap_or(4) as u64 { XL_LOW[ip].store(0, Relaxed); let k = XL_K[ip].load(Relaxed); if k > 0 { XL_K[ip].store(k - 1, Relaxed); } } }
    else { XL_LOW[ip].store(0, Relaxed); }
}
fn xl_print() {
    use std::sync::atomic::Ordering::Relaxed;
    let mut tot = [0u64; 7]; let mut lines = String::new();
    for (il, s) in XL_ST.iter().enumerate() { let v: Vec<u64> = s.iter().map(|a| a.load(Relaxed)).collect(); if v[0] == 0 { continue; }
        for i in 0..7 { tot[i] += v[i]; } let ne = v[5].max(1);
        lines += &format!(" L{il}:{:.0}/{:.0}/{:.0}%/{:.2}", v[1] as f64 / v[0] as f64 / 1e3, v[2] as f64 / ne as f64 / 1e3, 100.0 * v[3] as f64 / ne as f64, v[4] as f64 / v[0] as f64); }
    let ne = tot[5].max(1);
    eprintln!("xlstat calls {} pool_us/layer {:.0} excess_us {:.0} gpu_crit {:.1}% moved/layer {:.2} hits/layer {:.2} | per layer pool/excess/crit/moved:{lines}", tot[0], tot[1] as f64 / tot[0].max(1) as f64 / 1e3, tot[2] as f64 / ne as f64 / 1e3, 100.0 * tot[3] as f64 / ne as f64, tot[4] as f64 / tot[0].max(1) as f64, tot[6] as f64 / tot[0].max(1) as f64);
}
/// PYRE_QN_XEARLY=1 (XEXACT, opt-in; e39: pool +1-2 ms/window, no gain): see lcm_moe_p
fn xearly() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XEARLY").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_XPDIRECT (XGATE, default on; =0 off): see lcm_moe_p
fn xpdirect() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XPDIRECT").map(|v| v != "0").unwrap_or(true)) }
fn xfuse() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XFUSE").map(|v| v != "0").unwrap_or(true)) }
fn combm() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_COMBM").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_MOEGRAPH=1 (opt-in, needs PYRE_QN_VGRAPH=1): a verify window's GPU MoE glue (window q8pack, cached-expert
/// gate/up + down for all tokens, shared-expert silu_mul + down) as one captured graph per (layer, T). The routing
/// goes in as a fixed-size device table (t * top_k entries, padded; cumulative per-token counts in a 4-word buffer)
/// written before the replay, so the launch sequence no longer depends on which experts were picked. Idea after
/// Strata (MIT, github.com/Niko1221/Strata: fixed verify graphs fed by device-side expert plans); reimplemented here.
/// PYRE_QN_XEXACT=1 (opt-in, in progress): GPU cached experts with the CPU expert arithmetic (qdot Q8K/Q8D activations,
/// integer lanes, fma order, AVX sum tree), so where an expert runs cannot change its output.
thread_local! { static EXACT_PARTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// PYRE_QN_XALLCPU=1 (test): every routed expert runs on the CPU (placement-invariance checks)
fn xallcpu() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XALLCPU").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_XEXACT: layers whose expert types have no exact GPU kernel (only IQ2_S/IQ2_XXS + Q2_0 do) run all experts on the CPU
fn xexact_types(l: &Layer) -> bool { let t = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9)); t.0 == t.1 && t.0 <= 1 && t.2 == 3 }
/// PYRE_QN_XSCR_EX=0: the exact decode kernels allocate their temporaries per call instead of using the persistent XScr
fn xscr_ex() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XSCR_EX").map(|v| v != "0").unwrap_or(true)) }
/// XEXACT default expert cache (GiB): native IQ4_XS dense + native LM head leave room for more experts
pub fn xwmax() -> usize { static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *K.get_or_init(|| env_usize("PYRE_QN_XWMAX").unwrap_or(if crate::spec::policy_on() { 4 } else { 3 }).clamp(1, 4)) }
fn xexact_cache_gb() -> f64 { 5.6 } // e22: 2 arenas of 2900 MiB hold 5.66 GiB; 4-7 token window buffers gone (XWMAX=3) freed ~0.6 GB
fn xexact() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_XGATE (XEXACT, default on; =0 off): bounded host gate in the exact verify path, ported from qwen-pipe's
/// PYRE_QN_PIPE=2 (design after Strata, github.com/Niko1221/Strata, MIT, verify.cpp wait_flag/copy_rows_from_mapped;
/// reimplemented). Per MoE layer with >= PYRE_QN_XGATE_MINROWS CPU rows, beside the CPU pool the host queues: a bounded
/// device wait on a mapped host u32 (pyre_wait_flag: __nanosleep backoff, %globaltimer timeout PYRE_QN_XGATE_SPIN_US,
/// default 100000, capped at 200 ms in cubecl), the pinned->device copy of the CPU rows, rsum_ex + comb3_m and the next
/// layer's dense half. The host bumps the flag when the pool is done (also on unwind). A device timeout sets an err word;
/// the check after each window turns the gate off for the rest of the run.
fn xgate() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| xexact() && env_var("PYRE_QN_XGATE").map(|v| v != "0").unwrap_or(true)) && !XG_ERR.load(std::sync::atomic::Ordering::Relaxed) }
/// PYRE_QN_XZC (XEXACT, default on; =0 off): zero-copy per layer, design after Strata (github.com/Niko1221/Strata, MIT;
/// reimplemented). PYRE_QN_XZC & 1: the gated CPU rows are pulled by a kernel straight from the mapped pinned slot (one
/// launch = the bounded flag wait + 128-bit PCIe loads; no H2D copy). PYRE_QN_XZC & 2: the routing readback (rd) is
/// written by a kernel into mapped host memory followed by a doorbell; the host polls the doorbell (bounded,
/// PYRE_QN_XZC_SPIN_MS default 1000, then falls back to a normal blocking read). No copy-engine ops in the layer loop.
fn xzc() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| if xexact() { env_usize("PYRE_QN_XZC").map(|v| v as u32).unwrap_or(3) } else { 0 }) }
static XZ_ERR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static XZ_N: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// (doorbell page, 4 slots, slot bytes)
static XZ_RB: std::sync::Mutex<(usize, Vec<usize>, usize, u32)> = std::sync::Mutex::new((0, Vec::new(), 0, 0));
/// rd readback through mapped memory + doorbell (XZC & 2); None = not used (caller does the normal read)
/// PYRE_QN_XGR=1 (XEXACT; opt-in): the layer's routing top-k + hit classification run on the GPU (k_xgr_topk) right after
/// the router GEMV and the rd readback, and the hit expert kernels (xrun_exact over t*k entry slots, the unused ones
/// idle) are enqueued from the readback point, before the host has the routing; the host only applies its own softmax
/// weights (k_xgr_scale) and builds the sum table. Per token the host checks that its top_k set has no tie at the
/// boundary and that the layer table the GPU used still matches the cache; otherwise that layer runs the old host path.
/// PYRE_QN_XGR_CHK=1 (debug): reads the GPU's selected ids, hit entries and counts back every layer and compares them
/// with the host's (counts in the xhost line: xgr used/tie/tab/chk-mismatch).
fn xgr() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| xexact() && env_var("PYRE_QN_XGR").map(|v| v == "1").unwrap_or(false)) }
fn xgr_chk() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XGR_CHK").map(|v| v == "1").unwrap_or(false)) }
static XGR_N: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
struct XgrPend { il: usize, t: usize, y: H, mh: H, oc: H, dbg: H, rt: Vec<u32> }
thread_local! { static XGR_P: std::cell::RefCell<Option<XgrPend>> = const { std::cell::RefCell::new(None) }; }
static XGR_RT: std::sync::Mutex<Vec<Option<(H, Vec<u32>)>>> = std::sync::Mutex::new(Vec::new());
fn xz_read(cb: &super::qn_cube::Cube, h: &H, hk: &mut dyn FnMut()) -> Option<Vec<f32>> {
    if xzc() & 2 == 0 || XZ_ERR.load(std::sync::atomic::Ordering::Relaxed) { return None; }
    let len = h.size_in_used() as usize; if len % 4 != 0 || len == 0 { return None; }
    let mut g = XZ_RB.lock().unwrap();
    if g.0 == 0 || len > g.2 {
        let sz = len.max(1 << 18).next_power_of_two();
        let page = if g.0 != 0 { g.0 } else { match cb.client.pyre_host_pinned(4096) { Ok(p) => { unsafe { std::ptr::write_bytes(p as *mut u8, 0, 4096); } p } Err(e) => { eprintln!("PYRE_QN_XZC: pinned alloc failed ({e}); readback zero-copy off"); XZ_ERR.store(true, std::sync::atomic::Ordering::Relaxed); return None; } } };
        let mut v = vec![]; for _ in 0..4 { match cb.client.pyre_host_pinned(sz) { Ok(p) => v.push(p), Err(e) => { eprintln!("PYRE_QN_XZC: pinned alloc failed ({e}); readback zero-copy off"); XZ_ERR.store(true, std::sync::atomic::Ordering::Relaxed); return None; } } }
        g.0 = page; g.1 = v; g.2 = sz;
    }
    g.3 = g.3.wrapping_add(1).max(1); let seq = g.3; let slot = g.1[(seq as usize) % 4]; let fl = g.0 + 128;
    drop(g);
    let xcal = super::xchain::cal(cb);
    if let Err(e) = unsafe { cb.client.pyre_copy_to_mapped(h, len, slot, fl, seq) } { eprintln!("PYRE_QN_XZC: pyre_copy_to_mapped failed ({e}); readback zero-copy off"); XZ_ERR.store(true, std::sync::atomic::Ordering::Relaxed); return None; }
    cb.client.flush_queue();
    hk(); cb.client.flush_queue();
    let a = unsafe { &*(fl as *const std::sync::atomic::AtomicU32) };
    let t0 = std::time::Instant::now(); let lim = std::time::Duration::from_millis(env_usize("PYRE_QN_XZC_SPIN_MS").unwrap_or(1000) as u64); let mut n = 0u32;
    loop {
        if a.load(std::sync::atomic::Ordering::Acquire) == seq { break; }
        n += 1; if n < 2000 { std::hint::spin_loop(); } else { if t0.elapsed() > lim { break; } std::thread::yield_now(); }
    }
    if a.load(std::sync::atomic::Ordering::Acquire) != seq {
        eprintln!("PYRE_QN_XZC: doorbell not seen after {:?}; blocking read fallback, readback zero-copy off", t0.elapsed());
        XZ_ERR.store(true, std::sync::atomic::Ordering::Relaxed); XZ_N[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed); return None;
    }
    super::xchain::door(xcal);
    let mut v = vec![0f32; len / 4]; unsafe { std::ptr::copy_nonoverlapping(slot as *const f32, v.as_mut_ptr(), len / 4); }
    XZ_N[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(v)
}
static XG_ERR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static XG_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// gated layers / ungated (below MINROWS or not eligible) exact layers, for XWIN
static XG_N: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
thread_local! { static XG_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// True while this thread has a gate queued and not yet opened: no graph capture then (graph_prepare syncs the stream).
fn xg_open() -> bool { XG_OPEN.with(|g| g.get()) }
static XG_FLAG: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
fn xg_flag(cb: &super::qn_cube::Cube) -> Option<(usize, usize)> {
    if let Some(f) = XG_FLAG.get() { return Some(*f); }
    match cb.client.pyre_host_pinned(4096) { Ok(p) => { unsafe { std::ptr::write_bytes(p as *mut u8, 0, 4096); } let _ = XG_FLAG.set((p, p + 64)); XG_FLAG.get().copied() }
        Err(e) => { eprintln!("PYRE_QN_XGATE: pinned flag alloc failed ({e}); gate off"); XG_ERR.store(true, std::sync::atomic::Ordering::Relaxed); None } }
}
fn xg_err_check() { if let Some(&(_, er)) = XG_FLAG.get() { if unsafe { std::ptr::read_volatile(er as *const u32) } != 0 && !XG_ERR.swap(true, std::sync::atomic::Ordering::Relaxed) { eprintln!("PYRE_QN_XGATE: device wait timed out; gate disabled for the rest of the run (that window's output is invalid)"); } } }
/// Bumps the flag to the queued target on drop (also on unwind), so a queued device wait never outlives the host side.
struct XGOpen<'a>(usize, &'a std::cell::Cell<u32>);
impl Drop for XGOpen<'_> { fn drop(&mut self) { let tg = self.1.replace(0); if tg != 0 { unsafe { (*(self.0 as *const std::sync::atomic::AtomicU32)).store(tg, std::sync::atomic::Ordering::Release); } } XG_OPEN.with(|g| g.set(false)); } }
/// Ring of 8 page-locked host slots for the CPU rows (a slot is reused only after later blocking rd reads, i.e. after its copy ran).
fn xg_slot(cb: &super::qn_cube::Cube, bytes: usize, cap: usize) -> Option<usize> {
    static R: std::sync::Mutex<(Vec<usize>, usize, usize)> = std::sync::Mutex::new((Vec::new(), 0, 0));
    let mut g = R.lock().unwrap();
    if g.0.is_empty() { let sz = cap.max(bytes).max(1 << 16); for _ in 0..8 { match cb.client.pyre_host_pinned(sz) { Ok(p) => g.0.push(p), Err(e) => { eprintln!("PYRE_QN_XGATE: pinned alloc failed: {e}; gate off"); XG_ERR.store(true, std::sync::atomic::Ordering::Relaxed); return None; } } } g.2 = sz; }
    if bytes > g.2 { return None; }
    let i = g.1; g.1 = (i + 1) % g.0.len(); Some(g.0[i])
}
fn moegraph() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_MOEGRAPH").map(|v| v == "1").unwrap_or(false)) }
pub static MG_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// PYRE_QN_MOEGRAPH fixed buffers per window size: routing table, counts, routed-hit y, shared h and y, a copy of rd
struct MgB { meta: H, oc: H, y: H, hb: H, sh: H, rd: H }
#[allow(clippy::type_complexity)]
static MGB: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<MgB>>>> = std::sync::LazyLock::new(Default::default);
#[allow(clippy::type_complexity)]
static MG_RD: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<(usize, usize), String>>> = std::sync::LazyLock::new(Default::default);
fn mg_bufs(cb: &super::qn_cube::Cube, t: usize, max_ne: usize, d: usize, sf: usize, rn: usize) -> std::sync::Arc<MgB> {
    let mut m = MGB.lock().unwrap();
    m.entry(t).or_insert_with(|| std::sync::Arc::new(cb.client.memory_persistent_allocation((), |_| MgB { meta: cb.empty(max_ne * 4), oc: cb.empty(4), y: cb.empty(t * d), hb: cb.empty(t * sf), sh: cb.empty(t * d), rd: cb.empty(t * rn) }))).clone()
}
// PYRE_QN_XBATCH=0: the window's GPU expert hits token by token
fn xbatch() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XBATCH").map(|v| v != "0").unwrap_or(true)) }
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
        Ok(Model { mtp_gkv: Default::default(), vg: Default::default(), vxs: Default::default(), mtp_headc: Default::default(), mtp_dense: Default::default(), xpk: Default::default(), xper: Default::default(), lg: Default::default(), headc: None, xc: None, cube: None, cl: vec![], gpu: None, gmats: Default::default(), gl: vec![], embd: g.mat("token_embd.weight")?, out, head, ple_tab, layers, rope_inv, mtp: None, cfg, g })
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
                        eprintln!("cubebench gemv_m T=3 {nm} {}x{}: {us:.1} us ({:.0} GB/s)", m.rows, m.cols, gb / (us * 1e-6)); } if super::qn_cube::native_mode() == 2 { for (nm, m) in [("min", &c0.min), ("mout", &c0.mout), ("fin", &c0.fin)] { cb.natcheck(nm, m); } } }
                    bench("gate", &|| cb.gate(&x, &dn, &c0.hca.up, &o, 20000, 0, hc, d));
                    bench("comb", &|| cb.comb(&xn, &o, None, None, &dn, 320, hc, d));
                } }
                self.cl = cl;
                // LM head on the GPU (opt-in PYRE_QN_GPUHEAD=1: only ~+0.3 tok/s, and as Q8 it costs VRAM
                // default on with PYRE_QN_NATIVE=2 (same output on A/B/C; the verify head then runs as one batched GPU GEMV); PYRE_QN_GPUHEAD=0 off
                if env_var("PYRE_QN_GPUHEAD").map(|v| v == "1").unwrap_or(super::qn_cube::native2()) { self.headc = cb.upload_mix(&[&self.out], &[]); if self.headc.is_some() { eprintln!("gpu: lm head on gpu"); } }
                // routed-expert cache (default on, PYRE_QN_XCACHE_GB=<GiB>, 0 = off): up to that much of the VRAM left above the reserve
                let left = g.budget_left().max(0) as usize;
                let cap = env_var("PYRE_QN_XCACHE_GB").ok().and_then(|v| v.parse::<f64>().ok()).map(|v| (v * (1u64 << 30) as f64) as usize).unwrap_or(if xexact() { (xexact_cache_gb() * (1u64 << 30) as f64) as usize } else { 5usize << 29 }); // XEXACT: 3 GiB (e5: +hits, C +1.6 tok/s, ~3.3 GB VRAM still free); default on, 2.5 GiB: on the RTX 3060 (quiet GPU) 2 GiB gives ~17 vs ~14 tok/s cache off; bigger budgets push VRAM past the point where WDDM pages and get slower. 0 = off
                // PYRE_MTP_GX=2 (resident drafter, ~0.9 GB): the default cache gives up what the drafter needs beyond PYRE_MTP_GXR_SLACK_MB (default 300; g7: 512 left 1011 MiB free at the 1024 floor, 4.9 GiB left 1482) of the
                // headroom, so the 1 GB floor holds (XEXACT output is the same for any cache size); an explicit PYRE_QN_XCACHE_GB wins
                let cap = if super::mtp::gxr_on() && env_var("PYRE_QN_XCACHE_GB").is_err() && super::mtp::find().is_some() { let need = super::mtp::gxr_bytes(self.cfg.d, self.cfg.ff_exp, self.cfg.n_exp).saturating_sub(env_usize("PYRE_MTP_GXR_SLACK_MB").unwrap_or(300) << 20); eprintln!("gpu: resident drafter: expert cache cap -{:.2} GiB", need as f64 / (1u64 << 30) as f64); cap.saturating_sub(need) } else { cap };
                let budget = left.saturating_sub(env_usize("PYRE_QN_XHEAD_MB").unwrap_or(if xexact() { 512 } else { 2048 }) << 20).min(cap); // 2 GiB more headroom: past that, WDDM starts paging
                if budget > 0 { g.budget.fetch_sub(budget as i64, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("gpu: expert cache budget {:.2} GiB", budget as f64 / (1u64 << 30) as f64);
                    let slot = self.layers.iter().filter_map(|l| Some((super::qn_cube::xtype(l.gate_e.ty)?, super::qn_cube::xtype(l.up_e.ty)?, super::qn_cube::xtype(l.down_e.ty)?))).map(|t| { let w = super::qn_cube::xwords(t, self.cfg.d, self.cfg.ff_exp); 4 * (w.0 + w.1 + w.2) }).max().unwrap_or(4usize).next_multiple_of(256);
                    if env_var("PYRE_QN_XSTAT").is_ok() { let mut sz: std::collections::BTreeMap<usize, usize> = Default::default(); for l in &self.layers { if let (Some(a), Some(b), Some(c2)) = (super::qn_cube::xtype(l.gate_e.ty), super::qn_cube::xtype(l.up_e.ty), super::qn_cube::xtype(l.down_e.ty)) { let w = super::qn_cube::xwords((a, b, c2), self.cfg.d, self.cfg.ff_exp); *sz.entry(4 * (w.0 + w.1 + w.2)).or_default() += 1; } }
                        eprintln!("xstat: slot {slot} B, expert bytes by layer count {sz:?}, raw gguf expert bytes l0 {}", self.layers[0].gate_e.row_bytes() * self.cfg.ff_exp * 2 + self.layers[0].down_e.row_bytes() * self.cfg.d); }
                    let mut nslot = budget / slot; let maxper = (env_usize("PYRE_QN_ARENA_MB").unwrap_or(if xexact() { 2900 } else { 2560 }) << 20) / slot;
                    // e22: the exact kernels (and the t=1 / verify fast paths) take at most 2 arenas; at 5.4 GiB with 2.5 GiB arenas the cache
                    // had 3, so XEXACT silently fell back to the non-exact per-arena path (slower, and C diverged at token 76)
                    if xexact() && nslot > 2 * maxper { eprintln!("gpu: XEXACT expert cache capped at {} slots (2 arenas of {} MiB; the exact kernels take at most 2)", 2 * maxper, (maxper * slot) >> 20); nslot = 2 * maxper; }
                    let narena = nslot.div_ceil(maxper.max(1)).max(1); let per = nslot.div_ceil(narena).max(1); let ar_bytes = per * slot; // even arenas, each <= 2.5 GiB (one buffer > ~3.1 GB fails to allocate)
                    self.xc = Some(std::sync::Mutex::new(XCache { tabs: cb.xtabs(), arena: (0..narena).map(|_| if xarena_persist() { cb.client.memory_persistent_allocation((), |_| cb.xarena(ar_bytes)) } else { cb.xarena(ar_bytes) }).collect(), ar_bytes, per, slot, free: (0..nslot).rev().collect(), map: Default::default(), cnt: vec![vec![0; self.cfg.n_exp]; self.layers.len()], stream: vec![], used: 0, budget, calls: 0, hits: 0, total: 0, scr: cb.xscratch(self.cfg.d, self.cfg.ff_exp, self.cfg.top_k), sloc: vec![] }));
                    if xclass() { let xm = self.xc.as_ref().unwrap(); let mut x = xm.lock().unwrap();
                        // classes: expert bytes (256-aligned) -> layer count; each layer gets q slots of its class, largest first
                        let mut cls: std::collections::BTreeMap<usize, usize> = Default::default();
                        for l in &self.layers { if let (Some(a), Some(b), Some(c2)) = (super::qn_cube::xtype(l.gate_e.ty), super::qn_cube::xtype(l.up_e.ty), super::qn_cube::xtype(l.down_e.ty)) { let w = super::qn_cube::xwords((a, b, c2), self.cfg.d, self.cfg.ff_exp); *cls.entry((4 * (w.0 + w.1 + w.2)).next_multiple_of(256)).or_default() += 1; } }
                        let tot: usize = x.arena.len() * ar_bytes; let per_q: usize = cls.iter().map(|(s, n)| s * n).sum::<usize>().max(1);
                        let mut q = tot / per_q; let mut sloc = vec![];
                        loop { sloc.clear(); let (mut ai, mut off) = (0usize, 0usize); let mut ok = true;
                            for (&sz, &n) in cls.iter().rev() { for _ in 0..n * q { if off + sz > ar_bytes { ai += 1; off = 0; } if ai >= x.arena.len() { ok = false; break; } sloc.push((ai, off, sz)); off += sz; } if !ok { break; } }
                            if ok || q == 0 { break; } q -= 1; }
                        nslot = sloc.len(); x.free = (0..nslot).rev().collect(); x.budget = sloc.iter().map(|v| v.2).sum(); x.sloc = sloc;
                        eprintln!("xcache: XCLASS {} slots ({} per layer; classes {:?}) vs {} uniform", nslot, q, cls, budget / slot); }
                    self.xpk_prefill(nslot, Some(&cb)); }
                // PYRE_QN_XSTREAM=K: K cache slots kept free for streaming missed experts to the GPU per layer
                if let Some(xm) = &self.xc { let mut x = xm.lock().unwrap(); let k = xstream().min(x.free.len()); for _ in 0..k { let mx = x.free.iter().map(|&s| x.cap(s)).max().unwrap_or(0); let i = x.free.iter().position(|&s| x.cap(s) == mx).unwrap_or(0); let sl = x.free.remove(i); let cp = x.cap(sl); x.stream.push(sl); x.budget = x.budget.saturating_sub(cp); } }
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
            let csb = self.rope_h(cb, pos, half); let att = cb.empty(nh * hd);
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
            let sel = self.route_sel(il, &rv[d..d + ne], Some(&xcg)); let seln: Vec<usize> = sel.iter().map(|x| x.0).collect();
            for &(e, _) in &sel { xcg.cnt[il][e] += xinc(); }
            xcg.total += sel.len() as u64;
            let (on, mut off): (Vec<(usize, f32)>, Vec<(usize, f32)>) = sel.into_iter().partition(|(e, _)| !xallcpu() && !(xexact() && !xexact_types(l)) && xcg.map.contains_key(&(il, *e)));
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
            xcg.hits += on.len() as u64; XW[1].fetch_add((on.len() + strm.len()) as u64, std::sync::atomic::Ordering::Relaxed);
            if xpred() { let miss: Vec<usize> = off.iter().chain(strm.iter()).map(|x| x.0).collect(); self.xpred_step(il, &seln, &miss, &rv[..d]); } crate::metrics::expert_cache(on.len() as u64, off.len() as u64);
            // PYRE_QN_XEXACT single token: hits with the CPU arithmetic, CPU rows unsummed, one in-order sum (as the verify path)
            if xexact() && xfast() && tys0.0 == tys0.1 && tys0.0 <= 1 && tys0.2 == 3 && xcg.arena.len() <= 2 && !xcg.arena.is_empty() && d % 256 == 0 && c.ff_exp % 64 == 0 {
                let kk = c.top_k; let mut tab = vec![0u32; kk + 1]; let mut n = 0usize;
                for &e in &seln { let cd = if let Some(p2) = on.iter().position(|x| x.0 == e) { Some(p2 as u32) } else if let Some(p3) = strm.iter().position(|x| x.0 == e) { Some((on.len() + p3) as u32) } else { off.iter().position(|x| x.0 == e).map(|i| (1u32 << 31) | i as u32) };
                    if let Some(cd) = cd { if n < kk { tab[1 + n] = cd; n += 1; } } }
                tab[0] = n as u32;
                let xcr = &*xcg; let mut ex_y: Option<H> = None; let mut tbh: Option<H> = None;
                let mut gpu = || {
                    // streamed misses (XSTREAM/XSHARE) join the hits: same exact kernels, so placement cannot change output
                    let sents: Vec<(super::qn_cube::XEnt, f32)> = strm.iter().enumerate().map(|(j, &(e, w))| { let sl = xcr.stream[j];
                        let pk = self.xpacked(il, e, tys0); let mut ent = xcr.up(cb, sl, &pk.0, pk.1, pk.2);
                        ent.arena = xcr.loc(sl).0; ent.slot = sl; (ent, w) }).collect();
                    if !(on.is_empty() && sents.is_empty()) { let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).chain(sents.iter().map(|(e, w)| (e, *w))).collect(); let n = ents.len() as u32;
                        // one upload: entry table | 4 counts | in-order sum table
                        let mut mv = super::qn_cube::Cube::xmeta(&xcr.arena, &ents); let ml = mv.len(); mv.extend_from_slice(&[n, n, n, n]); mv.extend_from_slice(&tab);
                        let mh = cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&mv)); let och = mh.clone().offset_start((ml * 4) as u64);
                        tbh = Some(mh.clone().offset_start(((ml + 4) * 4) as u64));
                        let yent = cb.empty(ents.len() * d); cb.xrun_exact(&xcr.tabs, &xcr.arena, xcr.ar_bytes, 1, ents.len(), &mh, &och, &rd, rn, d, c.ff_exp, tys0, &yent, if xscr_ex() { Some(&xcr.scr) } else { None }); ex_y = Some(yent); }
                    cb.silu_mul(&rd, rn, d + ne, d + ne + sf, &hb, sf);
                    cb.gemv(&cl.sdown, &hb, sf, 0, &sh, d, 0);
                    if kick() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } }
                };
                let mut yp = vec![0f32; off.len().max(1) * d];
                if !off.is_empty() {
                    let lists: Vec<(usize, Vec<(usize, f32)>)> = off.iter().map(|&(e, w)| (e, vec![(0usize, w)])).collect();
                    let work: Vec<(usize, &Vec<(usize, f32)>)> = lists.iter().map(|(e, v)| (*e, v)).collect();
                    let ord = vec![(0..off.len()).map(|i| (i, 0usize)).collect::<Vec<_>>()];
                    let tr = std::time::Instant::now(); EXACT_PARTS.with(|c| c.set(true));
                    self.routed_pool_ord(l, &rv[..d], &work, 1, &mut yp, &mut gpu, Some(&ord)); EXACT_PARTS.with(|c| c.set(false));
                    { let rns = tr.elapsed().as_nanos() as u64; ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
                } else { gpu(); }
                drop(xcg);
                let q6 = std::time::Instant::now();
                let yb = cb.buf(&yp); let tb = match tbh { Some(h) => h, None => cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&tab)) }; let ysum = cb.empty(d);
                cb.rsum_ex(ex_y.as_ref().unwrap_or(&yb), &yb, &tb, &ysum, d, kk, 1);
                cb.comb3(x, &sh, Some(&ysum), None, Some((&rd, rn, d + ne + 2 * sf)), &df, lrf, hc, d);
                let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
                HY[0].fetch_add(ns(q0, q1) + ns(q3, q4) + ns(q6, std::time::Instant::now()), Relaxed); HY[1].fetch_add(ns(q1, q2) + ns(q4, q5), Relaxed); XW[9].fetch_add(ns(q1, q2) + ns(q4, q5), Relaxed); HY[2].fetch_add(ns(q2, q3), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed);
                return;
            }
            // CPU workers start on the missed experts while this thread enqueues the GPU side
            let mut gpu = || {
                let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                let sents: Vec<(super::qn_cube::XEnt, f32)> = strm.iter().enumerate().map(|(j, &(e, w))| { let sl = xcg.stream[j]; let ff = c.ff_exp;
                    let _ = ff; let pk = self.xpacked(il, e, tys); let mut ent = xcg.up(cb, sl, &pk.0, pk.1, pk.2);
                    ent.arena = xcg.loc(sl).0; ent.slot = sl; (ent, w) }).collect();
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
                if kick() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } }
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
            if kick() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } }
            self.routed(l, &rv[..d], &rv[d..d + ne], 1, &mut yr, false);
        }
        let q6 = std::time::Instant::now();
        let yb = cb.buf(&yr);
        cb.comb3(x, &sh, Some(&yb), yg.as_ref(), Some((&rd, rn, d + ne + 2 * sf)), &df, lrf, hc, d);
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
        HY[0].fetch_add(ns(q0, q1) + ns(q3, q4) + ns(q6, std::time::Instant::now()), Relaxed); HY[1].fetch_add(ns(q1, q2) + ns(q4, q5), Relaxed); XW[9].fetch_add(ns(q1, q2) + ns(q4, q5), Relaxed); HY[2].fetch_add(ns(q2, q3), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed); crate::metrics::add_gpu_ns(ns(q1, q2) + ns(q4, q5));
    }

    fn gdn_bufs(&self, st: &State) -> Vec<(H, H)> {
        let cb = self.cube.as_ref().unwrap();
        let mk = |_: ()| st.conv.iter().zip(&st.ssm).map(|(a, b)| (cb.buf(a), cb.buf(b))).collect::<Vec<_>>();
        if vgraph_on() { /* PYRE_QN_VGRAPH: one set of GPU state buffers reused by every State (refilled from its CPU state), so verify graphs captured in prewarm keep their key for the real request */ let mut g = VG_STATE.lock().unwrap(); if g.0.len() == st.conv.len() { for ((h1, h2), (a, b)) in g.0.iter().zip(st.conv.iter().zip(&st.ssm)) { cb.client.write(h1, burn_cubecl::cubecl::bytes::Bytes::from_elems(a.clone())); cb.client.write(h2, burn_cubecl::cubecl::bytes::Bytes::from_elems(b.clone())); } return g.0.clone(); } let v = cb.client.memory_persistent_allocation((), mk); g.0 = v.clone(); g.1.clear(); return v; }
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
        if xarena_pin() && { let pk = self.xpk.lock().unwrap(); !pk.full && pk.bytes + by.len() <= cap0 } { if let Some(cb) = self.cube.as_ref() { let p0 = format!("{:?}", by.property()); cb.client.staging(std::iter::once(&mut by), false); static ONCE: std::sync::Once = std::sync::Once::new(); ONCE.call_once(|| eprintln!("xpk staging: {p0} -> {:?}", by.property())); } }
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
    /// The rope cos/sin row of `pos` on the device, uploaded once and shared by every attention layer (each small
    /// write costs tens of microseconds on the stream); PYRE_QN_ROPEC=0 uploads it per layer as before.
    fn rope_h(&self, cb: &super::qn_cube::Cube, pos: usize, half: usize) -> H {
        let mk = || { let cs: Vec<f32> = (0..half).map(|i| (pos as f32 * self.rope_inv[i]).cos()).chain((0..half).map(|i| (pos as f32 * self.rope_inv[i]).sin())).collect(); cb.buf(&cs) };
        if env_var("PYRE_QN_ROPEC").map(|v| v == "0").unwrap_or(false) { return mk(); }
        static C: std::sync::Mutex<Option<std::collections::HashMap<(usize, usize), H>>> = std::sync::Mutex::new(None);
        let mut g = C.lock().unwrap(); let m = g.get_or_insert_with(Default::default);
        if m.len() > 8192 { m.clear(); }
        m.entry((pos, half)).or_insert_with(mk).clone()
    }
    fn xpk_prefill(&self, skip: usize, cbp: Option<&super::qn_cube::Cube>) {
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
        // the model's Cube is not installed yet while loading, so xpacked() could not pin: pin the arena here, on this
        // thread (the decode thread's stream and its page-locked pool)
        let mut pk = pk;
        if xarena_pin() { if let Some(cb) = cbp { let t1 = std::time::Instant::now(); let keys: Vec<(usize, usize)> = pk.map.keys().cloned().collect();
            for kk in keys { let a = pk.map[&kk].clone(); if matches!(a.0.property(), burn_cubecl::cubecl::bytes::AllocationProperty::Pinned) { continue; }
                let mut by = a.0.clone(); cb.client.staging(std::iter::once(&mut by), false); pk.map.insert(kk, std::sync::Arc::new((by.shared(), a.1, a.2))); }
            eprintln!("host expert arena: pinned in {:.1}s", t1.elapsed().as_secs_f64()); } }
        { let np = pk.map.values().filter(|v| matches!(v.0.property(), burn_cubecl::cubecl::bytes::AllocationProperty::Pinned)).count(); eprintln!("host expert arena: {np} of {} pinned", pk.map.len()); }
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
                let mut ent = xc.up(cb, sl, &pk.0, pk.1, pk.2);
                ent.arena = xc.loc(sl).0; ent.slot = sl; xc.map.insert((il, e), ent); } }
        ADAPT_SWAPS.fetch_add(jobs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let dec = env_f32("PYRE_QN_ADAPT_DECAY").unwrap_or(0.7);
        for row in xc.cnt.iter_mut() { for c in row.iter_mut() { *c = (*c as f32 * dec) as u32; } }
    }
    fn xrefill(&self, cb: &super::qn_cube::Cube, xc: &mut XCache) {
        xc.calls += 1;
        // PYRE_QN_XSTAT=1: cache hit rate and misses per layer per token so far (every 16 calls)
        if xc.calls % 16 == 0 && env_var("PYRE_QN_XSTAT").is_ok() { let nl = xc.cnt.len().max(1); let tok = (xc.total as f64 / (nl as f64 * self.cfg.top_k as f64)).max(1.0);
            eprintln!("xstat: call {} hit {:.1}% misses/layer/token {:.2} resident {} slots {} slot {:.3} MB", xc.calls, 100.0 * xc.hits as f64 / xc.total.max(1) as f64, (xc.total - xc.hits) as f64 / (tok * nl as f64), xc.map.len(), xc.map.len() + xc.free.len(), xc.slot as f64 / 1e6); }
        if xc.calls <= 3 || xc.calls % 32 == 0 { crate::limits::vlog(&format!("refill call {} (cache {} experts)", xc.calls, xc.map.len())); }
        if adapt_on() && xc.calls != 1 && !xc.map.is_empty() { if xc.calls % adapt_every() == 0 { self.xadapt(cb, xc); } return; }
        let every = env_usize("PYRE_QN_XREFILL").unwrap_or(16);
        // an empty cache fills right away (the call before prefill has no routing counts yet, so the first
        // real fill used to wait until call `every` and then add only XSWAP experts)
        if xc.calls != 1 && xc.calls % every != 0 && !xc.map.is_empty() { return; }
        let ff = self.cfg.ff_exp; let d = self.cfg.d;
        let slot = xc.slot; let xcl = !xc.sloc.is_empty(); let (dd0, ff0) = (self.cfg.d, self.cfg.ff_exp);
        let size = |l: &Layer| if xcl { match (super::qn_cube::xtype(l.gate_e.ty), super::qn_cube::xtype(l.up_e.ty), super::qn_cube::xtype(l.down_e.ty)) { (Some(a), Some(b), Some(c2)) => { let w = super::qn_cube::xwords((a, b, c2), dd0, ff0); (4 * (w.0 + w.1 + w.2)).next_multiple_of(256) }, _ => slot } } else { slot };
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
        // PYRE_QN_XPERLAYER=1 (opt-in): every layer gets an equal share of the slots (Strata's per-layer slot ranges,
        // expert_cache.cpp layer_slot_range), ranked within the layer; a layer short of candidates leaves its share to the rest
        let pl_cap = if env_var("PYRE_QN_XPERLAYER").map(|v| v == "1").unwrap_or(false) { let nl = self.layers.len().max(1); let ns = xc.budget / slot.max(1); Some(ns.div_ceil(nl)) } else { None };
        let mut per: Vec<usize> = vec![0; self.layers.len()];
        for &(_, il, e) in &cand { if let Some(cap) = pl_cap { if per[il] >= cap { continue; } } let s = size(&self.layers[il]); if tot + s > xc.budget { if pl_cap.is_some() { continue; } break; } tot += s; per[il] += 1; want.insert((il, e)); }
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
            if xcl {
                // per-class slots: evict (least used first) only experts whose slot fits this one
                let need = size(l);
                while !xc.free.iter().any(|&s| xc.cap(s) >= need) { match evict.iter().rposition(|&(_, a, b)| xc.map.get(&(a, b)).map_or(false, |v| xc.cap(v.slot) >= need)) { Some(i) => { let (_, a, b) = evict.remove(i); if let Some(v) = xc.map.remove(&(a, b)) { xc.used -= v.bytes; xc.free.push(v.slot); } }, None => break } }
                let Some(i) = xc.free.iter().rposition(|&s| xc.cap(s) == need).or_else(|| xc.free.iter().rposition(|&s| xc.cap(s) >= need)) else { continue };
                let sl = xc.free.remove(i); let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
                xc.used += xc.cap(sl); jobs.push((il, e, sl, tys)); n += 1; continue;
            }
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
                let mut ent = xc.up(cb, sl, &pk.0, pk.1, pk.2);
                ent.arena = xc.loc(sl).0; ent.slot = sl; xc.map.insert((il, e), ent); } }
        if env_var("PYRE_QN_GPUBENCH").is_ok() && xc.calls == 1 { if let Some(((il, e), ent)) = xc.map.iter().next() {
            let l = &self.layers[*il]; let _ = e;
            let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
            let rbs = (l.gate_e.row_bytes(), l.up_e.row_bytes(), l.down_e.row_bytes());
            let xb = cb.buf(&vec![0.01f32; d]); let y0 = cb.buf(&vec![0f32; d]); let hx = cb.empty(ff);
            let _ = rbs; cb.xrun_b(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &[(ent, 0.1)], &xb, d, ff, tys, &hx, &y0); cb.read(&y0);
            let mh = cb.client.create_from_slice(&[ent.off[0], ent.off[1], ent.off[2], 0.1f32.to_bits()].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); if let Some(Some(c0)) = self.cl.iter().find(|x| x.is_some()) { let xx = cb.buf(&vec![0.01f32; 4 * d]); let oo = cb.empty(20000); cb.gemv(&c0.fin, &xx, 4 * d, 0, &oo, 20000, 0); cb.read(&oo); let t1 = std::time::Instant::now(); for _ in 0..200 { cb.gemv(&c0.fin, &xx, 4 * d, 0, &oo, 20000, 0); } cb.read(&oo); eprintln!("cubebench (late) gemv fin: {:.1} us", t1.elapsed().as_secs_f64() * 1e6 / 200.0); }
            for sk in [0u32, 6, 5, 3] { cb.xrun_m(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &mh, 1, &xb, d, ff, tys, &hx, &y0, sk); cb.read(&y0); let t1 = std::time::Instant::now(); for _ in 0..200 { cb.xrun_m(&xc.tabs, &xc.arena[ent.arena], xc.ar_bytes, &mh, 1, &xb, d, ff, tys, &hx, &y0, sk); } cb.read(&y0); eprintln!("cubebench expert skip {sk}: {:.1} us", t1.elapsed().as_secs_f64() * 1e6 / 200.0); }
            eprintln!("cubebench expert {:?}/{:?}/{:?} ({:.2} MB)", l.gate_e.ty, l.up_e.ty, l.down_e.ty, ent.bytes as f64 / 1e6);
            { let il0 = *il; let es: Vec<&super::qn_cube::XEnt> = xc.map.iter().filter(|((li, _), en)| *li == il0 && en.arena < 2).map(|(_, en)| en).take(8).collect();
              for ne in [1usize, 2, 4, 8] { if es.len() < ne { break; } let ents: Vec<(&super::qn_cube::XEnt, f32)> = es[..ne].iter().map(|e| (*e, 0.1f32)).collect();
                if !cb.xrun_fast_to(&xc.tabs, &xc.arena, xc.ar_bytes, &ents, &xb, d, d, ff, tys, &xc.scr, false, &y0) { eprintln!("cubebench xfast ne {ne}: not eligible"); break; } cb.read(&y0);
                let t1 = std::time::Instant::now(); for _ in 0..200 { cb.xrun_fast_to(&xc.tabs, &xc.arena, xc.ar_bytes, &ents, &xb, d, d, ff, tys, &xc.scr, false, &y0); } cb.read(&y0);
                let us = t1.elapsed().as_secs_f64() * 1e6 / 200.0; eprintln!("cubebench xfast ne {ne}: {us:.1} us ({:.0} GB/s)", ne as f64 * ent.bytes as f64 / (us * 1e-6) / 1e9); } }
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
        let pool = qn_pool(); let z0 = std::time::Instant::now(); let parts_out = EXACT_PARTS.with(|c| c.get());
        if parts_out { if let Some(q) = XHX_T.with(|c| c.get()) { XHX[3].fetch_add((z0 - q).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } }
        let usek = qdot::supports_k(l.gate_e.ty) && qdot::supports_k(l.up_e.ty) && env_var("PYRE_QN_Q8K").map(|v| v != "0").unwrap_or(true);
        // the plain Q8 activations are only read when the K path is off
        let acts: Vec<qdot::Q8Act> = if usek { vec![] } else { (0..t).map(|ti| qdot::quantize(&x[ti * d..(ti + 1) * d])).collect() };
        let exq = xexact(); let sil = |v: f32| if exq { qdot::xsilu(v) } else { silu(v) };
        let actk: Vec<qdot::Q8K> = if usek { (0..t).map(|ti| { let mut a = qdot::Q8K::default(); if exq { qdot::quantize_k_into_x(&x[ti * d..(ti + 1) * d], &mut a); } else { qdot::quantize_k_into(&x[ti * d..(ti + 1) * d], &mut a); } a }).collect() } else { vec![] };
        let blk = env_usize("PYRE_QN_SBLK").unwrap_or(32); let nb = ff.div_ceil(blk);
        { let np: usize = work.iter().map(|w| w.1.len()).sum(); XW2[1].fetch_add(np as u64, std::sync::atomic::Ordering::Relaxed); XW2[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        if fused() && !exq && !parts_out { self.routed_fused(l, work, t, y, side, ord, &acts, &actk, usek, blk, nb); XW2[0].fetch_add(z0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); return; }
        let mut hoff = vec![0usize; work.len() + 1]; for (i, (_, toks)) in work.iter().enumerate() { hoff[i + 1] = hoff[i] + toks.len() * ff; }
        let mut hid = vec![0f32; hoff[work.len()]]; let hp = SyncPtr(hid.as_mut_ptr());
        thread_local! { static SC: std::cell::RefCell<(super::qdot::Scratch, super::qdot::Scratch)> = std::cell::RefCell::new(Default::default()); }
        // PYRE_QN_LFIRST=1: gate/up tasks of the experts with the most window tokens first (task order only; every
        // hidden row is computed by the same code, so results are unchanged)
        let lfo: Vec<usize> = if env_var("PYRE_QN_LFIRST").map(|v| v == "1").unwrap_or(false) && work.len() > 1 { let mut o: Vec<usize> = (0..work.len()).collect(); o.sort_by_key(|&i| std::cmp::Reverse(work[i].1.len())); o } else { vec![] };
        // PYRE_QN_GUQ (default on): the task finishing an expert's last gate/up block quantizes that expert's hidden
        // rows for down (same function, same input -> identical), instead of a serial pass between the two rounds
        let used = l.down_e.ty == super::gguf_q::QType::Q2_0 && ff % 64 == 0 && env_var("PYRE_QN_Q8D").map(|v| v != "0").unwrap_or(true);
        let guq = used && env_var("PYRE_QN_GUQ").map(|v| v != "0").unwrap_or(true);
        let mut hdq: Vec<Vec<qdot::Q8D>> = if guq { work.iter().map(|(_, toks)| (0..toks.len()).map(|_| qdot::Q8D::default()).collect()).collect() } else { vec![] };
        struct QP(*mut Vec<qdot::Q8D>); unsafe impl Sync for QP {} unsafe impl Send for QP {}
        let qp = QP(hdq.as_mut_ptr()); let qp = &qp;
        // PYRE_QN_GGUIDE (default on): gate/up row blocks in 3 tiers (64-row blocks for ~50% of ff, 32 for ~30%, 16 for the
        // rest), claimed tier-major (all experts' big blocks first), so the job tail is a 16-row block; every hidden row is
        // still computed by one task with the same per-row code -> identical results. PYRE_QN_GGUIDE=0: uniform SBLK blocks.
        let gtask: Vec<(usize, usize, usize)> = if env_var("PYRE_QN_GGUIDE").map(|v| v != "0").unwrap_or(true) && env_var("PYRE_QN_SBLK").is_err() {
            let mut bs: Vec<Vec<(usize, usize)>> = vec![vec![]; 3]; let mut r = 0usize;
            for (ti, &(f, sz)) in [(0.5f64, 64usize), (0.3, 32), (0.2, 16)].iter().enumerate() { let end = if ti == 2 { ff } else { (r + ((f * ff as f64) / sz as f64).round() as usize * sz).min(ff) }; while r < end { let e2 = (r + sz).min(end); bs[ti].push((r, e2)); r = e2; } }
            let mut v = vec![]; for tier in &bs { for wi in 0..work.len() { let w2 = if lfo.is_empty() { wi } else { lfo[wi] }; for &(a, b) in tier { v.push((w2, a, b)); } } } v } else { vec![] };
        let gnb = if gtask.is_empty() { nb } else { gtask.len() / work.len().max(1) }; let gtn = if gtask.is_empty() { work.len() * nb } else { gtask.len() }; let gtask = &gtask;
        let left: Vec<std::sync::atomic::AtomicUsize> = if guq { (0..work.len()).map(|_| std::sync::atomic::AtomicUsize::new(gnb)).collect() } else { vec![] };
        let (t_first, t_last) = (std::sync::atomic::AtomicU64::new(u64::MAX), std::sync::atomic::AtomicU64::new(0)); let zp = std::time::Instant::now();
        if parts_out { XHX[4].fetch_add((zp - z0).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        let gu = |k: usize| { let tb = std::time::Instant::now(); t_first.fetch_min((tb - zp).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            struct B<'a>(std::time::Instant, std::time::Instant, &'a std::sync::atomic::AtomicU64); impl Drop for B<'_> { fn drop(&mut self) { BUSY.fetch_add(self.0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); BUSYN.fetch_add(1, std::sync::atomic::Ordering::Relaxed); self.2.fetch_max(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } } let _b = B(tb, zp, &t_last);
            let (wi, r0, r1) = if !gtask.is_empty() { gtask[k] } else { let (wi, b) = (k / nb, k % nb); (if lfo.is_empty() { wi } else { lfo[wi] }, b * blk, ((b + 1) * blk).min(ff)) }; let (e, toks) = work[wi];
            let gm = l.gate_e.slice_rows(e * ff + r0, r1 - r0); let um = l.up_e.slice_rows(e * ff + r0, r1 - r0);
            struct Q<'a>(bool, &'a [std::sync::atomic::AtomicUsize], usize, &'a QP, usize, usize, &'a SyncPtr, &'a [usize], bool);
            impl Drop for Q<'_> { fn drop(&mut self) { if !self.0 { return; } if self.1[self.2].fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                let v = unsafe { &mut *self.3 .0.add(self.2) }; let (ff, wi) = (self.5, self.2);
                for j in 0..self.4 { let hx = unsafe { self.6.slice(self.7[wi] + j * ff, ff) }; if self.8 { super::qdot::quantize_d_into_x(hx, &mut v[j]); } else { super::qdot::quantize_d_into(hx, &mut v[j]); } } } } }
            let _q = Q(guq, &left, wi, qp, toks.len(), ff, &hp, &hoff, exq);
            SC.with(|c| { let (sc, s2) = &mut *c.borrow_mut();
                let pf = env_usize("PYRE_QN_PREFETCH").unwrap_or(2);
                let multi = usek && toks.len() >= 2 && toks.len() <= 16 && gm.ty == um.ty && matches!(gm.ty, super::gguf_q::QType::IQ2S | super::gguf_q::QType::IQ2XXS);
                if multi { let al: Vec<&qdot::Q8K> = toks.iter().map(|&(ti, _)| &actk[ti]).collect(); let (mut gv, mut uv) = ([0f32; 16], [0f32; 16]);
                    for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                        if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                        qdot::row_dot_k_multi(gm.ty, gr, &al, &mut gv); qdot::row_dot_k_multi(um.ty, ur, &al, &mut uv);
                        for j in 0..toks.len() { unsafe { hp.slice(hoff[wi] + j * ff + r0 + r, 1)[0] = sil(gv[j]) * uv[j]; } } }
                    return; }
                for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                    if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                    for (j, &(ti, _)) in toks.iter().enumerate() {
                        let v = if usek { let pr = if pair_on() && gm.ty == um.ty { qdot::row_dot_k_pair(gm.ty, gr, ur, &actk[ti]) } else { None }; if let Some((g, u)) = pr { sil(g) * u } else { sil(qdot::row_dot_k(gm.ty, gr, &actk[ti])) * qdot::row_dot_k(um.ty, ur, &actk[ti]) } } else { silu(qdot::row_dot(gm.ty, gr, &acts[ti], sc)) * qdot::row_dot(um.ty, ur, &acts[ti], s2) };
                        unsafe { hp.slice(hoff[wi] + j * ff + r0 + r, 1)[0] = v; } } } });
        };
        let dblk = env_usize("PYRE_QN_SDBLK").unwrap_or(64); let yp = SyncPtr(y.as_mut_ptr());
        // PYRE_QN_DGUIDE (default on): down row blocks in 3 tiers of `threads` tasks each (55% / 30% / 15% of the rows,
        // claimed largest first), so the job tail is a small block instead of a full one; each row is still computed
        // by one task with the same code -> identical results. PYRE_QN_DGUIDE=0: uniform SDBLK blocks.
        let dbnd: Vec<usize> = if env_var("PYRE_QN_DGUIDE").map(|v| v != "0").unwrap_or(true) { let nt = pool.threads.max(1); let mut v = vec![0usize]; let mut acc = 0f64;
            for f in [0.55f64, 0.30, 0.15] { for _ in 0..nt { acc += f / nt as f64; let r = (((acc * d as f64) / 8.0).round() as usize * 8).min(d); if r > *v.last().unwrap() { v.push(r); } } }
            if *v.last().unwrap() < d { v.push(d); } v } else { (0..=d.div_ceil(dblk)).map(|b| (b * dblk).min(d)).collect() };
        let nd = dbnd.len() - 1; let dbnd = &dbnd;
        let mut slot0 = vec![0usize; work.len()]; let mut slot_n = 0usize; for (wi, (_, toks)) in work.iter().enumerate() { slot0[wi] = slot_n; slot_n += toks.len(); } let (slot0, slot_n) = (&slot0, slot_n);
        let down = |b: usize, hdq: &[Vec<qdot::Q8D>], hq: &[Vec<qdot::Q8Act>]| {
            let (r0, r1) = (dbnd[b], dbnd[b + 1]);
            if let Some(ord) = ord { SC.with(|c| { let (sc, _) = &mut *c.borrow_mut();
                // expert-major (sequential, prefetched) into per-(expert, token) partials, then each token
                // sums its partials in its own expert order: the same additions as a one-token call
                let nr = r1 - r0; let mut part = vec![0f32; slot_n * nr];
                for (wi, &(e, toks)) in work.iter().enumerate() { let dm = l.down_e.slice_rows(e * d + r0, nr); let al: Vec<&qdot::Q8D> = if used { hdq[wi].iter().collect() } else { vec![] };
                    for r in 0..nr { let dr = dm.row_raw(r); if r + 4 < nr { prefetch(dm.row_raw(r + 4)); }
                        if used && toks.len() >= 2 && toks.len() <= 8 && dmulti() { let mut ov = [0f32; 8]; qdot::q2_0_dot_d_multi(dr, &al, &mut ov);
                            for (j, &(_, w)) in toks.iter().enumerate() { part[(slot0[wi] + j) * nr + r] = w * ov[j]; } continue; }
                        for (j, &(_, w)) in toks.iter().enumerate() { part[(slot0[wi] + j) * nr + r] = w * if used { qdot::q2_0_dot_d(dr, &hdq[wi][j]) } else { qdot::row_dot(dm.ty, dr, &hq[wi][j], sc) }; } } }
                if parts_out { for sl in 0..slot_n { let yt = unsafe { yp.slice(sl * d + r0, nr) }; yt.copy_from_slice(&part[sl * nr..(sl + 1) * nr]); }
                    // XCK: this column chunk of every CPU row is in the pinned slot -> the device may pull it now
                    let tg = XCK_TG.load(std::sync::atomic::Ordering::Acquire); if tg != 0 && r0 == b * xck_blk() && r1 == (r0 + xck_blk()).min(d) { if let Some(&cp) = XCK_PG.get() { unsafe { (*((cp + 64 * b) as *const std::sync::atomic::AtomicU32)).store(tg, std::sync::atomic::Ordering::Release); } } }
                    return; }
                for (ti, lst) in ord.iter().enumerate() { let yt = unsafe { yp.slice(ti * d + r0, nr) };
                    for &(wi, j) in lst { let pp = &part[(slot0[wi] + j) * nr..(slot0[wi] + j + 1) * nr]; for r in 0..nr { yt[r] += pp[r]; } } } }); return; }
            SC.with(|c| { let (sc, _) = &mut *c.borrow_mut();
                for (wi, &(e, toks)) in work.iter().enumerate() { let dm = l.down_e.slice_rows(e * d + r0, r1 - r0); let al: Vec<&qdot::Q8D> = if used { hdq[wi].iter().collect() } else { vec![] };
                    for r in 0..r1 - r0 { let dr = dm.row_raw(r); if r + 4 < r1 - r0 { prefetch(dm.row_raw(r + 4)); }
                        if used && toks.len() >= 2 && toks.len() <= 8 && dmulti() { let mut ov = [0f32; 8]; qdot::q2_0_dot_d_multi(dr, &al, &mut ov);
                            for (j, &(ti, w)) in toks.iter().enumerate() { let v = w * ov[j]; unsafe { yp.slice(ti * d + r0 + r, 1)[0] += v; } } continue; }
                        for (j, &(ti, w)) in toks.iter().enumerate() { let v = w * if used { qdot::q2_0_dot_d(dr, &hdq[wi][j]) } else { qdot::row_dot(dm.ty, dr, &hq[wi][j], sc) }; unsafe { yp.slice(ti * d + r0 + r, 1)[0] += v; } } } } });
        };
        let mut sidef = || { let ts = std::time::Instant::now(); side(); SIDE_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); };
        // PYRE_QN_CHAIN (default on with GUQ): gate/up and down in ONE pool job (down tasks after all gate/up tasks; a
        // worker that claims a down task waits for the gate/up count), so workers start down while the caller is still
        // in side() (GPU enqueue); same per-task code and per-row order -> identical results
        let chain = guq && env_var("PYRE_QN_CHAIN").map(|v| v != "0").unwrap_or(true);
        let (z1, z2);
        if chain { let ngu = gtn; let gd = std::sync::atomic::AtomicUsize::new(0); let hqe: Vec<Vec<qdot::Q8Act>> = vec![];
            if super::spin_pool::xpstat() { super::spin_pool::XP_TAG.store(work.len(), std::sync::atomic::Ordering::Relaxed); }
            pool.run_side(ngu + nd, &|k| { if k < ngu { gu(k); gd.fetch_add(1, std::sync::atomic::Ordering::Release); } else {
                while gd.load(std::sync::atomic::Ordering::Acquire) < ngu { std::hint::spin_loop(); }
                let hd: &[Vec<qdot::Q8D>] = unsafe { std::slice::from_raw_parts(qp.0 as *const Vec<qdot::Q8D>, work.len()) }; down(k - ngu, hd, &hqe); } }, &mut sidef);
            if super::spin_pool::xpstat() { super::spin_pool::XP_TAG.store(0, std::sync::atomic::Ordering::Relaxed); }
            z1 = std::time::Instant::now(); z2 = z1; TL[0].fetch_add(t_first.load(std::sync::atomic::Ordering::Relaxed).min(1 << 40), std::sync::atomic::Ordering::Relaxed); TL[1].fetch_add(t_last.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed); TL[2].fetch_add((z1 - zp).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        } else {
        pool.run_side(gtn, &gu, &mut sidef);
        z1 = std::time::Instant::now(); TL[0].fetch_add(t_first.load(std::sync::atomic::Ordering::Relaxed).min(1 << 40), std::sync::atomic::Ordering::Relaxed); TL[1].fetch_add(t_last.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed); TL[2].fetch_add((z1 - zp).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let hq: Vec<Vec<qdot::Q8Act>> = if used { vec![] } else { work.iter().enumerate().map(|(wi, (_, toks))| (0..toks.len()).map(|j| qdot::quantize(&hid[hoff[wi] + j * ff..hoff[wi] + (j + 1) * ff])).collect()).collect() };
        let hdq: Vec<Vec<qdot::Q8D>> = if !used { vec![] } else if guq { hdq } else { work.iter().enumerate().map(|(wi, (_, toks))| (0..toks.len()).map(|j| { let mut a = qdot::Q8D::default(); if exq { qdot::quantize_d_into_x(&hid[hoff[wi] + j * ff..hoff[wi] + (j + 1) * ff], &mut a); } else { qdot::quantize_d_into(&hid[hoff[wi] + j * ff..hoff[wi] + (j + 1) * ff], &mut a); } a }).collect()).collect() };
        z2 = std::time::Instant::now();
        pool.run(nd, &|b| down(b, &hdq, &hq));
        }
        use std::sync::atomic::Ordering::Relaxed; XW2[0].fetch_add(z0.elapsed().as_nanos() as u64, Relaxed); HY[4].fetch_add((z1 - z0).as_nanos() as u64, Relaxed); HY[5].fetch_add((z2 - z1).as_nanos() as u64, Relaxed); HY[6].fetch_add(z2.elapsed().as_nanos() as u64, Relaxed);
        if xwin() { let np: usize = work.iter().map(|w| w.1.len()).sum(); let tot = z0.elapsed().as_nanos() as u64;
            XW[0].fetch_add(np as u64, Relaxed); XW[2].fetch_add(1, Relaxed); XW[3].fetch_add(tot, Relaxed); XW[4].fetch_max(tot, Relaxed);
            XW[5].fetch_add((z1 - z0).as_nanos() as u64, Relaxed); XW[6].fetch_add((z2 - z1).as_nanos() as u64, Relaxed); XW[7].fetch_add(z2.elapsed().as_nanos() as u64, Relaxed); XW[8].fetch_add(work.len() as u64, Relaxed); }
    }

    /// PYRE_QN_FUSE=1: one pool wake per layer. Task wi = expert wi: its gate/up rows (the same per-row code as
    /// routed_pool_ord), its own hidden quantize (same function/input), then its down rows into per-(expert,token)
    /// partials w*dot (same expressions); afterwards every token adds its partials in its own expert order
    /// (ord, or work order when None) exactly as the ordered/unordered paths do: bit-identical.
    /// PYRE_QN_FUSE_SERIAL=N: run the tasks on the calling thread when the layer has <= N expert x token pairs.
    #[allow(clippy::too_many_arguments)]
    fn routed_fused(&self, l: &Layer, work: &[(usize, &Vec<(usize, f32)>)], t: usize, y: &mut [f32], side: &mut dyn FnMut(), ord: Option<&[Vec<(usize, usize)>]>, acts: &[super::qdot::Q8Act], actk: &[super::qdot::Q8K], usek: bool, blk: usize, nb: usize) {
        use super::qdot; use super::spin_pool::SyncPtr;
        let (d, ff) = (self.cfg.d, self.cfg.ff_exp);
        let mut slot0 = vec![0usize; work.len()]; let mut slot_n = 0usize; for (wi, (_, toks)) in work.iter().enumerate() { slot0[wi] = slot_n; slot_n += toks.len(); }
        let mut part = vec![0f32; slot_n * d]; let pp = SyncPtr(part.as_mut_ptr());
        let used = l.down_e.ty == super::gguf_q::QType::Q2_0 && ff % 64 == 0 && env_var("PYRE_QN_Q8D").map(|v| v != "0").unwrap_or(true);
        let pf = env_usize("PYRE_QN_PREFETCH").unwrap_or(2);
        let task = |wi: usize| { let (e, toks) = work[wi]; let nt = toks.len(); let mut hid = vec![0f32; nt * ff]; let mut sc = qdot::Scratch::default();
            for b in 0..nb { let (r0, r1) = (b * blk, ((b + 1) * blk).min(ff));
                let gm = l.gate_e.slice_rows(e * ff + r0, r1 - r0); let um = l.up_e.slice_rows(e * ff + r0, r1 - r0);
                let multi = usek && nt >= 2 && nt <= 16 && gm.ty == um.ty && matches!(gm.ty, super::gguf_q::QType::IQ2S | super::gguf_q::QType::IQ2XXS);
                if multi { let al: Vec<&qdot::Q8K> = toks.iter().map(|&(ti, _)| &actk[ti]).collect(); let (mut gv, mut uv) = ([0f32; 16], [0f32; 16]);
                    for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                        if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                        qdot::row_dot_k_multi(gm.ty, gr, &al, &mut gv); qdot::row_dot_k_multi(um.ty, ur, &al, &mut uv);
                        for j in 0..nt { hid[j * ff + r0 + r] = silu(gv[j]) * uv[j]; } }
                    continue; }
                for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
                    if pf > 0 && r + pf < r1 - r0 { prefetch(gm.row_raw(r + pf)); prefetch(um.row_raw(r + pf)); }
                    for (j, &(ti, _)) in toks.iter().enumerate() {
                        let v = if usek { let pr = if pair_on() && gm.ty == um.ty { qdot::row_dot_k_pair(gm.ty, gr, ur, &actk[ti]) } else { None }; if let Some((g, u)) = pr { silu(g) * u } else { silu(qdot::row_dot_k(gm.ty, gr, &actk[ti])) * qdot::row_dot_k(um.ty, ur, &actk[ti]) } } else { silu(qdot::row_dot(gm.ty, gr, &acts[ti], &mut sc)) * qdot::row_dot(um.ty, ur, &acts[ti], &mut sc) };
                        hid[j * ff + r0 + r] = v; } } }
            let hq: Vec<qdot::Q8Act> = if used { vec![] } else { (0..nt).map(|j| qdot::quantize(&hid[j * ff..(j + 1) * ff])).collect() };
            let hdq: Vec<qdot::Q8D> = if !used { vec![] } else { (0..nt).map(|j| { let mut a = qdot::Q8D::default(); qdot::quantize_d_into(&hid[j * ff..(j + 1) * ff], &mut a); a }).collect() };
            let dm = l.down_e.slice_rows(e * d, d); let al: Vec<&qdot::Q8D> = hdq.iter().collect();
            for r in 0..d { let dr = dm.row_raw(r); if r + 4 < d { prefetch(dm.row_raw(r + 4)); }
                if used && nt >= 2 && nt <= 8 && dmulti() { let mut ov = [0f32; 8]; qdot::q2_0_dot_d_multi(dr, &al, &mut ov);
                    for (j, &(_, w)) in toks.iter().enumerate() { unsafe { pp.slice((slot0[wi] + j) * d + r, 1)[0] = w * ov[j]; } } continue; }
                for (j, &(_, w)) in toks.iter().enumerate() { let v = w * if used { qdot::q2_0_dot_d(dr, &hdq[j]) } else { qdot::row_dot(dm.ty, dr, &hq[j], &mut sc) }; unsafe { pp.slice((slot0[wi] + j) * d + r, 1)[0] = v; } } } };
        let ser = env_usize("PYRE_QN_FUSE_SERIAL").unwrap_or(0);
        if slot_n <= ser { side(); for wi in 0..work.len() { task(wi); } } else {
            let pool = qn_pool(); pool.run_side(work.len(), &|wi| task(wi), &mut || { let ts = std::time::Instant::now(); side(); SIDE_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }); }
        // ordered sums: per token, its partials in its expert order (ord) or work order
        match ord { Some(ord) => for (ti, lst) in ord.iter().enumerate() { let yt = &mut y[ti * d..(ti + 1) * d]; for &(wi, j) in lst { let p = &part[(slot0[wi] + j) * d..(slot0[wi] + j + 1) * d]; for r in 0..d { yt[r] += p[r]; } } },
            None => for (wi, (_, toks)) in work.iter().enumerate() { for (j, &(ti, _)) in toks.iter().enumerate() { let p = &part[(slot0[wi] + j) * d..(slot0[wi] + j + 1) * d]; let yt = &mut y[ti * d..(ti + 1) * d]; for r in 0..d { yt[r] += p[r]; } } } }
        let _ = t;
    }
}
/// PYRE_QN_XWIN=1: one stderr line per decode window: CPU expert work (expert-token pairs, distinct experts, layers with
/// CPU work, pool wall total / max per layer / gate-up phase / quantize / down phase), GPU hits, window wall
/// PYRE_QN_XPRED=1 (measurement): miss predictors at t=1 decode. "nl": layer l+1's router on layer l's MoE input
/// (top 10 / top 20); "pw": the experts layer l+1 used for the previous token. Stats: share of actual experts and of
/// actual misses (CPU-run) that the prediction covers.
fn xpred() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XPRED").map(|v| v == "1").unwrap_or(false)) }
static XP: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
thread_local! { static XPS: std::cell::RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>, u64)> = const { std::cell::RefCell::new((vec![], vec![], 0)) }; }
impl Model {
    fn xpred_step(&self, il: usize, act: &[usize], miss: &[usize], x: &[f32]) {
        use std::sync::atomic::Ordering::Relaxed; let nl = self.layers.len(); let k = self.cfg.top_k;
        XPS.with(|c| { let mut c = c.borrow_mut(); if c.0.len() != nl { c.0 = vec![vec![]; nl]; c.1 = vec![vec![]; nl]; }
            let pr = std::mem::take(&mut c.0[il]);
            if !pr.is_empty() { let a10 = pr[..k.min(pr.len())].iter().filter(|e| act.contains(e)).count(); let a20 = pr.iter().filter(|e| act.contains(e)).count();
                let m10 = pr[..k.min(pr.len())].iter().filter(|e| miss.contains(e)).count(); let m20 = pr.iter().filter(|e| miss.contains(e)).count();
                XP[0].fetch_add(act.len() as u64, Relaxed); XP[1].fetch_add(miss.len() as u64, Relaxed); XP[2].fetch_add(a10 as u64, Relaxed); XP[3].fetch_add(a20 as u64, Relaxed); XP[4].fetch_add(m10 as u64, Relaxed); XP[5].fetch_add(m20 as u64, Relaxed); }
            let pw = std::mem::replace(&mut c.1[il], act.to_vec());
            if !pw.is_empty() { XP[6].fetch_add(act.len() as u64, Relaxed); XP[7].fetch_add(miss.len() as u64, Relaxed); XP[8].fetch_add(pw.iter().filter(|e| act.contains(e)).count() as u64, Relaxed); XP[9].fetch_add(pw.iter().filter(|e| miss.contains(e)).count() as u64, Relaxed); }
            if il + 1 < nl { let r = &self.layers[il + 1].router; let mut lg = vec![0f32; r.rows]; r.matmul(x, 1, &mut lg);
                let mut idx: Vec<usize> = (0..lg.len()).collect(); idx.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap()); idx.truncate(2 * k); c.0[il + 1] = idx; }
            if il == 0 { c.2 += 1; if c.2 % 32 == 0 { let g = |i: usize| XP[i].load(Relaxed) as f64; let p = |a: f64, b: f64| 100.0 * a / b.max(1.0);
                eprintln!("xpred tok {}: next-layer router top10 covers {:.1}% of experts / {:.1}% of misses, top20 {:.1}% / {:.1}% | prev-token same layer {:.1}% / {:.1}% | misses/layer {:.2}",
                    c.2, p(g(2), g(0)), p(g(4), g(1)), p(g(3), g(0)), p(g(5), g(1)), p(g(8), g(6)), p(g(9), g(7)), g(1) / (g(0) / k as f64).max(1.0)); } }
        });
    }
}
/// PYRE_QN_XCK (int: default on under XEXACT only with PYRE_QN_DGUIDE=0 - a down task stores its chunk flag only when its rows are exactly uniform chunk b; =0 off; gated layers with XPDIRECT/XZT): the CPU rows are pulled per column chunk (the pool's
/// down-phase task = PYRE_QN_SDBLK floats of every row) by cubecl-pipe pyre_wait_copy_chunks: each chunk's device block waits
/// (bounded) on its own mapped flag, stored by the pool task that finished it, or on the layer's gate flag (always stored when
/// the pool is done, also on unwind), so most rows cross PCIe while the pool still runs. Same bytes -> bit-identical.
fn xck() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XCK").map(|v| v != "0").unwrap_or(xexact() && env_var("PYRE_QN_DGUIDE").map(|v| v == "0").unwrap_or(false))) }
fn xck_blk() -> usize { static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *K.get_or_init(|| env_usize("PYRE_QN_SDBLK").unwrap_or(64)) }
static XCK_TG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static XCK_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static XCK_PG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn xck_page(cb: &super::qn_cube::Cube) -> Option<usize> {
    if let Some(&p) = XCK_PG.get() { return Some(p); }
    match cb.client.pyre_host_pinned(1 << 16) { Ok(p) => { unsafe { std::ptr::write_bytes(p as *mut u8, 0, 1 << 16); } let p64 = p.div_ceil(64) * 64; let _ = XCK_PG.set(p64); Some(p64) }
        Err(e) => { eprintln!("PYRE_QN_XCK: pinned alloc failed ({e}); per-chunk rows off"); None } }
}
fn xwin() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| env_var("PYRE_QN_XWIN").map(|v| v == "1").unwrap_or(false)) }
/// XWIN host side of a gated layer (ns/window): 0 routing start -> GPU side task start, 1 side start -> hits enqueued, 2 side total
thread_local! { static XHX_T: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) }; }
static XHX: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static XW: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16]; // 9 dense read wait ns, 10 refill ns, 11 head ns, 12 GPU expert enqueue ns
thread_local! { static XW_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
fn xwin_begin() -> Option<std::time::Instant> { if !xwin() { return None; } let d = XW_DEPTH.with(|c| { let v = c.get(); c.set(v + 1); v }); if d == 0 { for a in XW.iter() { a.store(0, std::sync::atomic::Ordering::Relaxed); } Some(std::time::Instant::now()) } else { None } }
fn xwin_end(w: Option<std::time::Instant>, t: usize) { if !xwin() { return; } XW_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
    if let Some(w) = w { if t <= 8 { let g = |i: usize| XW[i].load(std::sync::atomic::Ordering::Relaxed); let ms = |v: u64| v as f64 / 1e6;
        eprintln!("xwin T={t} wall {:.2} cpu_pairs {} cpu_exp {} lay {} pool {:.2} maxlay {:.3} gu {:.2} q {:.2} dn {:.2} gpu {} rdwait {:.2} refill {:.2} head {:.2} xenq {:.2} ple {:.2} route {:.2} post {:.2} rest {:.2} gated {} ungated {}", w.elapsed().as_secs_f64() * 1e3, g(0), g(8), g(2), ms(g(3)), ms(g(4)), ms(g(5)), ms(g(6)), ms(g(7)), g(1), ms(g(9)), ms(g(10)), ms(g(11)), ms(g(12)), ms(g(13)), ms(g(14)), ms(g(15)), w.elapsed().as_secs_f64() * 1e3 - ms(g(3)) - ms(g(9)) - ms(g(10)) - ms(g(11)), XG_N[0].swap(0, std::sync::atomic::Ordering::Relaxed), XG_N[1].swap(0, std::sync::atomic::Ordering::Relaxed)); eprintln!("xhost toside {:.2} tohits {:.2} side {:.2} topool {:.2} psetup {:.2} zc rd {} rows {} fb {} zt {} xgr {}/{}/{}/{} rt {}", ms(XHX[0].swap(0, std::sync::atomic::Ordering::Relaxed)), ms(XHX[1].swap(0, std::sync::atomic::Ordering::Relaxed)), ms(XHX[2].swap(0, std::sync::atomic::Ordering::Relaxed)), ms(XHX[3].swap(0, std::sync::atomic::Ordering::Relaxed)), ms(XHX[4].swap(0, std::sync::atomic::Ordering::Relaxed)), XZ_N[0].swap(0, std::sync::atomic::Ordering::Relaxed), XZ_N[1].swap(0, std::sync::atomic::Ordering::Relaxed), XZ_N[2].load(std::sync::atomic::Ordering::Relaxed), XZ_N[3].swap(0, std::sync::atomic::Ordering::Relaxed), XGR_N[0].swap(0, std::sync::atomic::Ordering::Relaxed), XGR_N[1].swap(0, std::sync::atomic::Ordering::Relaxed), XGR_N[2].swap(0, std::sync::atomic::Ordering::Relaxed), XGR_N[3].load(std::sync::atomic::Ordering::Relaxed), XGR_N[4].swap(0, std::sync::atomic::Ordering::Relaxed)); } } }
impl Model {
    #[allow(dead_code)] fn _xwin_dummy(&self) {}

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
        let xw = xwin_begin(); let lg = self.forward_inner(ids, st); xwin_end(xw, ids.len());
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
        { let qr = std::time::Instant::now(); if let (Some(cb), Some(xm), 1) = (&self.cube, &self.xc, t) { let mut g = xm.lock().unwrap(); if !g.map.is_empty() || g.calls > 0 { self.xrefill(cb, &mut g); } } XW[10].fetch_add(qr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
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
        // PYRE_MTP_GPU=1 (default with PYRE_QN_NATIVE=2, whose dense weights leave the VRAM for it; =0 off): the drafter's dense matrices (~95 MB Q8) run as GPU GEMVs on this thread; experts stay on the CPU
        let gh: Option<Box<dyn Fn(&super::qdot::Q8Mat, &[f32], usize) -> Option<Vec<f32>> + '_>> = match (&self.cube, env_var("PYRE_MTP_GPU").map(|v| v != "0").unwrap_or(true)) { (Some(cb), true) => {
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
        if env_var("PYRE_MTP_DEBUG").is_ok() { eprintln!("mtp: catch n {n} lo {lo} kv {}..{} first {first}", kv.c0, kv.end()); }
        if lo < n - 1 {
            let mut rs = Vec::with_capacity((n - 1 - lo) * w); let mut es = vec![0f32; (n - 1 - lo) * d]; let mut ok = true;
            for (j, i) in (lo..n - 1).enumerate() { match self.mtp_r_get(st, i) { Some(r) => rs.extend_from_slice(r), None => { ok = false; break; } } self.embd.row(st.tokens[i + 1] as usize, &mut es[j * d..(j + 1) * d]); }
            let gsc = mtp_gpustep() && env_var("PYRE_MTP_GPUCATCH").map(|v| v != "0").unwrap_or(true);
            let mut gdone_c = false;
            if ok && gsc { if let (Some(cb), Some(map), Some(Some(_))) = (self.cube.as_ref(), self.mtp_dense.get(), self.mtp_headc.get()) {
                // PYRE_MTP_GPUCATCH (default on, =0 off; pp26 B 27.7 vs 26.6, C/A same, greedy == refs): catch-up cells on the GPU, written straight into the resident K/V ring; inputs in one upload, K/V copies in one readback
                let g = |q: &super::qdot::Q8Mat| map.get(&(q as *const _ as usize));
                if kv.end() != lo || kv.n == 0 { m.kv_reset(&mut kv, lo); }
                let mut rg = self.mtp_gkv.lock().unwrap(); m.kv_sync(cb, &mut rg, &mut kv); let gr = rg.as_mut().unwrap(); let (kc, vc, cap) = (gr.k.clone(), gr.v.clone(), gr.cap);
                let mut okc = true; let nc = n - 1 - lo; let rsb = cb.buf(&rs); let esb = cb.buf(&es); let pkn = 2 * nc * row; let pka = cb.empty(pkn);
                for j in 0..nc { if m.kv_cell_gpu(cb, &g, &kc, &vc, cap, &rsb.clone().offset_start((j * w * 4) as u64), &esb.clone().offset_start((j * d * 4) as u64), lo + j, &pka, pkn, j).is_none() { okc = false; break; } }
                if okc { let hv = cb.read(&pka); for j in 0..nc { m.kv_push(&mut kv, &hv[2 * j * row..(2 * j + 1) * row], &hv[(2 * j + 1) * row..(2 * j + 2) * row]); } }
                if okc { gr.end = kv.end(); gr.gid = kv.gid; gdone_c = true; } else { kv = Default::default(); gr.end = 0; gdone_c = true; }
            } }
            if gdone_c {} else if ok { if let Some(h) = &gh { super::mtp::with_mm_hook(h, || m.catch_up(&mut kv, &rs, &es, n - 1 - lo, lo)); } else { mtp_cpu(|| m.catch_up(&mut kv, &rs, &es, n - 1 - lo, lo)); } } else { kv = Default::default(); }
        }
        let t1 = std::time::Instant::now();
        let mut r = self.mtp_r_get(st, n - 1)?.to_vec(); let mut tok = pending; let mut out = vec![];
        // a later draft is kept only when the drafter is this sure of it (Strata gates draft length on the MTP probability too; PYRE_MTP_MIN_P=0 drafts the full length)
        let min_p: f32 = crate::spec::draft_min_p().unwrap_or_else(|| env_var("PYRE_MTP_MIN_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9));
        let mut e = vec![0f32; d];
        // PYRE_MTP_GPUSTEP=1 (needs PYRE_MTP_GPU=1): each draft step on the GPU (draft-only numerics), routed experts on the CPU, head + k_best on the GPU
        let mut gdone = false;
        if mtp_gpustep() && kv.end() == n - 1 && kv.n > 0 {
            if let (Some(cb), Some(map), Some(Some(hm))) = (self.cube.as_ref(), self.mtp_dense.get(), self.mtp_headc.get()) {
                let g = |q: &super::qdot::Q8Mat| map.get(&(q as *const _ as usize));
                let mut rg = self.mtp_gkv.lock().unwrap(); m.kv_sync(cb, &mut rg, &mut kv); let gr = rg.as_mut().unwrap(); let (kc, vc, cap) = (gr.k.clone(), gr.v.clone(), gr.cap); let kc0 = kv.c0; let mut rh = cb.buf(&r);
                // PYRE_MTP_GRAPH (default on, =0 off): the step as two captured CUDA graphs (head included); pp24 A 33.5 vs 29.8* steady, B 27.2 vs 26.8, C 29.1 vs 29.2, greedy == refs
                let gmode = env_var("PYRE_MTP_GRAPH").map(|v| v != "0").unwrap_or(true); let rid = gr.id;
                // PYRE_MTP_GX=1: routed experts on the GPU (resident LRU pool) + the k draft steps chained on the GPU, one readback per run
                let gxm = gmode && super::mtp::gx_on();
                if gxm { self.embd.row(tok as usize, &mut e); let th = std::time::Instant::now();
                    if let Some((bs, krow, vrow)) = m.draft_gx(cb, &g, hm, &kc, &vc, cap, kc0, &rh, &e, n - 1, rid, k, &|t, o| self.embd.row(t as usize, o)) {
                        m.kv_push(&mut kv, &krow, &vrow);
                        for (j, &(i, p)) in bs.iter().enumerate() { if j > 0 && crate::spec::draft_stop() { break; } if !crate::spec::draft_gate(j, p, min_p) { break; } out.push((m.head_ids()[i.min(hm.rows - 1)], None)); }
                    }
                    HEAD_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                for j in 0..(if gmode && !gxm { k } else { 0 }) {
            if j > 0 && crate::spec::draft_stop() { break; }
                    self.embd.row(tok as usize, &mut e);
                    let Some((lg, rr2, krow, vrow)) = m.step_gpu_g(cb, &g, hm, &kc, &vc, cap, kc0, &rh, &e, n - 1 + j, rid) else { break };
                    if j == 0 { m.kv_push(&mut kv, &krow, &vrow); }
                    let th = std::time::Instant::now(); let (i, p) = cb.best(&lg, hm.rows); let id = m.head_ids()[i.min(hm.rows - 1)];
                    HEAD_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    if !crate::spec::draft_gate(j, p, min_p) { break; }
                    out.push((id, None)); rh = rr2; tok = id;
                }
                for j in 0..(if gmode { 0 } else { k }) {
            if j > 0 && crate::spec::draft_stop() { break; }
                    self.embd.row(tok as usize, &mut e);
                    let Some(gs) = m.step_gpu(cb, &g, &kc, &vc, cap, kc0, &rh, &e, n - 1 + j) else { break };
                    if j == 0 { m.kv_push(&mut kv, &gs.krow, &gs.vrow); }
                    let th = std::time::Instant::now(); let o = cb.empty(hm.rows); cb.gemv(hm, &gs.s, d, 0, &o, hm.rows, 0); let (i, p) = cb.best(&o, hm.rows); let id = m.head_ids()[i.min(hm.rows - 1)];
                    HEAD_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    if !crate::spec::draft_gate(j, p, min_p) { break; }
                    out.push((id, None)); rh = gs.rr; tok = id;
                }
                gr.end = gr.end.min(kv.end()).max(if kv.end() == n { n } else { 0 }); gr.gid = kv.gid; gdone = true;
            }
        }
        for j in 0..(if gdone { 0 } else { k }) {
            if j > 0 && crate::spec::draft_stop() { break; }
            self.embd.row(tok as usize, &mut e);
            let (s, r2) = if let Some(h) = &gh { super::mtp::with_mm_hook(h, || m.step(&mut kv, &r, &e, n - 1 + j)) } else { mtp_cpu(|| m.step(&mut kv, &r, &e, n - 1 + j)) }; let th = std::time::Instant::now();
            // the draft head: the main head on the GPU when it is there (full vocabulary), else the CPU subset
            // the draft-vocabulary head on the GPU (PYRE_MTP_HEAD_GPU=0: off): ~100 MB of VRAM, a fraction of the full head's time
            let hg = if env_var("PYRE_MTP_HEAD_GPU").map(|v| v != "0").unwrap_or(true) { self.cube.as_ref().and_then(|cb| self.mtp_headc.get_or_init(|| { let h = m.head_q8(); if h.rows == 0 { return None; } let r = cb.upload_q8(h.rows, h.cols, h.blocks()); if r.is_some() { eprintln!("mtp: draft head on gpu ({} rows)", h.rows); } r }).as_ref().map(|hm| (cb, hm))) } else { None };
            let (id, p) = if let Some((cb, hm)) = hg { let xb = cb.buf(&s); let o = cb.empty(hm.rows); if !(env_var("PYRE_MTP_HEAD_DP4A").map(|v| v == "1").unwrap_or(false) && cb.gemv_q8d(hm, &xb, &o)) { cb.gemv(hm, &xb, s.len(), 0, &o, hm.rows, 0); } let (i, p) = if env_var("PYRE_MTP_BEST_GPU").map(|v| v != "0").unwrap_or(true) { cb.best(&o, hm.rows) } else { super::mtp::best(&cb.read(&o)) }; (m.head_ids()[i.min(hm.rows - 1)], p) }
                else if self.headc.is_some() && env_var("PYRE_MTP_HEAD_CPU").is_err() { let (i, p) = super::mtp::best(&self.head_logits(&s)); (i as u32, p) } else { m.head_cpu(&s) };
            HEAD_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            if !crate::spec::draft_gate(j, p, min_p) { break; }
            out.push((id, None)); r = r2; tok = id;
        }
        kv.truncate(n, row); st.mtp_kv = kv;
        if env_var("PYRE_MTP_DEBUG").is_ok() { eprintln!("mtp: graph replays {} captures {} | head {:.1} ms catch-up {:.1} ms, {} drafts {:.1} ms | {:?}", super::mtp::GG_REPLAYS.load(std::sync::atomic::Ordering::Relaxed), super::mtp::GG_CAPTURES.load(std::sync::atomic::Ordering::Relaxed), HEAD_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6, (t1 - t0).as_secs_f64() * 1e3, out.len(), t1.elapsed().as_secs_f64() * 1e3, super::mtp::TM.iter().map(|a| (a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e5).round() / 10.0).collect::<Vec<_>>()); }
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
        // the draft-vocabulary subset head stays even with the main head on the GPU (A 33.2 vs 30.0 steady; PYRE_MTP_HEAD_FULL=1 drafts with the full GPU head)
        let ids = if self.headc.is_some() && env_var("PYRE_MTP_HEAD_FULL").map(|v| v == "1").unwrap_or(false) { vec![] } else { ids };
        let m = super::mtp::Mtp::load(&p, &geo, ids, |r, y| out.row(r, y))?;
        if env_var("PYRE_MTP_LAZY").is_err() { m.warm(); }
        crate::limits::vlog("mtp loaded"); eprintln!("mtp: drafter {} loaded in {:.1} s", p.display(), t.elapsed().as_secs_f64());
        self.mtp = Some(m);
        // upload the drafter's GPU copies now instead of in the first draft round (PYRE_MTP_EAGER=0: lazy)
        if env_var("PYRE_MTP_EAGER").map(|v| v != "0").unwrap_or(true) { self.mtp_gpu_init(); }
        Ok(())
    }
    fn mtp_gpu_on(&self) -> bool { env_var("PYRE_MTP_GPU").map(|v| v != "0").unwrap_or(true) }
    fn mtp_gpu_init(&self) {
        let (Some(m), Some(cb)) = (self.mtp.as_ref(), self.cube.as_ref()) else { return };
        if self.mtp_gpu_on() { self.mtp_dense.get_or_init(|| { let mut mp = std::collections::HashMap::new(); let mut by = 0usize;
            for q in m.dense_mats() { if let Some(c) = cb.upload_q8(q.rows, q.cols, q.blocks()) { by += q.blocks().len(); mp.insert(q as *const _ as usize, c); } }
            eprintln!("mtp: {} dense drafter matrices on gpu ({:.0} MB)", mp.len(), by as f64 / 1e6); mp }); }
        // compile the drafter's GEMV kernels now (1, 2-4 and >4 columns): a first catch-up of 5+ cells mid-decode otherwise pays ~150 ms of JIT
        if let Some(map) = self.mtp_dense.get() { let t0 = std::time::Instant::now(); for q in m.dense_mats() { if let Some(g) = map.get(&(q as *const _ as usize)) { for t in [1usize, 2, 3, 4, 5, 8, 64] { let xb = cb.buf(&vec![0f32; t * q.cols]); let o = cb.empty(t * q.rows);
            if t == 1 { cb.gemv(g, &xb, q.cols, 0, &o, q.rows, 0); } else { cb.gemv_m(g, &xb, t * q.cols, 0, q.cols, &o, t * q.rows, 0, q.rows, t); } let _ = cb.read(&o); } } }
            eprintln!("mtp: drafter kernels warmed in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3); }
        if env_var("PYRE_MTP_HEAD_GPU").map(|v| v != "0").unwrap_or(true) { self.mtp_headc.get_or_init(|| { let h = m.head_q8(); if h.rows == 0 { return None; } let r = cb.upload_q8(h.rows, h.cols, h.blocks()); if r.is_some() { eprintln!("mtp: draft head on gpu ({} rows)", h.rows); } r }); }
        self.mtp_step_warm();
    }

    // ---- speculative decoding: multi-token verify on the decode (CubeCL) path, with rollback ----

    /// True when `forward_multi` can run: every layer on the CubeCL decode path with DeltaNet on the GPU.
    /// Compile the decode and verify-window kernels now (opt-in PYRE_QN_PREWARM=1; b326: same output, no clear gain: NATIVE2 A tps 30.3 vs 33.1 off): a throwaway state runs a
    /// one-token step and one verify window of every size up to the spec maximum; the expert cache's routing
    /// counts and stats are reset afterwards so the cache fills exactly as without it.
    /// PYRE_MTP_GPUSTEP: compile the GPU draft step's kernels (one throwaway step), so the first draft does not pay ~300 ms of JIT
    fn mtp_step_warm(&self) {
        if !mtp_gpustep() || !self.mtp_gpu_on() { return; }
        let (Some(m), Some(cb)) = (self.mtp.as_ref(), self.cube.as_ref()) else { return };
        let hm = self.mtp_headc.get_or_init(|| { let h = m.head_q8(); if h.rows == 0 { return None; } cb.upload_q8(h.rows, h.cols, h.blocks()) });
        match (self.mtp_dense.get(), hm) { (Some(map), Some(hm)) => { let t0 = std::time::Instant::now(); let g = |q: &super::qdot::Q8Mat| map.get(&(q as *const _ as usize)); m.warm_gpu(cb, &g, hm); eprintln!("mtp: gpu step warmed in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                if super::mtp::gx_on() && env_var("PYRE_MTP_GRAPH").map(|v| v != "0").unwrap_or(true) { let t1 = std::time::Instant::now(); let ok = m.warm_gx(cb, &g, hm, &|t, o| self.embd.row(t as usize, o)); eprintln!("mtp gx: chained draft warmed ({ok}) in {:.0} ms", t1.elapsed().as_secs_f64() * 1e3); } }
            (a, b) => eprintln!("mtp: gpu step NOT warmed (dense {} head {})", a.is_some(), b.is_some()) }
    }
    pub fn prewarm_verify(&self) {
        if env_var("PYRE_QN_PREWARM").map(|v| v == "0").unwrap_or(false) || !self.multi_ok() { return; }
        self.mtp_step_warm();
        let t0 = std::time::Instant::now(); let mut st = self.new_state();
        let _ = self.forward_inner(&[1, 2], &mut st);
        let mw = self.spec_max_window().min(env_usize("PYRE_QN_PREWARM_MAX").unwrap_or(if super::qn_cube::native_mode() == 2 { 7 } else { 3 })); // only the windows decode uses (Q8 T=8 buffers would cost ~2.5 GB VRAM)
        for t in 1..=mw { let ids: Vec<u32> = (0..t as u32).map(|i| 3 + i).collect(); let _ = if t == 1 { self.forward_inner(&ids, &mut st) } else { self.forward_multi(&ids, &mut st).pop().unwrap_or_default() }; }
        if let Some(xm) = &self.xc { let mut g = xm.lock().unwrap(); if !g.map.is_empty() { eprintln!("prewarm: warning, expert cache already filled ({})", g.map.len()); } for c in g.cnt.iter_mut() { for v in c.iter_mut() { *v = 0; } } g.calls = 0; g.hits = 0; g.total = 0; }
        eprintln!("prewarm: decode/verify kernels (T<={mw}) in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
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
    pub fn forward_multi(&self, ids: &[u32], st: &mut State) -> Vec<Vec<f32>> { let xw = xwin_begin(); let r = self.forward_multi_x(ids, st, false); xwin_end(xw, ids.len()); r }
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
        { let qr = std::time::Instant::now(); if let Some(xm) = &self.xc { let mut g = xm.lock().unwrap(); for _ in 0..t { if !g.map.is_empty() || g.calls > 0 { self.xrefill(cb, &mut g); } } } XW[10].fetch_add(qr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        if !st.gdn_gpu { st.gdn = self.gdn_bufs(st); st.gdn_gpu = true; }
        // checkpoint buffers for t-1 positions
        let ch = 2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv; let (cn, sn) = ((c.conv_k - 1) * ch, c.lin_nv * c.lin_dk * c.lin_dv);
        while !st.pf && st.ck.len() < t - 1 { let i = st.ck.len(); let v: Vec<(H, H)> = if vgraph_on() { let mut g = VG_STATE.lock().unwrap(); if g.1.len() <= i { let nv = cb.client.memory_persistent_allocation((), |_| (0..st.gdn.len()).map(|_| (cb.empty(cn), cb.empty(sn))).collect()); g.1.push(nv); } g.1[i].clone() } else { (0..st.gdn.len()).map(|_| (cb.empty(cn), cb.empty(sn))).collect() }; st.ck.push(v); }
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
        // windows of 5..8 tokens run as two groups (4 + rest) the same way, so every group stays on the 2..4-token
        // kernels (one-token kernels per token were ~2x the cost of a 3-token window); PYRE_QN_WAVE8=0 reverts
        // PYRE_QN_T8_SINGLE=n (hybrid, with T8): windows of 5..=n tokens run as one group, larger ones as WAVE8 groups
        let t8s = if super::qn_cube::t8() { env_usize("PYRE_QN_T8_SINGLE").unwrap_or(8) } else { 4 };
        let wave8 = t > 4 && t <= 8 && t > t8s && env_var("PYRE_QN_WAVE8").map(|v| v != "0").unwrap_or(true);
        if (t >= 2 && env_var("PYRE_QN_WAVE").map(|v| v != "0").unwrap_or(false)) || wave8 {
            let ta = if wave8 { 4 } else { t.div_ceil(2) }; let grp = [(0usize, ta), (ta, t)];
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
                if let Some(&(nl, ng)) = stages.get(i + 1) { if !(ng == 0 && self.layers[nl].ple.is_some()) && dq[ng].is_none() {
                    let (n0, n1) = grp[ng]; let q4d = std::time::Instant::now();
                    dq[ng] = Some(self.lcm_dense(nl, &self.layers[nl], self.cl[nl].as_ref().unwrap(), &xs.as_ref().unwrap()[n0..n1], st, ctr[nl].0, ctr[nl].1, start + n0, n0, t));
                    let q4f = std::time::Instant::now(); WV[3].fetch_add((q4f - q4d).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    if wflush() { let _ = cb.client.flush(); } else if kickq() { cb.client.flush_queue(); } WV[4].fetch_add(q4f.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                } }
                let q4b = std::time::Instant::now();
                if prewake() { qn_pool().prewake(); }
                let rall = cb.read_finish(fut); XW[9].fetch_add(q4b.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                HY[1].fetch_add(q4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                WV[g].fetch_add(q4b.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); WV[2].fetch_add((q4b - q4).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                // PIPE: this group's next stage (same group, next layer) is queued behind this stage's host gate
                if pipe_on() && env_var("PYRE_QN_PIPE_WAVE").map(|v| v == "1").unwrap_or(false) && i + 2 < stages.len() && stages[i + 2].1 == g && self.layers[stages[i + 2].0].ple.is_none() {
                    let nl = stages[i + 2].0; let xsr = xs.as_ref().unwrap(); let ctrn = ctr[nl]; let slot = &mut dq[g];
                    let mut f = |s: &mut State| { *slot = Some(self.lcm_dense(nl, &self.layers[nl], self.cl[nl].as_ref().unwrap(), &xsr[o0..o1], s, ctrn.0, ctrn.1, start + o0, o0, t)); };
                    self.lcm_moe_p(il, l, cl, &xsr[o0..o1], st, dd, rall, o0, Some(&mut f));
                } else {
                self.lcm_moe(il, l, cl, &xs.as_ref().unwrap()[o0..o1], st, dd, rall, o0);
                }
            }
        } else {
        let mut pre: Option<(H, H, H)> = None;
        for (il, l) in self.layers.iter().enumerate() {
            if let Some(p) = &l.ple { let qp = std::time::Instant::now();
                if let Some(hs) = xs.take() { xh = hs.iter().map(|h| cb.read(h)).collect(); }
                let mut h = std::mem::take(&mut st.ple_hist[pi]);
                for ti in 0..t { self.ple(p, &mut h, &emb.as_ref().unwrap()[ti * ew..(ti + 1) * ew], &mut xh[ti], 1); if ti + 1 < t && !st.pf { st.ple_ck[pi].push(h.clone()); } }
                st.ple_hist[pi] = h; pi += 1; XW[13].fetch_add(qp.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let cl = self.cl[il].as_ref().unwrap();
            if xs.is_none() { xs = Some(if vgraph_on() { self.vxs_fill(cb, &xh) } else { xh.iter().map(|v| cb.buf(v)).collect() }); }
            self.layer_cube_multi_p(il, l, cl, xs.as_ref().unwrap(), st, &mut ai, &mut li, start, &mut pre);
        }
        }
        if pipe2() { if let Some(cb) = &self.cube { let f = pipe_flag(cb); let _ = PIPE_FLAG_ERR.set(f); } pipe_err_check(); }
        if env_var("PYRE_QN_VGCOUNT").is_ok() { static W: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0); if W.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 10 == 9 { eprintln!("vgcount (graph/eager layers by T): {}", (2..=8).map(|t| format!("T{t} {}/{}", VGT[t][0].load(std::sync::atomic::Ordering::Relaxed), VGT[t][1].load(std::sync::atomic::Ordering::Relaxed))).collect::<Vec<_>>().join(" ")); } }
        if xgate() { xg_err_check(); }
        let qh = std::time::Instant::now(); let hs = xs.unwrap();
        let mut lg: Vec<Vec<f32>> = vec![];
        let headb = env_var("PYRE_QN_HEADB").map(|v| v != "0").unwrap_or(true); let mut hb: Vec<f32> = vec![]; let mut nb = 0usize;
        for (ti, h) in hs.iter().enumerate() { if last_only && ti + 1 < t && self.mtp.is_none() { continue; } let x = cb.read(h); if self.mtp.is_some() { self.mtp_keep_r(st, start + ti, &x); } if last_only && ti + 1 < t { continue; } let (hh, _) = self.hc_mix(&self.head, &x, 1); if headb { hb.extend_from_slice(&hh); nb += 1; } else { lg.push(self.head_logits(&hh)); } }
        // one head GEMV for all verify tokens: the weights are read once; each column uses the same per-row arithmetic
        if headb && nb > 0 { let mut done = 0; while done < nb { let c = (nb - done).min(if self.headc.is_some() { 4 } else { 8 }); let y = if let Some(hm) = &self.headc { /* GPU Q8 head (PYRE_QN_GPUHEAD=1): up to 4 columns per gemv_m, weights read once */ let xb = cb.buf(&hb[done * d..(done + c) * d]); let o = cb.empty(c * hm.rows); if c == 1 { cb.gemv(hm, &xb, d, 0, &o, hm.rows, 0); } else { cb.gemv_m(hm, &xb, c * d, 0, d, &o, c * hm.rows, 0, hm.rows, c); } cb.read(&o) } else { self.mm(&[&self.out], &hb[done * d..(done + c) * d], c).pop().unwrap() }; let r = y.len() / c; for i in 0..c { lg.push(y[i * r..(i + 1) * r].to_vec()); } done += c; } }
        if env_var("PYRE_QN_XWIN2").is_ok() { use std::sync::atomic::Ordering::Relaxed; eprintln!("xwin2 T={t} pool_ms {:.2} pairs {} calls {}", XW2[0].swap(0, Relaxed) as f64 / 1e6, XW2[1].swap(0, Relaxed), XW2[2].swap(0, Relaxed)); }
        XW[11].fetch_add(qh.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); if env_var("PYRE_QN_PROFILE").is_ok() { use std::sync::atomic::Ordering::Relaxed; let f = |i: usize| HY[i].swap(0, Relaxed) as f64 / 1e6;
            eprint!("cpu distinct {} entries {} gpu hits {} | ", MISS[0].swap(0, Relaxed), MISS[1].swap(0, Relaxed), MISS[2].swap(0, Relaxed));
            eprint!("wave wait A {:.1} B {:.1} enqueue-next {:.1} (dense {:.1} flush {:.1}) | ", WV[0].swap(0, Relaxed) as f64 / 1e6, WV[1].swap(0, Relaxed) as f64 / 1e6, WV[2].swap(0, Relaxed) as f64 / 1e6, WV[3].swap(0, Relaxed) as f64 / 1e6, WV[4].swap(0, Relaxed) as f64 / 1e6);
            eprint!("xrun {:.1} upl {:.1} flush {:.1} | ", SD[0].swap(0, Relaxed) as f64 / 1e6, SD[1].swap(0, Relaxed) as f64 / 1e6, SD[2].swap(0, Relaxed) as f64 / 1e6); eprint!("gu first {:.1} last {:.1} wall {:.1} | ", TL[0].swap(0, Relaxed) as f64 / 1e6, TL[1].swap(0, Relaxed) as f64 / 1e6, TL[2].swap(0, Relaxed) as f64 / 1e6); eprint!("route-to-meta {:.1} route {:.1} | ", RTN.swap(0, Relaxed) as f64 / 1e6, RTN2.swap(0, Relaxed) as f64 / 1e6); eprint!("busy {:.1} tasks {} side {:.1} | ", BUSY.swap(0, Relaxed) as f64 / 1e6, BUSYN.swap(0, Relaxed), SIDE_NS.swap(0, Relaxed) as f64 / 1e6);
            if false {}
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
        let w = (2..=every + 1).find(|j| (c + j) % every == 0).map(|j| j - 1).unwrap_or(every);
        // PYRE_QN_XWMAX (XEXACT, default 3, 1..=4): longest verify window; the exact expert path covers t <= 4 only and
        // e21 4-token windows were 124-276 ms vs 82-93 for 3 (NATIVE2 had let them grow to 7)
        if xexact() { w.min(xwmax()) } else { w }
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
    fn layer_cube_multi_p(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, ai: &mut usize, li: &mut usize, start: usize, pre: &mut Option<(H, H, H)>) {
        let (ai0, li0) = (*ai, *li);
        let chained = pre.is_some();
        let dd = if let Some(p) = pre.take() { p } else { match if vgraph_on() { self.lcm_dense_g(il, l, cl, xs, st, li0) } else { None } { Some(x) => { VGT[xs.len().min(8)][0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); x }, None => { VGT[xs.len().min(8)][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); self.lcm_dense(il, l, cl, xs, st, ai0, li0, start, 0, xs.len()) } } };
        match l.mix { Mixer::Attn { .. } => *ai = ai0 + 1, Mixer::Lin { .. } => *li = li0 + 1 }
        let q4 = std::time::Instant::now(); if prewake() { qn_pool().prewake(); } let pf = st.pf; let mut xgp: Option<XgrPend> = None; let xran = std::cell::Cell::new(false);
        let rall = { let cb = self.cube.as_ref().unwrap(); let mut hk = || { xran.set(true); if xgr() && !pf { xgp = self.xgr_launch(il, l, &dd.0, xs.len()); } };
            let v = match xz_read(cb, &dd.0, &mut hk) { Some(v) => v, None => self.xwarm_read(il, l, &dd.0) }; if !xran.get() { hk(); } v };
        XGR_P.with(|p| *p.borrow_mut() = xgp); XW[9].fetch_add(q4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if chained { xl_rdwait(il, xs.len(), q4.elapsed().as_nanos() as u64); }
        HY[1].fetch_add(q4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // int merge: one gate per path. XEXACT verify -> XGATE (moe) continuation rules; else qwen-pipe PIPE rules (PIPE slot is never taken when ex_ok).
        let (an, ln) = (*ai, *li); let nl = il + 1; let t = xs.len(); let mut nx: Option<(H, H, H)> = None;
        let nx_ok = nl < self.layers.len() && !VG_SKIP.lock().unwrap().contains(&(nl, t)) && if xgate() { !st.pf && self.dense_gpu_only(nl, st, start, t) } else { pipe_on() && env_var("PYRE_QN_PIPE_NEXT").map(|v| v != "0").unwrap_or(true) && self.layers[nl].ple.is_none() };
        if nx_ok {
            let mut f = |s: &mut State| { let (ly, cn) = (&self.layers[nl], self.cl[nl].as_ref().unwrap()); super::xchain::dbeg(self.cube.as_ref().unwrap(), matches!(ly.mix, Mixer::Attn { .. }) as u32, t); nx = Some(match if vgraph_on() { self.lcm_dense_g(nl, ly, cn, xs, s, ln) } else { None } { Some(x) => x, None => self.lcm_dense(nl, ly, cn, xs, s, an, ln, start, 0, xs.len()) }); super::xchain::dend(); };
            self.lcm_moe_p(il, l, cl, xs, st, dd, rall, 0, Some(&mut f));
        } else { self.lcm_moe(il, l, cl, xs, st, dd, rall, 0); }
        *pre = nx;
    }
    /// PYRE_QN_XGR: k_xgr_topk + the hit kernels for layer il's window, enqueued at the rd readback (see xgr()).
    fn xgr_launch(&self, il: usize, l: &Layer, rd: &H, t: usize) -> Option<XgrPend> {
        let c = &self.cfg; let cb = self.cube.as_ref()?; let xm = self.xc.as_ref()?; let cl = self.cl[il].as_ref()?;
        let (d, ne, kk) = (c.d, c.n_exp, c.top_k); let rn = d + cl.fin.rows;
        if !(xfast() && combm() && (1..=4).contains(&t) && ne <= 1024 && kk >= 1 && kk < ne && xexact_types(l) && !xallcpu() && !xbal() && xshare() <= 0.0
            && crate::spec::route_prior() <= 0.0 && d % 256 == 0 && c.ff_exp % 64 == 0) { return None; }
        let tyg = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
        let xcg = xm.lock().unwrap();
        if xcg.arena.is_empty() || xcg.arena.len() > 2 { return None; }
        let mut rt = vec![0u32; ne * 4];
        for e in 0..ne { if let Some(ent) = xcg.map.get(&(il, e)) { let m = super::qn_cube::Cube::xmeta(&xcg.arena, &[(ent, 1.0)]); rt[e * 4..e * 4 + 3].copy_from_slice(&m[..3]); rt[e * 4 + 3] = 1; } }
        let rth = { let mut g = XGR_RT.lock().unwrap(); if g.len() <= il { g.resize(il + 1, None); }
            match &g[il] { Some((h, v)) if *v == rt => h.clone(), _ => { let h = cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&rt)); g[il] = Some((h.clone(), rt.clone())); XGR_N[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed); h } } };
        let mx = t * kk; let (mh, oc, dbg, y) = (cb.empty(mx * 4), cb.empty(4), cb.empty(mx), cb.empty(mx * d));
        cb.xgr_topk(rd, t * rn, &rth, ne, &mh, kk, &oc, &dbg, rn, d, t);
        if !cb.xrun_exact(&xcg.tabs, &xcg.arena, xcg.ar_bytes, t, mx, &mh, &oc, rd, t * rn, d, c.ff_exp, tyg, &y, None) { return None; }
        Some(XgrPend { il, t, y, mh, oc, dbg, rt })
    }
    /// XGATE: layer nl's lcm_dense for this window runs entirely on the GPU (no host reads), so it may be queued behind a gate
    fn dense_gpu_only(&self, nl: usize, st: &State, start: usize, t: usize) -> bool {
        let (l, cl, c) = (&self.layers[nl], self.cl[nl].as_ref().unwrap(), &self.cfg);
        if l.ple.is_some() { return false; }
        match l.mix { Mixer::Lin { .. } => cl.gdn.is_some() && st.gdn_gpu, Mixer::Attn { .. } => cl.an.is_some() && attn_gpu(c) && start + t <= self.dense_n() }
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
        if !matches!(l.mix, Mixer::Lin { .. }) || !st.gdn_gpu || st.pf || !((2..=4).contains(&t) || t <= 8 && super::qn_cube::t8()) || c.lin_dk != c.lin_dv || st.ck.len() < t - 1
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
            super::xchain::dmark(cb, "rms_a"); cb.gemv_m(&cl.hca.di, &b.xn, t * w, 0, w, &b.da, t * dar, 0, dar, t);
            super::xchain::dmark(cb, "di_a"); if !(gatem() && cb.gate_m(&b.xn, &b.da, &cl.hca.up, &b.mixed, t * d, 0, hc, d, t, w, dar, d)) { for ti in 0..t { cb.gate(&at(&b.xn, ti * w), &at(&b.da, ti * dar), &cl.hca.up, &at(&b.mixed, ti * d), d, 0, hc, d); } }
            super::xchain::dmark(cb, "gate_a"); cb.gemv_m(&cl.min, &b.mixed, t * d, 0, d, &b.proj, t * pr, 0, pr, t);
            let (mut cks, mut ckm) = (vec![], 0u32);
            super::xchain::dmark(cb, "min"); if xfuse() && convm() && cn == ch * (c.conv_k - 1) {
                // PYRE_QN_XFUSE: the window's convs + conv-state checkpoints in one k_conv_m (as the eager path)
                let mut ckc: Vec<Option<H>> = vec![];
                for ti in 0..t { if ti + 1 < t { let (a, bb) = &cka[ti]; ckc.push(Some(a.clone())); if ti < 8 { ckm |= 1 << ti; } cks.push(Some(bb.clone())); } else { ckc.push(None); cks.push(None); } }
                cb.conv_m(&b.proj, t * pr, pr, &gw.conv, &cs, &b.co, ch, c.conv_k, t, &ckc);
            } else {
            for ti in 0..t { let pj = at(&b.proj, ti * pr); cb.conv(&pj, pr, &gw.conv, &cs, &at(&b.co, ti * ch), ch, c.conv_k);
                if ti + 1 < t { let (a, bb) = &cka[ti]; cb.copy(&cs, cn, 0, a, cn, 0, cn); if ti < 8 { ckm |= 1 << ti; } cks.push(Some(bb.clone())); } else { cks.push(None); } } }
            super::xchain::dmark(cb, "conv"); cb.delta_m(&b.co, ch, &b.proj, pr, gw, &ss, &b.core, core_n, &cks, ckm, nk, nv, dk, sp[1], sp[2], sp[3], c.eps, t);
            super::xchain::dmark(cb, "delta"); cb.gemv_m(&cl.mout, &b.core, t * core_n, 0, core_n, &b.o, t * d, 0, d, t);
            super::xchain::dmark(cb, "mout"); if xfuse() && combm() && t <= 4 { cb.comb_m(xs, &b.o, &b.da, dar, lra, hc, d); } else { for ti in 0..t { cb.comb(&xs[ti], &at(&b.o, ti * d), None, None, &at(&b.da, ti * dar), lra, hc, d); } } super::xchain::dmark(cb, "comb"); cb.rms_t(xs, &cl.hcf.norm, &b.xn, hc, d, c.eps);
            super::xchain::dmark(cb, "rms_f"); cb.gemv_m(&cl.hcf.di, &b.xn, t * w, 0, w, &b.df, t * dfr, 0, dfr, t);
            super::xchain::dmark(cb, "di_f"); if !(gatem() && cb.gate_m(&b.xn, &b.df, &cl.hcf.up, &b.rd, t * rn, 0, hc, d, t, w, dfr, rn)) { for ti in 0..t { cb.gate(&at(&b.xn, ti * w), &at(&b.df, ti * dfr), &cl.hcf.up, &at(&b.rd, ti * rn), rn, 0, hc, d); } }
            super::xchain::dmark(cb, "gate_f"); cb.gemv_m(&cl.fin, &b.rd, t * rn, 0, rn, &b.rd, t * rn, d, rn, t); super::xchain::dmark(cb, "fin"); 
        };
        if b.g.is_some() && b.key == key { unsafe { b.g.as_ref().unwrap().replay() }.expect("graph replay"); VG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); super::xchain::dmark(cb, "graph"); }
        else if gate_open() || xg_open() { VG_EAGER.fetch_add(1, std::sync::atomic::Ordering::Relaxed); VG_SKIP.lock().unwrap().insert((il, t)); if env_var("PYRE_QN_XDEBUG").is_ok() { eprintln!("lg eager behind gate: il {il} t {t} had {}", if b.g.is_some() { "other key" } else { "none" }); } seg(b); }
        else {
            b.g = None; b.keep.clear(); let _ = super::xchain::dpause();
            let vf = |s: &str| if env_var("PYRE_QN_VGDBG").is_ok() { let _ = cb.client.flush(); std::thread::sleep(std::time::Duration::from_millis(30)); if let Some(v) = crate::limits::vram() { eprintln!("vgdbg il {il} t {t} {s}: free {} MiB", v.free >> 20); } };
            vf("pre");
            if cb.client.graph_prepare().is_ok() { vf("prepared"); cb.client.memory_persistent_allocation((), |_| seg(b)); vf("eager");
                match cb.client.start_capture() {
                    Ok(()) => { cb.client.memory_persistent_allocation((), |_| seg(b)); let r = cb.client.stop_capture(); vf("captured"); match r { Ok(g) => { b.g = Some(g); b.key = key; VG_SKIP.lock().unwrap().remove(&(il, t)); if env_var("PYRE_QN_XDEBUG").is_ok() { let n = VG_CAPTURES.load(std::sync::atomic::Ordering::Relaxed); if n < 4 || n % 12 == 0 { let r = cb.client.memory_report(burn_cubecl::cubecl::MemoryScope::CurrentStream); let s = format!("{r:?}"); crate::limits::vlog(&format!("vgraph capture {n}")); eprintln!("vgraph: captured layer {il} T={t} usage {:?} | {}", r.usage(), &s[..s.len().min(1500)]); } } let mut k = vec![cs.clone(), ss.clone()]; k.extend(xs.iter().cloned()); for (a, bb) in &cka { k.push(a.clone()); k.push(bb.clone()); } b.keep = k; VG_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
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
        let core_n = match l.mix { Mixer::Attn { .. } => c.n_head * c.hd, Mixer::Lin { .. } => c.lin_nv * c.lin_dk };
        // PYRE_QN_VGRAPH: attention layers of a verify window run their pre/post-mixer halves as captured graphs on
        // persistent per-(layer, T) buffers (the mixer itself, position-dependent, stays eager)
        let ag = vgraph_on() && env_var("PYRE_QN_VGATTN").map(|v| v == "1").unwrap_or(false) && matches!(l.mix, Mixer::Attn { .. }) && !st.pf && (2..=4).contains(&t);
        let gk: String = if ag { xs.iter().map(|h| format!("{:?}/{:?};", h.memory.descriptor().id, h.offset_start)).collect() } else { String::new() };
        let (xn, da, mixed, proj, core, o, df, rd) = if ag { let mut gm = self.vg.lock().unwrap(); let b = gm.entry((1000 + il, t)).or_insert_with(|| cb.client.memory_persistent_allocation((), |_| VGraph { xn: cb.empty(t * w), da: cb.empty(t * dar), mixed: cb.empty(t * d), proj: cb.empty(t * pr), core: cb.empty(t * core_n), co: cb.empty(1), o: cb.empty(t * d), df: cb.empty(t * dfr), rd: cb.empty(t * rn), g: None, key: String::new(), keep: vec![] }));
            (b.xn.clone(), b.da.clone(), b.mixed.clone(), b.proj.clone(), b.core.clone(), b.o.clone(), b.df.clone(), b.rd.clone()) }
            else { (cb.empty(t * w), cb.empty(t * dar), cb.empty(t * d), cb.empty(t * pr), cb.empty(t * core_n), cb.empty(t * d), cb.empty(t * dfr), cb.empty(t * rn)) };
        let pre = || {
        cb.rms_t(xs, &cl.hca.norm, &xn, hc, d, c.eps);
        super::xchain::dmark(cb, "rms_a"); cb.gemv_m(&cl.hca.di, &xn, t * w, 0, w, &da, t * dar, 0, dar, t);
        super::xchain::dmark(cb, "di_a"); if !(gatem() && cb.gate_m(&xn, &da, &cl.hca.up, &mixed, t * d, 0, hc, d, t, w, dar, d)) { for ti in 0..t { cb.gate(&at(&xn, ti * w), &at(&da, ti * dar), &cl.hca.up, &at(&mixed, ti * d), d, 0, hc, d); } }
        super::xchain::dmark(cb, "gate_a"); cb.gemv_m(&cl.min, &mixed, t * d, 0, d, &proj, t * pr, 0, pr, t);
        };
        if ag { vg_run(cb, (1, il, t), &gk, &pre); } else { pre(); } super::xchain::dmark(cb, "min"); 
        // the mixer, token by token (recurrent / causal state)
        // PYRE_QN_DELTAM (default on): the DeltaNet recurrence of all t tokens in one k_delta_m launch (the convs
        // and their state checkpoints stay per token, in the same order)
        let mut dm_done = false;
        if let (Mixer::Lin { .. }, Some(gw), true) = (&l.mix, &cl.gdn, st.gdn_gpu) { if t >= 2 && (t <= 4 || t <= 8 && super::qn_cube::t8()) && c.lin_dk == c.lin_dv && env_var("PYRE_QN_DELTAM").map(|v| v != "0").unwrap_or(true) {
            let (nk, nv, dk) = (c.lin_nk, c.lin_nv, c.lin_dk); let ch = cl.min.splits[1]; let sp = &cl.min.splits;
            let (cs, ss) = st.gdn[li0].clone(); let co_all = cb.empty(t * ch);
            let cn = (c.conv_k - 1) * (2 * c.lin_nk * c.lin_dk + c.lin_nv * c.lin_dv);
            let (mut cks, mut ckm) = (vec![], 0u32);
            if convm() && cn == ch * (c.conv_k - 1) {
                // one k_conv_m launch for the window's convs and conv-state checkpoints (same per-token arithmetic and order)
                let mut cka: Vec<Option<H>> = vec![];
                for ti in 0..t { if off + ti + 1 < ttot && !st.pf { let (a, b) = &st.ck[off + ti][li0]; cka.push(Some(a.clone())); if ti < 8 { ckm |= 1 << ti; } cks.push(Some(b.clone())); } else { cka.push(None); cks.push(None); } }
                cb.conv_m(&proj, t * pr, pr, &gw.conv, &cs, &co_all, ch, c.conv_k, t, &cka);
            } else {
            for ti in 0..t { let pj = at(&proj, ti * pr); cb.conv(&pj, pr, &gw.conv, &cs, &at(&co_all, ti * ch), ch, c.conv_k);
                if off + ti + 1 < ttot && !st.pf { let (a, b) = &st.ck[off + ti][li0]; cb.copy(&cs, cn, 0, a, cn, 0, cn); if ti < 8 { ckm |= 1 << ti; } cks.push(Some(b.clone())); } else { cks.push(None); } }
            }
            super::xchain::dmark(cb, "conv"); cb.delta_m(&co_all, ch, &proj, pr, gw, &ss, &core, core_n, &cks, ckm, nk, nv, dk, sp[1], sp[2], sp[3], c.eps, t);
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
                let csb = self.rope_h(cb, pos, half);
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
        super::xchain::dmark(cb, "mixer"); let post = || {
        cb.gemv_m(&cl.mout, &core, t * core_n, 0, core_n, &o, t * d, 0, d, t);
        // FFN hyper-connection read + router / shared expert projection, batched
        super::xchain::dmark(cb, "mout"); if combm() && (t <= 4 || super::qn_cube::t8()) { cb.comb_m(xs, &o, &da, dar, lra, hc, d); } else { for ti in 0..t { cb.comb(&xs[ti], &at(&o, ti * d), None, None, &at(&da, ti * dar), lra, hc, d); } } super::xchain::dmark(cb, "comb"); cb.rms_t(xs, &cl.hcf.norm, &xn, hc, d, c.eps);
        super::xchain::dmark(cb, "rms_f"); cb.gemv_m(&cl.hcf.di, &xn, t * w, 0, w, &df, t * dfr, 0, dfr, t);
        super::xchain::dmark(cb, "di_f"); if !(gatem() && cb.gate_m(&xn, &df, &cl.hcf.up, &rd, t * rn, 0, hc, d, t, w, dfr, rn)) { for ti in 0..t { cb.gate(&at(&xn, ti * w), &at(&df, ti * dfr), &cl.hcf.up, &at(&rd, ti * rn), rn, 0, hc, d); } }
        super::xchain::dmark(cb, "gate_f"); cb.gemv_m(&cl.fin, &rd, t * rn, 0, rn, &rd, t * rn, d, rn, t); super::xchain::dmark(cb, "fin"); 
        };
        if ag { vg_run(cb, (2, il, t), &gk, &post); } else { post(); }
        HY[0].fetch_add(q0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        (rd, df, da)
    }
    /// The expert half of a verify layer: routing from `rall` (rd read back), cached experts on the GPU beside the
    /// CPU misses, and the residual update of xs.
    fn lcm_moe(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, dd: (H, H, H), rall: Vec<f32>, off: usize) { self.lcm_moe_p(il, l, cl, xs, st, dd, rall, off, None); }
    /// `lcm_moe` with the PYRE_QN_PIPE continuation: `next` (the next layer's dense enqueue) runs on the host thread while the CPU pool works. Returns whether `next` ran.
    #[allow(clippy::too_many_arguments)]
    fn lcm_moe_p(&self, il: usize, l: &Layer, cl: &super::qn_cube::CLayer, xs: &[burn_cubecl::cubecl::server::Handle], st: &mut State, dd: (H, H, H), rall: Vec<f32>, off: usize, mut next: Option<&mut dyn FnMut(&mut State)>) -> bool {
        let mut piped = false;
        type H = burn_cubecl::cubecl::server::Handle;
        let t = xs.len();
        let cb = self.cube.as_ref().unwrap(); let c = &self.cfg; let (d, hc, ne) = (c.d, c.hc, c.n_exp);
        let lrf = cl.hcf.up.cols; let rn = d + cl.fin.rows; let dfr = cl.hcf.di.rows;
        let at = |h: &H, off: usize| -> H { h.clone().offset_start((off * 4) as u64) };
        let (rd, df, _da) = dd;
        let mut xgp = XGR_P.with(|p| p.borrow_mut().take()).filter(|p| p.il == il && p.t == t);
        let q_rt = std::time::Instant::now(); XHX_T.with(|c| c.set(Some(q_rt))); let rvs: Vec<Vec<f32>> = (0..t).map(|ti| rall[ti * rn..(ti + 1) * rn].to_vec()).collect();
        let dfs: Vec<H> = (0..t).map(|ti| at(&df, ti * dfr)).collect();
        let rds: Vec<H> = (0..t).map(|ti| at(&rd, ti * rn)).collect();
        let q5 = std::time::Instant::now();
        let sf = l.sh_gate.rows;
        let mut yr = vec![0f32; t * d];
        let mut ygs: Vec<Option<burn_cubecl::cubecl::server::Handle>> = vec![None; t];
        let mut shs: Vec<burn_cubecl::cubecl::server::Handle> = Vec::with_capacity(t);
        let (hb_all, sh_all) = (cb.empty(t * sf), cb.empty(t * d)); let mut sh_b = false; let mut sh_g: Option<H> = None; let mut ex_y: Option<H> = None; let mut ex_tab: Option<Vec<u32>> = None;
        let xin: Vec<f32> = rvs.iter().flat_map(|r| r[..d].iter().copied()).collect();
        if let Some(xm) = &self.xc {
            let mut xcg = xm.lock().unwrap();
            let (mut ons, mut offs) = (Vec::with_capacity(t), Vec::with_capacity(t)); let mut sels: Vec<Vec<usize>> = Vec::with_capacity(t);
            for (ti, rv) in rvs.iter().enumerate() {
                let sel = self.route_sel(il, &rv[d..d + ne], Some(&xcg)); sels.push(sel.iter().map(|x| x.0).collect());
                for &(e, _) in &sel { xcg.cnt[il][e] += xinc(); if let Some(v) = st.spec_sel.get_mut(off + ti) { v.push((il, e)); } }
                xcg.total += sel.len() as u64;
                let (on, mut off): (Vec<(usize, f32)>, Vec<(usize, f32)>) = sel.into_iter().partition(|(e, _)| !xallcpu() && !(xexact() && !xexact_types(l)) && xcg.map.contains_key(&(il, *e)));
            tail_skip(&mut off, &on);
            // Strata-style PCIe share: the first few misses are copied into the stream slots and run on the GPU
            // with the cached ones, while the CPU pool takes the rest
            let tys0 = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
            let _ = tys0;
                xcg.hits += on.len() as u64; XW[1].fetch_add(on.len() as u64, std::sync::atomic::Ordering::Relaxed); crate::metrics::expert_cache(on.len() as u64, off.len() as u64); ons.push(on); offs.push(off);
            }
            // PYRE_QN_XGR: the GPU's hit entries are in ascending expert order per token (entry order does not change any
            // entry's arithmetic); valid only when no token has a tie at the top-k boundary and the GPU's table matches the cache
            if let Some(p) = xgp.as_ref() {
                let mut ok = (true, true);
                for ti in 0..t { ons[ti].sort_by_key(|x| x.0);
                    let r = &rvs[ti][d..d + ne]; let s = &sels[ti];
                    let mn = s.iter().map(|&e| r[e]).fold(f32::INFINITY, f32::min);
                    if (0..ne).any(|e| !s.contains(&e) && !(r[e] < mn)) { ok.0 = false; }
                    for &e in s { let on = ons[ti].iter().any(|x| x.0 == e); if on != (p.rt[e * 4 + 3] != 0) { ok.1 = false; }
                        if on { let m = super::qn_cube::Cube::xmeta(&xcg.arena, &[(&xcg.map[&(il, e)], 1.0)]); if m[..3] != p.rt[e * 4..e * 4 + 3] { ok.1 = false; } } } }
                if !ok.0 { XGR_N[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); } else if !ok.1 { XGR_N[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                if !(ok.0 && ok.1) { xgp = None; } else { XGR_N[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            }
            RTN2.fetch_add(q_rt.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); XW[14].fetch_add(q_rt.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            // PYRE_QN_XCMP=N (diagnostic): for the first N layer calls, token 0's GPU-cached experts are also run alone
            // through the CPU path and the GPU fast path (weight 1) on the same input row; prints how many of the d
            // outputs differ in bits and the largest difference (placement-dependence of the expert math)
            if let Some(nmax) = env_usize("PYRE_QN_XCMP") { static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                if !ons[0].is_empty() && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < nmax {
                    let tyf = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                    for &(e, _) in ons[0].iter().take(3) { let ent = &xcg.map[&(il, e)];
                        let y0 = cb.empty(d); let ok = if xexact() { let mh = cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&super::qn_cube::Cube::xmeta(&xcg.arena, &[(ent, 1.0)]))); let oc = cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&[1u32, 1, 1, 1]));
                            cb.xrun_exact(&xcg.tabs, &xcg.arena, xcg.ar_bytes, 1, 1, &mh, &oc, &rds[0], rn, d, c.ff_exp, tyf, &y0, None) } else { cb.xrun_fast_to(&xcg.tabs, &xcg.arena, xcg.ar_bytes, &[(ent, 1.0)], &rds[0], rn, d, c.ff_exp, tyf, &xcg.scr, false, &y0) };
                        if !ok { eprintln!("xcmp: layer {il} expert {e}: gpu fast path not eligible"); continue; }
                        let yg = cb.read(&y0); let mut yc = vec![0f32; d]; let wv = vec![(0usize, 1.0f32)];
                        self.routed_pool_ord(l, &xin[..d], &[(e, &wv)], 1, &mut yc, &mut || {}, None);
                        let nd = (0..d).filter(|&i| yg[i].to_bits() != yc[i].to_bits()).count();
                        let md = (0..d).map(|i| (yg[i] - yc[i]).abs()).fold(0f32, f32::max); let my = yc.iter().fold(0f32, |m, v| m.max(v.abs()));
                        eprintln!("xcmp: layer {il} expert {e} types {:?}/{:?}/{:?}: {nd}/{d} outputs differ, max |diff| {md:.3e} (max |y| {my:.3e}, rel {:.2e})", l.gate_e.ty, l.up_e.ty, l.down_e.ty, md / my.max(1e-30));
                    } } }
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
                        // PYRE_QN_XCOPYS=1: the share copies go on their own CUDA stream (as Strata's copy stream); CubeCL makes the
                        // compute stream wait for the arena writes before the expert kernels read them
                        let up = || xcg.up(cb, sl, &pk.0, pk.1, pk.2);
                        let mut ent = if xcopys() { cubecl_environment::stream::StreamId { value: 0x5EED_0001 }.executes(up) } else { up() }; ent.arena = xcg.loc(sl).0; ent.slot = sl;
                        xcg.map.insert((il, e), ent); sins.push((il, e)); }
                    SD[1].fetch_add(qu.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); let set: std::collections::HashSet<usize> = v[..k].iter().map(|x| x.0).collect();
                    for ti in 0..t { let (mv, keep): (Vec<(usize, f32)>, Vec<(usize, f32)>) = offs[ti].drain(..).partition(|(e, _)| set.contains(e)); offs[ti] = keep; ons[ti].extend(mv); }
                }
            }
            let xcr = &*xcg;
            let tyg = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
            let mg_ok = moegraph() && vgraph_on() && xbatch() && combm() && xfast() && xmeta1() && !st.pf && (2..=4).contains(&t) && xcr.arena.len() <= 2 && !xcr.arena.is_empty()
                && tyg.0 == tyg.1 && tyg.0 <= 1 && tyg.2 == 3 && d % 256 == 0 && c.ff_exp % 32 == 0;
            let ex_ok = xexact() && xfast() && combm() && (1..=4).contains(&t) && xcr.arena.len() <= 2 && !xcr.arena.is_empty() && tyg.0 == tyg.1 && tyg.0 <= 1 && tyg.2 == 3 && d % 256 == 0 && c.ff_exp % 64 == 0;
            if ex_ok && (xbal() || xlstat()) && il < 128 && !st.pf {
                use std::sync::atomic::Ordering::Relaxed;
                let mut cnt: Vec<(usize, usize)> = vec![]; for o in &ons { for &(e, _) in o { match cnt.iter_mut().find(|x| x.0 == e) { Some(x) => x.1 += 1, None => cnt.push((e, 1)) } } }
                XL_ST[il][0].fetch_add(1, Relaxed); XL_ST[il][6].fetch_add(cnt.len() as u64, Relaxed);
                let k = if xbal() { xl_idx(il, t).map(|i| XL_K[i].load(Relaxed) as usize).unwrap_or(0) } else { 0 };
                let k = k.min(cnt.len()); if xbal() { if let Some(i) = xl_idx(il, t) { if XL_K[i].load(Relaxed) as usize > cnt.len() { XL_K[i].store(cnt.len() as u64, Relaxed); } } }
                if k > 0 { cnt.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0))); let mv: Vec<usize> = cnt[..k].iter().map(|x| x.0).collect();
                    for ti in 0..t { let (m, keep): (Vec<(usize, f32)>, Vec<(usize, f32)>) = ons[ti].drain(..).partition(|(e, _)| mv.contains(e)); ons[ti] = keep; offs[ti].extend(m); }
                    XL_ST[il][4].fetch_add(k as u64, Relaxed); }
                if il == 0 { static NC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); let n = NC.fetch_add(1, Relaxed) + 1; if xlstat() && n % 32 == 0 { xl_print(); } }
            }
            // union of the missed experts; each token keeps its own order for the accumulation
            let mut lists: Vec<(usize, Vec<(usize, f32)>)> = vec![]; let mut pos_of: std::collections::HashMap<usize, usize> = Default::default();
            let mut ord: Vec<Vec<(usize, usize)>> = vec![vec![]; t];
            for (ti, off) in offs.iter().enumerate() { for &(e, wt) in off {
                let wi = *pos_of.entry(e).or_insert_with(|| { lists.push((e, vec![])); lists.len() - 1 });
                ord[ti].push((wi, lists[wi].1.len())); lists[wi].1.push((ti, wt)); } }
            if ex_ok {
                // per token, in routing (route_sel) order: bit 31 = CPU row (slot of the miss list), else GPU entry index
                let kk = c.top_k; let mut tab = vec![0u32; t * (kk + 1)]; let mut slot0 = vec![0usize; lists.len()]; let mut sn = 0usize; for (wi, x) in lists.iter().enumerate() { slot0[wi] = sn; sn += x.1.len(); }
                let mut eo = 0usize;
                for ti in 0..t { let mut n = 0usize;
                    for &e in &sels[ti] {
                        let code = if let Some(p) = ons[ti].iter().position(|x| x.0 == e) { Some((eo + p) as u32) }
                            else if let Some(&wi) = pos_of.get(&e) { lists[wi].1.iter().position(|x| x.0 == ti).map(|j| (1u32 << 31) | (slot0[wi] + j) as u32) } else { None };
                        if let Some(cd) = code { if n < kk { tab[ti * (kk + 1) + 1 + n] = cd; n += 1; } } }
                    tab[ti * (kk + 1)] = n as u32; eo += ons[ti].len(); }
                ex_tab = Some(tab);
            }
            // XGATE: gate this layer when the CPU pool has enough rows
            let sn_all: usize = lists.iter().map(|x| x.1.len()).sum();
            let xslot = if ex_ok && xgate() && sn_all >= env_usize("PYRE_QN_XGATE_MINROWS").unwrap_or(4).max(1) && ex_tab.is_some() { xg_flag(cb).and_then(|_| xg_slot(cb, sn_all * d * 4, 4 * c.top_k * d * 4)) } else { None };
            if ex_ok { XG_N[xslot.is_none() as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            let xg_tg = std::cell::Cell::new(0u32); let xg_guard = XGOpen(XG_FLAG.get().map(|f| f.0).unwrap_or(0), &xg_tg);
            // int merge: PIPE stays off where the XEXACT / MOEGRAPH verify paths sum on the GPU (their own combine)
            // PYRE_QN_PIPE_MINROWS: gate only layers with at least this many CPU (token, expert) rows; below it the gate/spin overhead outweighs the overlap
            let pslot = if pipe_on() && !ex_ok && !mg_ok && offs.iter().map(|o| o.len()).sum::<usize>() >= env_usize("PYRE_QN_PIPE_MINROWS").unwrap_or(4).max(1) { pipe_slot(cb, t * d * 4) } else { None };
            piped = pslot.is_some();
            let pgate = std::sync::Arc::new(PGate { m: std::sync::Mutex::new(false), cv: std::sync::Condvar::new(), tgt: std::sync::Mutex::new(None) });
            let pg_open = PGateOpen(pgate.clone());
            let stp: &mut State = st;
            // PYRE_QN_XEXACT: the hits with the CPU arithmetic, one weighted row per entry (summed later with the CPU rows in routing order)
            let zt_tb: std::cell::Cell<Option<H>> = std::cell::Cell::new(None);
            let hits = || -> Option<H> {
                if let Some(p) = xgp.as_ref() {
                    let wts: Vec<u32> = ons.iter().flat_map(|o| o.iter().map(|x| x.1.to_bits())).collect();
                    if xgr_chk() {
                        let (gm, go, gd) = (cb.read(&p.mh), cb.read(&p.oc), cb.read(&p.dbg)); let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                        let (gm, go, gd) = (b(&gm), b(&go), b(&gd)); let kk = c.top_k; let mut bad = false; let mut acc = 0usize;
                        for ti in 0..t { let mut s: Vec<u32> = sels[ti].iter().map(|&e| e as u32).collect(); s.sort(); if gd[ti * kk..ti * kk + kk] != s[..] { bad = true; }
                            for (j, &(e, _)) in ons[ti].iter().enumerate() { let m = super::qn_cube::Cube::xmeta(&xcr.arena, &[(&xcr.map[&(il, e)], 1.0)]); if gm[(acc + j) * 4..(acc + j) * 4 + 4] != m[..] { bad = true; } }
                            acc += ons[ti].len(); if go[ti] as usize != acc { bad = true; } }
                        if (t..4).any(|ti| go[ti] as usize != acc) { bad = true; }
                        if bad { if XGR_N[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 { eprintln!("xgr_chk MISMATCH layer {il} t {t}: gpu sel {:?} oc {:?} | host sel {:?} hits {:?}", gd, go, sels, ons.iter().map(|o| o.iter().map(|x| x.0).collect::<Vec<_>>()).collect::<Vec<_>>()); } }
                    }
                    if wts.is_empty() { return None; }
                    let wh = if xzt() { let tabw: &[u32] = if xslot.is_some() { ex_tab.as_deref().unwrap_or(&[]) } else { &[] }; let o1 = tabw.len().div_ceil(4) * 4;
                        let mut w = vec![0u32; o1 + wts.len()]; w[..tabw.len()].copy_from_slice(tabw); w[o1..].copy_from_slice(&wts);
                        let h = xt_up(cb, &w); if !tabw.is_empty() { zt_tb.set(Some(h.clone())); } h.offset_start((o1 * 4) as u64) }
                        else { cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&wts)) };
                    cb.xgr_scale(&p.y, &wh, wts.len(), d);
                    return Some(p.y.clone());
                }
                let mut all: Vec<u32> = vec![]; let mut oc = [0u32; 4]; let mut acc = 0u32;
                for (ti, o) in oc.iter_mut().enumerate() { if ti < t { let ents: Vec<(&super::qn_cube::XEnt, f32)> = ons[ti].iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect(); all.extend(super::qn_cube::Cube::xmeta(&xcr.arena, &ents)); acc += ents.len() as u32; } *o = acc; }
                if acc == 0 { return None; }
                // XZT: routing table (gated layers), hit meta and counts in one zero-copy upload (one blocking server call per layer)
                let (mh, och) = if xzt() { let tabw: &[u32] = if xslot.is_some() { ex_tab.as_deref().unwrap_or(&[]) } else { &[] };
                    let (o1, o2) = (tabw.len().div_ceil(4) * 4, tabw.len().div_ceil(4) * 4 + all.len().div_ceil(4) * 4);
                    let mut w = vec![0u32; o2 + 4]; w[..tabw.len()].copy_from_slice(tabw); w[o1..o1 + all.len()].copy_from_slice(&all); w[o2..o2 + 4].copy_from_slice(&oc);
                    let h = xt_up(cb, &w); if !tabw.is_empty() { zt_tb.set(Some(h.clone())); } (h.clone().offset_start((o1 * 4) as u64), h.offset_start((o2 * 4) as u64)) }
                    else { (cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&all)), cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&oc))) };
                let yent = cb.empty(acc as usize * d);
                let q = std::time::Instant::now(); cb.xrun_exact(&xcr.tabs, &xcr.arena, xcr.ar_bytes, t, acc as usize, &mh, &och, &rd, t * rn, d, c.ff_exp, tyg, &yent, None); SD[0].fetch_add(q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); XW[12].fetch_add(q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                Some(yent)
            };
            // PYRE_QN_XEARLY=1 (opt-in): the GPU hits are enqueued before the CPU pool's activation setup, not in its side task
            let early = ex_ok && xearly() && !lists.is_empty();
            if early { ex_y = hits(); }
            let mut gpu = || {
                let hx0 = std::time::Instant::now(); XHX[0].fetch_add((hx0 - q_rt).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                if ex_ok {
                    if !early { ex_y = hits(); }
                    XHX[1].fetch_add(hx0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    cb.silu_mul_m(&rd, t * rn, rn, d + ne, d + ne + sf, &hb_all, sf, t);
                    cb.gemv_m(&cl.sdown, &hb_all, t * sf, 0, sf, &sh_all, t * d, 0, d, t);
                    for ti in 0..t { shs.push(at(&sh_all, ti * d)); } sh_b = true;
                    if let (Some(ps), Some(&(fl, er)), Some(tab)) = (xslot, XG_FLAG.get(), ex_tab.as_ref()) {
                        // XGATE: tables up first, then the bounded wait, the CPU rows, the sum + combine and the next layer's dense
                        let tb = match zt_tb.take() { Some(h) => h, None => xt_up(cb, tab) }; let ycpu = cb.empty(sn_all * d); let ysum = cb.empty(t * d);
                        let tg = XG_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed).wrapping_add(1).max(1); xg_tg.set(tg);
                        let xchr = super::xchain::rec(cb, sn_all * d * 4); if let Some(a) = xchr { super::xchain::stamp(cb, a); }
                        { let tmo = env_usize("PYRE_QN_XGATE_SPIN_US").unwrap_or(100_000) as u64 * 1000;
                          let ckp = if xck() && xpdirect() && xzt() && xzt_check() && (sn_all * d * 4) % 16 == 0 && ps % 16 == 0 && (xck_blk() * 4) % 16 == 0 && xzc() & 1 != 0 && !XZ_ERR.load(std::sync::atomic::Ordering::Relaxed) { xck_page(cb) } else { None };
                          let zc = if let Some(cp) = ckp { unsafe { cb.client.pyre_wait_copy_chunks_async(fl, cp, tg, tmo, er, ps, sn_all, d * 4, xck_blk() * 4, &ycpu, XZT_FAIL.as_ptr() as usize); }
                              XCK_TG.store(tg, std::sync::atomic::Ordering::Release); XCK_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed); true } else {
                          xzc() & 1 != 0 && !XZ_ERR.load(std::sync::atomic::Ordering::Relaxed) && if xzt() && xzt_check() && (sn_all * d * 4) % 16 == 0 && ps % 16 == 0 { unsafe { cb.client.pyre_wait_copy_from_mapped_async(fl, tg, tmo, er, ps, sn_all * d * 4, &ycpu, XZT_FAIL.as_ptr() as usize); } Ok(()) } else { unsafe { cb.client.pyre_wait_copy_from_mapped(fl, tg, tmo, er, ps, sn_all * d * 4, &ycpu) } }.map_err(|e| { eprintln!("PYRE_QN_XZC: pyre_wait_copy_from_mapped failed ({e}); rows zero-copy off"); XZ_ERR.store(true, std::sync::atomic::Ordering::Relaxed); }).is_ok() };
                          if zc { XZ_N[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); } else { unsafe { cb.client.pyre_wait_flag(fl, tg, tmo, er); cb.client.pyre_host_gate(None, ps, sn_all * d * 4, &ycpu); } } }
                        if let Some(a) = xchr { super::xchain::stamp(cb, a + 8); }
                        XG_OPEN.with(|g| g.set(true));
                        if xfuse() { let _ = &ysum; cb.rsum_comb3_m(ex_y.as_ref().unwrap_or(&ycpu), &ycpu, &tb, c.top_k, xs, &sh_all, &rd, t * rn, rn, d + ne + 2 * sf, &df, dfr, lrf, hc, d); } else {
                        cb.rsum_ex(ex_y.as_ref().unwrap_or(&ycpu), &ycpu, &tb, &ysum, d, c.top_k, t);
                        let none: Vec<Option<H>> = vec![None; t];
                        cb.comb3_m(xs, &none, &sh_all, &ysum, &rd, t * rn, rn, d + ne + 2 * sf, &df, dfr, lrf, hc, d); }
                        if let Some(a) = xchr { super::xchain::stamp(cb, a + 16); }
                        if let Some(nx) = next.as_mut() { nx(&mut *stp); }
                        if let Some(a) = xchr { super::xchain::stamp(cb, a + 24); }
                        cb.client.flush_queue();
                        XHX[2].fetch_add(hx0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                    let qf = std::time::Instant::now(); if kick() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } } SD[2].fetch_add(qf.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
                if mg_ok {
                    let maxne = t * c.top_k; let mut all: Vec<u32> = Vec::with_capacity(maxne * 4); let mut oc = [0u32; 4]; let mut acc = 0u32;
                    for (ti, o) in oc.iter_mut().enumerate() { if ti < t { let ents: Vec<(&super::qn_cube::XEnt, f32)> = ons[ti].iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect(); all.extend(super::qn_cube::Cube::xmeta(&xcr.arena, &ents)); acc += ents.len() as u32; } *o = acc; }
                    if all.len() <= maxne * 4 {
                        all.resize(maxne * 4, 0);
                        let mg = mg_bufs(cb, t, maxne, d, sf, rn);
                        cb.client.write(&mg.meta, burn_cubecl::cubecl::bytes::Bytes::from_elems(all)); cb.client.write(&mg.oc, burn_cubecl::cubecl::bytes::Bytes::from_elems(oc.to_vec()));
                        // rd straight from the dense half when its buffer was the same last time for this (layer, T) (the VGRAPH
                        // DeltaNet graphs own a fixed rd); otherwise copied into the fixed buffer so the graph is not recaptured
                        let rid = format!("{:?}/{:?}", rd.memory.descriptor().id, rd.offset_start);
                        let stable = { let mut m = MG_RD.lock().unwrap(); let e = m.entry((il, t)).or_default(); let same = *e == rid; *e = rid.clone(); same };
                        let src = if stable { rd.clone() } else { cb.copy(&rd, t * rn, 0, &mg.rd, t * rn, 0, t * rn); mg.rd.clone() };
                        let key: String = xcr.arena.iter().map(|h| format!("{:?};", h.memory.descriptor().id)).collect::<String>() + &format!("{:?}/{:?}", src.memory.descriptor().id, src.offset_start);
                        let q = std::time::Instant::now();
                        let f = || { cb.xrun_graph_multi(&xcr.tabs, &xcr.arena, xcr.ar_bytes, t, maxne, &mg.meta, &mg.oc, &src, t * rn, d, c.ff_exp, tyg, &mg.y);
                            cb.silu_mul_m(&src, t * rn, rn, d + ne, d + ne + sf, &mg.hb, sf, t);
                            cb.gemv_m(&cl.sdown, &mg.hb, t * sf, 0, sf, &mg.sh, t * d, 0, d, t); };
                        vg_run(cb, (3, il, t), &key, &f); { let n = MG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); if n == 0 || n == 5000 { eprintln!("moegraph: run {n} (layer {il} t {t} maxne {maxne}), vgraph captures so far {}", VG_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)); } }
                        SD[0].fetch_add(q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        for ti in 0..t { ygs[ti] = Some(at(&mg.y, ti * d)); shs.push(at(&mg.sh, ti * d)); } sh_b = true; sh_g = Some(mg.sh.clone());
                        let qf = std::time::Instant::now(); if kick() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } } SD[2].fetch_add(qf.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                }
                // PYRE_QN_XMETA1=0 reverts: the window's expert tables in one upload (one small write per layer, not per token)
                let mut moff = vec![usize::MAX; t]; let mut mall: Option<H> = None;
                if xmeta1() && xfast() && xcr.arena.len() <= 2 {
                    let mut all: Vec<u32> = vec![];
                    for ti in 0..t { if ons[ti].is_empty() { continue; } let ents: Vec<(&super::qn_cube::XEnt, f32)> = ons[ti].iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect(); moff[ti] = all.len(); all.extend(super::qn_cube::Cube::xmeta(&xcr.arena, &ents)); }
                    RTN.fetch_add(q_rt.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    if !all.is_empty() { mall = Some(cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&all))); }
                }
                // PYRE_QN_XBATCH (default on): the GPU expert hits of all window tokens in 4 launches, not 4 per token
                let mut done_b = false;
                if xbatch() && (t <= 4 || super::qn_cube::t8()) { if let Some(mh) = &mall {
                    let tyf = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                    let cnt: Vec<usize> = ons.iter().map(|o| o.len()).collect(); let yall = cb.empty(t * d);
                    if cb.xrun_fast_multi(&xcr.tabs, &xcr.arena, xcr.ar_bytes, &cnt, mh, &rd, t * rn, d, c.ff_exp, tyf, &yall) { for ti in 0..t { if !ons[ti].is_empty() { ygs[ti] = Some(at(&yall, ti * d)); } } done_b = true; }
                } }
                for ti in 0..t {
                    let (rd, on) = (&rds[ti], &ons[ti]);
                    let tyf = (super::qn_cube::xtype(l.gate_e.ty).unwrap_or(9), super::qn_cube::xtype(l.up_e.ty).unwrap_or(9), super::qn_cube::xtype(l.down_e.ty).unwrap_or(9));
                    let fast = !on.is_empty() && (done_b || xfast() && tyf.0 == tyf.1 && tyf.0 <= 1 && tyf.2 == 3 && {
                        let y0 = cb.empty(d);
                        let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect();
                        let mut first = true; let mut ok = true;
                        if xcr.arena.len() <= 2 { let q = std::time::Instant::now(); ok = cb.xrun_fast_to_m(&xcr.tabs, &xcr.arena, xcr.ar_bytes, &ents, rd, rn, d, c.ff_exp, tyf, &xcr.scr, false, &y0, mall.as_ref().filter(|_| moff[ti] != usize::MAX).map(|h| h.clone().offset_start((moff[ti] * 4) as u64))); SD[0].fetch_add(q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); } else {
                        for ar in 0..xcr.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ar).cloned().collect(); if grp.is_empty() { continue; }
                            ok &= cb.xrun_fast_to(&xcr.tabs, std::slice::from_ref(&xcr.arena[ar]), xcr.ar_bytes, &grp, rd, rn, d, c.ff_exp, tyf, &xcr.scr, !first, &y0); first = false; } }
                        if ok { ygs[ti] = Some(y0); } ok });
                    if !on.is_empty() && !fast {
                        let xb = cb.empty(d); let y0 = cb.empty(d);
                        cb.q8copy(rd, rn, 0, &xb, &y0, d);
                        let tys = (super::qn_cube::xtype(l.gate_e.ty).unwrap(), super::qn_cube::xtype(l.up_e.ty).unwrap(), super::qn_cube::xtype(l.down_e.ty).unwrap());
                        let ents: Vec<(&super::qn_cube::XEnt, f32)> = on.iter().map(|(e, w)| (&xcr.map[&(il, *e)], *w)).collect();
                        for ar in 0..xcr.arena.len() { let grp: Vec<(&super::qn_cube::XEnt, f32)> = ents.iter().filter(|(e, _)| e.arena == ar).cloned().collect(); if grp.is_empty() { continue; }
                            let hx = cb.empty(c.ff_exp * grp.len()); cb.xrun_b(&xcr.tabs, &xcr.arena[ar], xcr.ar_bytes, &grp, &xb, d, c.ff_exp, tys, &hx, &y0); }
                        ygs[ti] = Some(y0);
                    }
                    if !combm() { cb.silu_mul(rd, rn, d + ne, d + ne + sf, &at(&hb_all, ti * sf), sf); }
                }
                if combm() { cb.silu_mul_m(&rd, t * rn, rn, d + ne, d + ne + sf, &hb_all, sf, t); }
                cb.gemv_m(&cl.sdown, &hb_all, t * sf, 0, sf, &sh_all, t * d, 0, d, t);
                for ti in 0..t { shs.push(at(&sh_all, ti * d)); } sh_b = true;
                if let Some(ps) = pslot {
                    // PIPE: gate + CPU-row copy + combine + next layer's dense, queued before the CPU rows exist
                    let yball = cb.empty(t * d); let g2 = pgate.clone();
                    if pipe2() { let (fl, er) = pipe_flag(cb); let tg = PIPE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1; *pgate.tgt.lock().unwrap() = Some((fl, tg));
                        unsafe { cb.client.pyre_wait_flag(fl, tg, env_usize("PYRE_QN_PIPE_SPIN_US").unwrap_or(100_000) as u64 * 1000, er); cb.client.pyre_host_gate(None, ps, t * d * 4, &yball); } drop(g2);
                    } else { unsafe { cb.client.pyre_host_gate(Some(Box::new(move || g2.wait())), ps, t * d * 4, &yball); } } GATE_OPEN.with(|g| g.set(true));
                    if combm() && t <= 4 { cb.comb3_m(xs, &ygs, &sh_all, &yball, &rd, t * rn, rn, d + ne + 2 * sf, &df, dfr, lrf, hc, d); } else {
                    for ti in 0..t { let yb = at(&yball, ti * d); cb.comb3(&xs[ti], &shs[ti], Some(&yb), ygs[ti].as_ref(), Some((&rds[ti], rn, d + ne + 2 * sf)), &dfs[ti], lrf, hc, d); } }
                    if let Some(nx) = next.as_mut() { nx(&mut *stp); }
                    cb.client.flush_queue();
                }
                // no blocking flush (drop-queue event wait) behind an open host gate: it would stall until the gate timed out
                let qf = std::time::Instant::now(); if kick() && pslot.is_none() { if kickq() { cb.client.flush_queue(); } else { let _ = cb.client.flush(); } } SD[2].fetch_add(qf.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            };
            if !lists.is_empty() {
                let work: Vec<(usize, &Vec<(usize, f32)>)> = lists.iter().map(|(e, v)| (*e, v)).collect();
                if xwarm() > 0 { if let Some(v) = XWARM_L.lock().unwrap().get_mut(il) { *v = lists.iter().map(|x| x.0).collect(); } } MISS[0].fetch_add(lists.len() as u64, std::sync::atomic::Ordering::Relaxed); MISS[1].fetch_add(lists.iter().map(|x| x.1.len() as u64).sum(), std::sync::atomic::Ordering::Relaxed);
                MISS[2].fetch_add(ons.iter().map(|o| o.len() as u64).sum(), std::sync::atomic::Ordering::Relaxed);
                let tr = std::time::Instant::now();
                // PYRE_QN_XPDIRECT (default on; =0 off): a gated layer's CPU rows are written by the pool straight into its pinned
                // slot (the GPU reads it only after the flag), not into a Vec copied there after the pool
                let pdir = if ex_ok && xpdirect() { xslot } else { None };
                if ex_ok && pdir.is_none() { let sn: usize = lists.iter().map(|x| x.1.len()).sum(); yr = vec![0f32; sn.max(1) * d]; }
                if ex_ok { EXACT_PARTS.with(|c| c.set(true)); }
                if let Some(ps) = pdir { let yp = unsafe { std::slice::from_raw_parts_mut(ps as *mut f32, sn_all * d) }; self.routed_pool_ord(l, &xin, &work, t, yp, &mut gpu, Some(&ord)); }
                else { self.routed_pool_ord(l, &xin, &work, t, &mut yr, &mut gpu, Some(&ord)); }
                EXACT_PARTS.with(|c| c.set(false));
                { let rns = tr.elapsed().as_nanos() as u64; if ex_ok && il < 128 && (xbal() || xlstat()) { XL_ST[il][1].fetch_add(rns, std::sync::atomic::Ordering::Relaxed); } if ex_ok { super::xchain::pool(rns); } ROUTED_NS.fetch_add(rns, std::sync::atomic::Ordering::Relaxed); crate::metrics::add_cpu_expert_ns(rns); }
            } else { gpu(); }
            if let Some(ps) = pslot { unsafe { std::ptr::copy_nonoverlapping(yr.as_ptr(), ps as *mut f32, t * d); } }
            drop(pg_open); GATE_OPEN.with(|g| g.set(false));
            if xg_tg.get() != 0 && !(ex_ok && xpdirect()) { if let Some(ps) = xslot { unsafe { std::ptr::copy_nonoverlapping(yr.as_ptr(), ps as *mut f32, sn_all * d); } } }
            XCK_TG.store(0, std::sync::atomic::Ordering::Relaxed);
            let xpiped = xg_tg.get() != 0; if xpiped { super::xchain::flag(); } drop(xg_guard);
            for k in &sins { xcg.map.remove(k); }
            drop(xcg);
            if xpiped { HY[3].fetch_add((std::time::Instant::now() - q5).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); return true; }
        } else {
            for ti in 0..t {
                let hb = cb.empty(sf); let sh = cb.empty(d);
                cb.silu_mul(&rds[ti], rn, d + ne, d + ne + sf, &hb, sf);
                cb.gemv(&cl.sdown, &hb, sf, 0, &sh, d, 0); shs.push(sh);
                self.routed(l, &rvs[ti][..d], &rvs[ti][d..d + ne], 1, &mut yr[ti * d..(ti + 1) * d], false);
            }
        }
        let q6 = std::time::Instant::now();
        if piped { HY[3].fetch_add(ns0(q5, q6), std::sync::atomic::Ordering::Relaxed); return true; }
        let yball = cb.buf(&yr);
        let sh_all = sh_g.unwrap_or(sh_all);
        if let Some(tab) = ex_tab.take() {
            let tb = cb.hup(<u32 as burn_cubecl::cubecl::CubeElement>::as_bytes(&tab)); let ysum = cb.empty(t * d);
            cb.rsum_ex(ex_y.as_ref().unwrap_or(&yball), &yball, &tb, &ysum, d, c.top_k, t);
            let none: Vec<Option<H>> = vec![None; t];
            cb.comb3_m(xs, &none, &sh_all, &ysum, &rd, t * rn, rn, d + ne + 2 * sf, &df, dfr, lrf, hc, d);
            let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
            HY[0].fetch_add(ns(q6, std::time::Instant::now()), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed); XW[15].fetch_add(ns(q6, std::time::Instant::now()), Relaxed);
            return false;
        }
        if sh_b && combm() && (t <= 4 || super::qn_cube::t8()) { cb.comb3_m(xs, &ygs, &sh_all, &yball, &rd, t * rn, rn, d + ne + 2 * sf, &df, dfr, lrf, hc, d); } else {
        for ti in 0..t {
            let yb = at(&yball, ti * d);
            cb.comb3(&xs[ti], &shs[ti], Some(&yb), ygs[ti].as_ref(), Some((&rds[ti], rn, d + ne + 2 * sf)), &dfs[ti], lrf, hc, d);
        }
        }
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64; use std::sync::atomic::Ordering::Relaxed;
        HY[0].fetch_add(ns(q6, std::time::Instant::now()), Relaxed); HY[3].fetch_add(ns(q5, q6), Relaxed);
        false
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
/// PYRE_QN_XWIN2: per-window routed_pool_ord wall ns / pairs / calls
static XW2: [std::sync::atomic::AtomicU64; 3] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
fn fused() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_FUSE").map(|v| v == "1").unwrap_or(false)) }
static BUSY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BUSYN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TL: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static SIDE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SD: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static MISS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
static HEAD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WV: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
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
/// PYRE_QN_XWARM=1 (opt-in): while the caller waits for a verify layer's routing readback, the pool workers prefetch
/// (into L2/L3) the weight bytes of the experts this layer ran on the CPU in the previous window, so the CPU misses
/// that repeat start from cache. Prefetch only: no arithmetic changes (bit-identical). =2: gate/up only.
fn xwarm() -> u8 { static P: std::sync::OnceLock<u8> = std::sync::OnceLock::new(); *P.get_or_init(|| env_usize("PYRE_QN_XWARM").unwrap_or(0) as u8) }
static XWARM_L: std::sync::Mutex<Vec<Vec<usize>>> = std::sync::Mutex::new(Vec::new());
impl Model {
    fn xwarm_read(&self, il: usize, l: &Layer, h: &burn_cubecl::cubecl::server::Handle) -> Vec<f32> {
        let cb = self.cube.as_ref().unwrap();
        let es: Vec<usize> = if xwarm() > 0 { let mut g = XWARM_L.lock().unwrap(); if g.len() < self.layers.len() { g.resize(self.layers.len(), vec![]); } g[il].clone() } else { vec![] };
        if es.is_empty() { return cb.read(h); }
        let (ff, d) = (self.cfg.ff_exp, self.cfg.d);
        let span = |m: &super::gguf_q::QMat, r0: usize, n: usize| -> (usize, usize) { let a = m.row_raw(r0).as_ptr() as usize; (a, m.row_raw(r0 + n - 1).as_ptr_range().end as usize - a) };
        let mut rg: Vec<(usize, usize)> = vec![];
        for &e in &es { rg.push(span(&l.gate_e, e * ff, ff)); rg.push(span(&l.up_e, e * ff, ff)); if xwarm() == 1 { rg.push(span(&l.down_e, e * d, d)); } }
        const CH: usize = 32768; let mut ch: Vec<(usize, usize)> = vec![];
        for (a, n) in rg { let mut o = 0; while o < n { ch.push((a + o, (n - o).min(CH))); o += CH; } }
        let mut out = vec![];
        qn_pool().run_side(ch.len(), &|k| { let (a, n) = ch[k]; #[cfg(target_arch = "x86_64")] unsafe { use std::arch::x86_64::*; let mut o = 0; while o < n { _mm_prefetch::<_MM_HINT_T1>((a + o) as *const i8); o += 64; } } }, &mut || { out = cb.read(h); });
        out
    }
}
fn prewake() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_PREWAKE").map(|v| v != "0").unwrap_or(true)) }
fn qn_pool() -> &'static super::spin_pool::Pool {
    static P: std::sync::OnceLock<super::spin_pool::Pool> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        // default: one worker per fast physical core, pinned (PYRE_QN_PIN="0,2,..." overrides, "none" disables)
        let auto = super::spin_pool::fast_cores();
        let mut pin: Vec<usize> = match env_var("PYRE_QN_PIN").ok().as_deref() { Some("none") => vec![], Some(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(), None => auto.clone() };
        // PYRE_QN_LSIB=1: the calling (launcher) thread moves to the SMT sibling of the first fast core and an extra
        // pinned worker takes that core, so GPU launches in run_side's side closure (PIPE) no longer cost a P-core of expert work
        let lsib = env_var("PYRE_QN_LSIB").map(|v| v == "1").unwrap_or(env_usize("PYRE_QN_PIPE").unwrap_or(0) >= 1) && pin.len() >= 2 && pin[1] != pin[0] + 1;
        let auto_pin = pin == auto;
        if lsib { let p0 = pin[0]; super::spin_pool::pin_to(p0 + 1); pin.insert(0, p0 + 1); eprintln!("expert pool: launcher pinned to sibling cpu {}", p0 + 1); }
        let n = env_usize("PYRE_QN_SPIN_THREADS").unwrap_or(if pin.len() >= 4 { pin.len() } else { rayon::current_num_threads() });
        eprintln!("expert pool: {n} workers, pinned to {:?}", pin);
        { // PYRE_QN_ETAIL: a job's last N tasks go to P-core workers only (E-cores ~2.3x slower would finish last)
            let nf = if auto_pin { super::spin_pool::n_fast() + lsib as usize } else { usize::MAX }; let et = env_usize("PYRE_QN_ETAIL").unwrap_or(0);
            eprintln!("expert pool: {} fast workers, E-core tail reserve {et}", nf.min(n));
            super::spin_pool::Pool::new_hybrid(n, std::time::Duration::from_micros(env_usize("PYRE_QN_SPIN_US").unwrap_or(if super::spin_pool::xexact_env() { 800 } else { 100 }) as u64), &pin, nf, et) }
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

/// PYRE_QN_DMULTI (default on): verify down projection decodes a Q2_0 row once for all tokens sharing the expert
fn dmulti() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| env_var("PYRE_QN_DMULTI").map(|v| v != "0").unwrap_or(true)) }

static VGT: [[std::sync::atomic::AtomicU64; 2]; 9] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 9];

/// PYRE_QN_PBENCH: real expert pool wake round-trip + serial per-layer quantize cost (CPU only).
pub fn pool_bench() {
    use super::qdot; let pool = qn_pool(); let n = pool.threads;
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[v.len() / 2], v[v.len() * 9 / 10]) };
    let noop = |_k: usize| { std::hint::black_box(0); };
    for _ in 0..200 { pool.run(n, &noop); }
    for (name, gap_us, pw) in [("hot", 0u64, false), ("gap500 prewake", 500, true), ("gap500 parked", 500, false), ("gap2000 parked", 2000, false)] {
        for ntask in [n, n * 16, 1280] {
            let mut v = vec![]; for _ in 0..400 {
                if gap_us > 0 { let t = std::time::Instant::now(); while t.elapsed().as_micros() < (gap_us as u128) - if pw { 50 } else { 0 } { std::hint::spin_loop(); } if pw { pool.prewake(); let t = std::time::Instant::now(); while t.elapsed().as_micros() < 50 { std::hint::spin_loop(); } } }
                let t = std::time::Instant::now(); pool.run(ntask, &noop); v.push(t.elapsed().as_secs_f64() * 1e6); }
            let (m, p9) = med(&mut v); eprintln!("pbench wake {name} tasks={ntask}: median {m:.1} us p90 {p9:.1} us");
        } }
    let (d, ff) = (2560usize, 512usize); let x: Vec<f32> = (0..4 * d).map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 300.0).collect();
    for (t, pairs) in [(1usize, 9usize), (2, 15), (4, 28)] {
        let mut v = vec![]; for _ in 0..300 { let z = std::time::Instant::now();
            let acts: Vec<qdot::Q8Act> = (0..t).map(|ti| qdot::quantize(&x[ti * d..(ti + 1) * d])).collect();
            let actk: Vec<qdot::Q8K> = (0..t).map(|ti| { let mut a = qdot::Q8K::default(); qdot::quantize_k_into(&x[ti * d..(ti + 1) * d], &mut a); a }).collect();
            let z1 = z.elapsed();
            let hd: Vec<qdot::Q8D> = (0..pairs).map(|p| { let mut a = qdot::Q8D::default(); qdot::quantize_d_into(&x[p * 64 % (3 * d)..p * 64 % (3 * d) + ff], &mut a); a }).collect();
            std::hint::black_box((&acts, &actk, &hd)); v.push((z1.as_secs_f64() * 1e6, z.elapsed().as_secs_f64() * 1e6)); }
        let mut a: Vec<f64> = v.iter().map(|p| p.0).collect(); let mut b: Vec<f64> = v.iter().map(|p| p.1).collect();
        eprintln!("pbench quant t={t} pairs={pairs}: in-acts {:.1} us, total serial {:.1} us (median)", med(&mut a).0, med(&mut b).0); }
    // PYRE_QN_PBENCH expert: one expert's gate/up (ff x d) + down (d x ff, Q2_0) at T=1..3 tokens, synthetic weights;
    // hot = the same expert repeatedly (cache-resident), cold = 64 distinct experts (~100 MB, DRAM-streamed);
    // par = 24-way pool over distinct experts (aggregate). Reports us per expert and the implied weight GB/s.
    use super::gguf_q::QType;
    for gty in [QType::IQ2XXS, QType::IQ2S] {
        let (gb, db) = (gty.row_bytes(d), QType::Q2_0.row_bytes(ff)); let ne = 64usize; let eb = 2 * ff * gb + d * db;
        let mut w = vec![0u8; ne * eb]; let mut st = 0x9E3779B97F4A7C15u64; for c in w.chunks_mut(8) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; let bb = st.to_le_bytes(); let n = c.len(); c.copy_from_slice(&bb[..n]); }
        // keep f16 scales finite/small (clear high bits of every odd byte; timing only)
        for i in (1..w.len()).step_by(2) { w[i] &= 0x3B; }
        let actk: Vec<qdot::Q8K> = (0..3).map(|ti| { let mut a = qdot::Q8K::default(); qdot::quantize_k_into(&x[ti * d..(ti + 1) * d], &mut a); a }).collect();
        let hd: Vec<qdot::Q8D> = (0..3).map(|p| { let mut a = qdot::Q8D::default(); qdot::quantize_d_into(&x[p * ff..(p + 1) * ff], &mut a); a }).collect();
        let one = |e: usize, t: usize| -> f32 { let base = &w[e * eb..(e + 1) * eb]; let (gw, rest) = base.split_at(ff * gb); let (uw, dw) = rest.split_at(ff * gb);
            let al: Vec<&qdot::Q8K> = actk[..t].iter().collect(); let dl: Vec<&qdot::Q8D> = hd[..t].iter().collect(); let (mut gv, mut uv, mut ov) = ([0f32; 16], [0f32; 16], [0f32; 8]); let mut acc = 0f32;
            for r in 0..ff { let (gr, ur) = (&gw[r * gb..(r + 1) * gb], &uw[r * gb..(r + 1) * gb]);
                if t == 1 { gv[0] = qdot::row_dot_k(gty, gr, &actk[0]); uv[0] = qdot::row_dot_k(gty, ur, &actk[0]); } else { qdot::row_dot_k_multi(gty, gr, &al, &mut gv); qdot::row_dot_k_multi(gty, ur, &al, &mut uv); }
                acc += gv[0] * uv[0]; }
            for r in 0..d { let dr = &dw[r * db..(r + 1) * db]; if t == 1 { ov[0] = qdot::q2_0_dot_d(dr, &hd[0]); } else { qdot::q2_0_dot_d_multi(dr, &dl, &mut ov); } acc += ov[0]; }
            acc };
        for t in 1..=3usize {
            let mut vh = vec![]; for _ in 0..200 { let z = std::time::Instant::now(); std::hint::black_box(one(0, t)); vh.push(z.elapsed().as_secs_f64() * 1e6); }
            let mut vc = vec![]; for i in 0..400 { let z = std::time::Instant::now(); std::hint::black_box(one(1 + (i * 37) % (ne - 1), t)); vc.push(z.elapsed().as_secs_f64() * 1e6); }
            let mut vp = vec![]; for _ in 0..50 { let z = std::time::Instant::now(); pool.run(ne, &|e| { std::hint::black_box(one(e, t)); }); vp.push(z.elapsed().as_secs_f64() * 1e6 / ne as f64); }
            let (h, c, pp) = (med(&mut vh).0, med(&mut vc).0, med(&mut vp).0); let mb = eb as f64 / 1e6;
            eprintln!("pbench expert {gty:?}+Q2_0 {mb:.2} MB t={t}: 1-thread hot {h:.0} us ({:.1} GB/s), cold {c:.0} us ({:.1} GB/s); pool {n}x distinct {pp:.1} us/expert ({:.1} GB/s aggregate)", mb / h * 1e3, mb / c * 1e3, mb / pp * 1e3);
        } }
}
