//! The model's own multi-token-prediction (MTP) layer as a speculative drafter, on the CPU.
//!
//! Design follows Strata (github.com/Niko1221/Strata, MIT; src/core/mtp.cpp, include/strata/core/mtp.hpp):
//! cell i of the drafter's sequence pairs the main model's final multi-stream residual R_i with the token at
//! i+1, at rope position i; its output predicts token i+2 and its own residual feeds the next draft step.
//!
//!   e = fc_embedding(rms(embed(tok)) * (1+w))      h = fc_hidden per stream(rms_{hc*d}(R) * (1+w))
//!   R = h + e -> attn hyper-connection -> attention (own K/V, dense) -> MLP hyper-connection ->
//!   MoE (512 experts, top-k, gated shared expert) -> final mixer -> the main model's head (draft-vocab subset)
//!
//! Like Strata, only DRAFT quality depends on this file: every emitted token comes from the verify pass, so
//! greedy output stays identical to plain decode. Simplifications (all draft-only): int8 (Q8_0) weights built
//! from the BF16 checkpoint at load (experts lazily, on first use), dense attention over a sliding window of
//! cells (no QSA indexer), the head restricted to a token subset (Strata's data/draft_vocab_*.bin, MIT).
//! The tensors come from `scripts/mtp_fetch.py` (Strata's tools/mtp_fetch.py approach): one safetensors file.
//! This is pyre's own Rust implementation; no Strata code is linked.
use super::qdot::Q8Mat;
use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

struct St { mm: Mmap, base: usize, t: HashMap<String, (Vec<usize>, usize, usize)> }
impl St {
    fn open(p: &std::path::Path) -> Result<Self> {
        let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mm = unsafe { Mmap::map(&f)? };
        let n = u64::from_le_bytes(mm[..8].try_into().unwrap()) as usize;
        let h: serde_json::Value = serde_json::from_slice(&mm[8..8 + n])?;
        let mut t = HashMap::new();
        for (k, v) in h.as_object().context("header")? {
            if k == "__metadata__" { continue; }
            anyhow::ensure!(v["dtype"] == "BF16", "{k}: dtype {}", v["dtype"]);
            let sh: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let o = v["data_offsets"].as_array().unwrap();
            t.insert(k.trim_start_matches("mtp.").trim_start_matches("layers.0.").to_string(), (sh, o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize));
        }
        Ok(St { mm, base: 8 + n, t })
    }
    fn raw(&self, n: &str) -> Result<(&[usize], &[u8])> { let (s, a, b) = self.t.get(n).with_context(|| format!("mtp tensor {n}"))?; Ok((s, &self.mm[self.base + a..self.base + b])) }
    fn row(b: &[u8], r: usize, cols: usize, y: &mut [f32]) { let o = r * cols * 2; for i in 0..cols { y[i] = f32::from_bits((u16::from_le_bytes([b[o + 2 * i], b[o + 2 * i + 1]]) as u32) << 16); } }
    /// a 1-D norm weight, Gemma-style: 1 + w
    fn norm(&self, n: &str) -> Result<Vec<f32>> { let (s, b) = self.raw(n)?; let mut y = vec![0f32; s[0]]; Self::row(b, 0, s[0], &mut y); for v in y.iter_mut() { *v += 1.0; } Ok(y) }
    fn vec(&self, n: &str) -> Result<Vec<f32>> { let (s, b) = self.raw(n)?; let c = s.iter().product(); let mut y = vec![0f32; c]; Self::row(b, 0, c, &mut y); Ok(y) }
    fn q8(&self, n: &str) -> Result<Q8Mat> { self.q8s(&[n]) }
    /// several matrices with the same input, stacked by rows (one pass over the activation)
    fn q8s(&self, ns: &[&str]) -> Result<Q8Mat> {
        let ms: Vec<(&[usize], &[u8])> = ns.iter().map(|n| self.raw(n)).collect::<Result<_>>()?; let c = ms[0].0[1];
        let idx: Vec<(usize, usize)> = ms.iter().enumerate().flat_map(|(k, m)| (0..m.0[0]).map(move |r| (k, r))).collect();
        Ok(Q8Mat::from_fn(idx.len(), c, |i, y| { let (k, r) = idx[i]; Self::row(ms[k].1, r, c, y) }))
    }
}

/// PYRE_MTP_DEBUG timers (ns): input, attn-hc+kv, q+attention, o+combine, mlp-hc, shared, experts, mixer
pub static TM: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
fn tick(i: usize, t: &mut std::time::Instant) { let n = t.elapsed().as_nanos() as u64; TM[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed); *t = std::time::Instant::now(); }
struct Hc { norm: Vec<f32>, down: Q8Mat, up: Q8Mat, inject: Option<Q8Mat>, lr: usize }
fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn silu(x: f32) -> f32 { x * sigm(x) }
fn rms(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { y[i] = x[i] * s * w[i]; } }
/// A matmul hook for the calling thread (PYRE_MTP_GPU: the dense drafter matrices on the GPU); None = CPU.
type MmHook<'a> = &'a dyn Fn(&Q8Mat, &[f32], usize) -> Option<Vec<f32>>;
thread_local! { static HOOK: std::cell::Cell<Option<MmHook<'static>>> = const { std::cell::Cell::new(None) }; }
/// Run `f` with `h` deciding each drafter matmul on this thread (draft-only numerics; verify keeps the output).
pub fn with_mm_hook<R>(h: MmHook<'_>, f: impl FnOnce() -> R) -> R {
    // SAFETY: the hook is cleared before `h` goes out of scope (also on unwind, via the guard)
    let hs: MmHook<'static> = unsafe { std::mem::transmute(h) };
    struct G; impl Drop for G { fn drop(&mut self) { HOOK.with(|c| c.set(None)); } }
    HOOK.with(|c| c.set(Some(hs))); let _g = G; f()
}
fn mm(m: &Q8Mat, x: &[f32], t: usize) -> Vec<f32> {
    if let Some(h) = HOOK.with(|c| c.get()) { if let Some(y) = h(m, x, t) { return y; } }
    let mut y = vec![0f32; t * m.rows]; m.matmul(x, t, &mut y); y
}

/// The drafter's K/V: cells c0..c0+n (rope position = cell), [cell][n_kv*hd].
#[derive(Default)]
pub struct Kv { pub c0: usize, pub n: usize, k: Vec<f32>, v: Vec<f32>, pub gid: u64 }
impl Kv {
    pub fn end(&self) -> usize { self.c0 + self.n }
    pub fn truncate(&mut self, end: usize, row: usize) { if end < self.end() { let n = end.saturating_sub(self.c0); self.n = n; self.k.truncate(n * row); self.v.truncate(n * row); } }
}

pub struct Mtp {
    d: usize, hc: usize, nh: usize, nkv: usize, hd: usize, n_rot: usize, eps: f32, top_k: usize, ff: usize, rope_inv: Vec<f32>,
    pre_e: Vec<f32>, pre_h: Vec<f32>, fc_e: Q8Mat, fc_h: Q8Mat, hca: Hc, hcf: Hc, mix: Hc,
    q: Q8Mat, k: Q8Mat, v: Q8Mat, o: Q8Mat, qn: Vec<f32>, kn: Vec<f32>,
    rsu: Q8Mat, sd: Q8Mat, sgate: Vec<f32>, ne: usize,
    st: St, ex: Vec<OnceLock<(Q8Mat, Q8Mat)>>,
    head: Q8Mat, ids: Vec<u32>, pub window: usize,
    gw: OnceLock<Vec<burn_cubecl::cubecl::server::Handle>>,
}

pub struct Geo { pub d: usize, pub hc: usize, pub nh: usize, pub nkv: usize, pub hd: usize, pub n_rot: usize, pub eps: f32, pub top_k: usize, pub rope_base: f32 }

impl Mtp {
    /// `head_row(token, out)`: a row of the main model's output matrix; `ids`: the draft vocabulary.
    pub fn load(p: &std::path::Path, g: &Geo, ids: Vec<u32>, head_row: impl Fn(usize, &mut [f32]) + Sync) -> Result<Self> {
        let st = St::open(p)?;
        let hcm = |p: &str, inj: bool| -> Result<Hc> { let (dn, ij) = (format!("{p}.input_mix_weight_down.weight"), format!("{p}.block_inject_weight.weight"));
            Ok(Hc { norm: st.norm(&format!("{p}.hc_norm.weight"))?, lr: st.raw(&dn)?.0[0], down: if inj { st.q8s(&[&dn, &ij])? } else { st.q8(&dn)? }, up: st.q8(&format!("{p}.input_mix_weight_up.weight"))?, inject: None }) };
        let (ne, ff) = { let (s, _) = st.raw("mlp.experts.down_proj")?; (s[0], s[2]) };
        let half = g.n_rot / 2;
        let head = Q8Mat::from_fn(ids.len(), g.d, |r, y| head_row(ids[r] as usize, y));
        Ok(Mtp { d: g.d, hc: g.hc, nh: g.nh, nkv: g.nkv, hd: g.hd, n_rot: g.n_rot, eps: g.eps, top_k: std::env::var("PYRE_MTP_TOPK").ok().and_then(|v| v.parse().ok()).filter(|&k: &usize| k >= 1).map(|k| k.min(g.top_k)).unwrap_or(g.top_k), ff,
            rope_inv: (0..half).map(|i| g.rope_base.powf(-(2.0 * i as f32) / g.n_rot as f32)).collect(),
            pre_e: st.norm("pre_fc_norm_embedding.weight")?, pre_h: st.norm("pre_fc_norm_hidden.weight")?, fc_e: st.q8("fc_embedding.weight")?, fc_h: st.q8("fc_hidden.weight")?,
            hca: hcm("attn_hyper_connection", true)?, hcf: hcm("mlp_hyper_connection", true)?, mix: hcm("hyper_connection_mixer", false)?,
            q: st.q8("self_attn.q_proj.weight")?, k: st.q8("self_attn.k_proj.weight")?, v: st.q8("self_attn.v_proj.weight")?, o: st.q8("self_attn.o_proj.weight")?,
            qn: st.norm("self_attn.q_norm.weight")?, kn: st.norm("self_attn.k_norm.weight")?,
            rsu: st.q8s(&["mlp.gate.weight", "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight"])?, ne, sd: st.q8("mlp.shared_expert.down_proj.weight")?,
            sgate: st.vec("mlp.shared_expert_gate.weight")?, ex: (0..ne).map(|_| OnceLock::new()).collect(), st, head, ids, gw: OnceLock::new(),
            window: std::env::var("PYRE_MTP_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(128) /* 128: C tps 30.6->33.8 (less prompt catch-up), acceptance same; 64 loses acceptance */ })
    }
    pub fn row(&self) -> usize { self.nkv * self.hd }
    pub fn rdim(&self) -> usize { self.hc * self.d }

    fn expert(&self, e: usize) -> &(Q8Mat, Q8Mat) {
        self.ex[e].get_or_init(|| {
            let (d, ff) = (self.d, self.ff);
            let (_, gu) = self.st.raw("mlp.experts.gate_up_proj").unwrap(); let (_, dn) = self.st.raw("mlp.experts.down_proj").unwrap();
            let gu = &gu[e * 2 * ff * d * 2..(e + 1) * 2 * ff * d * 2]; let dn = &dn[e * d * ff * 2..(e + 1) * d * ff * 2];
            (Q8Mat::from_fn(2 * ff, d, |r, y| St::row(gu, r, d, y)), Q8Mat::from_fn(d, ff, |r, y| St::row(dn, r, ff, y)))
        })
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.n_rot / 2;
        for i in 0..half { let a = pos as f32 * self.rope_inv[i]; let (s, co) = a.sin_cos(); let (x0, x1) = (v[i], v[i + half]); v[i] = x0 * co - x1 * s; v[i + half] = x1 * co + x0 * s; }
    }

    /// Grouped RMSNorm + low-rank sigmoid gate + stream mean, as the main model's hyper-connections.
    fn hc_mix(&self, h: &Hc, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let (d, hc) = (self.d, self.hc); let w = hc * d;
        let mut xn = vec![0f32; t * w];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let s = i % hc; rms(&x[i * d..(i + 1) * d], &h.norm[s * d..(s + 1) * d], self.eps, o); });
        let o = mm(&h.down, &xn, t); let nr = h.down.rows; let lr = h.lr; let _ = &h.inject;
        let mut lo = vec![0f32; t * lr]; let mut inj = vec![0f32; t * hc];
        for ti in 0..t { for i in 0..lr { lo[ti * lr + i] = silu(o[ti * nr + i] / hc as f32); } if nr > lr { for s in 0..hc { inj[ti * hc + s] = o[ti * nr + lr + s] / hc as f32; } } }
        let gate = mm(&h.up, &lo, t);
        let mut mixed = vec![0f32; t * d];
        for ti in 0..t { for s in 0..hc { for i in 0..d { let k = ti * w + s * d + i; mixed[ti * d + i] += xn[k] * sigm(gate[k]); } } for i in 0..d { mixed[ti * d + i] /= hc as f32; } }
        (mixed, inj)
    }
    fn hc_combine(&self, x: &mut [f32], out: &[f32], inj: &[f32], t: usize) {
        let (d, hc) = (self.d, self.hc);
        for ti in 0..t { for s in 0..hc { let w = 2.0 * sigm(inj[ti * hc + s]); let r = &mut x[(ti * hc + s) * d..(ti * hc + s + 1) * d]; for i in 0..d { r[i] += out[ti * d + i] * w; } } }
    }

    /// The input branches: t rows of main-model residuals `r` [t][hc*d] and token embeddings `emb` [t][d].
    fn input(&self, r: &[f32], emb: &[f32], t: usize) -> Vec<f32> {
        let (d, hc) = (self.d, self.hc); let w = hc * d;
        let mut en = vec![0f32; t * d]; for ti in 0..t { rms(&emb[ti * d..(ti + 1) * d], &self.pre_e, self.eps, &mut en[ti * d..(ti + 1) * d]); }
        let e2 = mm(&self.fc_e, &en, t);
        let mut hn = vec![0f32; t * w]; for ti in 0..t { rms(&r[ti * w..(ti + 1) * w], &self.pre_h, self.eps, &mut hn[ti * w..(ti + 1) * w]); }
        let mut h2 = mm(&self.fc_h, &hn, t * hc);
        for ti in 0..t { for s in 0..hc { for i in 0..d { h2[(ti * hc + s) * d + i] += e2[ti * d + i]; } } }
        h2
    }

    /// K/V of t cells starting at `pos0`, appended to `kv`; returns (R after input, mixed, inject) for reuse.
    fn kv_append(&self, kv: &mut Kv, r: &[f32], emb: &[f32], t: usize, pos0: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let row = self.row();
        if kv.end() != pos0 { *kv = Kv { c0: pos0, ..Default::default() }; }
        let mut tt = std::time::Instant::now();
        let rr = self.input(r, emb, t); tick(0, &mut tt);
        let (m, inj) = self.hc_mix(&self.hca, &rr, t);
        let mut kf = mm(&self.k, &m, t); let vf = mm(&self.v, &m, t);
        for ti in 0..t { for h in 0..self.nkv { let kh = &mut kf[ti * row + h * self.hd..ti * row + (h + 1) * self.hd]; let mut y = vec![0f32; self.hd]; rms(kh, &self.kn, self.eps, &mut y); kh.copy_from_slice(&y); self.rope(kh, pos0 + ti); } }
        kv.k.extend_from_slice(&kf); kv.v.extend_from_slice(&vf); kv.n += t; tick(1, &mut tt);
        if kv.n > self.window { let drop = kv.n - self.window; kv.k.drain(..drop * row); kv.v.drain(..drop * row); kv.c0 += drop; kv.n -= drop; }
        (rr, m, inj)
    }

    /// Catch the K/V up over cells pos0..pos0+t (no draft output needed): batched.
    pub fn catch_up(&self, kv: &mut Kv, r: &[f32], emb: &[f32], t: usize, pos0: usize) {
        for c in (0..t).step_by(64) { let n = (t - c).min(64); let w = self.rdim(); self.kv_append(kv, &r[c * w..(c + n) * w], &emb[c * self.d..(c + n) * self.d], n, pos0 + c); }
    }

    /// Build every routed expert's int8 copy now (else on first use, which stalls drafting).
    pub fn warm(&self) { (0..self.ex.len()).into_par_iter().for_each(|e| { self.expert(e); }); }
    /// The draft head on the CPU over the draft vocabulary: (token, probability).
    pub fn head_q8(&self) -> &Q8Mat { &self.head }
    /// The dense (non-expert) matrices, for a GPU copy
    pub fn dense_mats(&self) -> Vec<&Q8Mat> { let mut v = vec![&self.fc_e, &self.fc_h, &self.q, &self.k, &self.v, &self.o, &self.rsu, &self.sd]; for h in [&self.hca, &self.hcf, &self.mix] { v.push(&h.down); v.push(&h.up); if let Some(i) = &h.inject { v.push(i); } } v }
    pub fn head_ids(&self) -> &[u32] { &self.ids }
    pub fn head_cpu(&self, s: &[f32]) -> (u32, f32) { let lg = mm(&self.head, s, 1); let (i, p) = best(&lg); (self.ids[i], p) }
    /// One full draft step at cell `pos`: returns (the final mixer's output for the head, the drafter's residual).
    pub fn step(&self, kv: &mut Kv, r: &[f32], emb: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let (nh, hd, d) = (self.nh, self.hd, self.d); let row = self.row(); let grp = nh / self.nkv;
        let (mut rr, m, inj) = self.kv_append(kv, r, emb, 1, pos);
        let mut tt = std::time::Instant::now();
        let qf = mm(&self.q, &m, 1);
        let (kc, vc, n) = (&kv.k, &kv.v, kv.n); let scale = (hd as f32).sqrt().recip();
        let mut att = vec![0f32; nh * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(h, out)| {
            let base = h * 2 * hd; let mut qh = vec![0f32; hd]; rms(&qf[base..base + hd], &self.qn, self.eps, &mut qh); self.rope(&mut qh, pos);
            let kvh = h / grp;
            let mut sc: Vec<f32> = (0..n).map(|p| { let k = &kc[p * row + kvh * hd..p * row + (kvh + 1) * hd]; qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale }).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            for (p, s) in sc.iter().enumerate() { let w = s / sum; let vv = &vc[p * row + kvh * hd..p * row + (kvh + 1) * hd]; for (o, v) in out.iter_mut().zip(vv) { *o += w * v; } }
            let gate = &qf[base + hd..base + 2 * hd]; for i in 0..hd { out[i] *= sigm(gate[i]); }
        });
        tick(2, &mut tt);
        let bo = mm(&self.o, &att, 1);
        self.hc_combine(&mut rr, &bo, &inj, 1);
        tick(3, &mut tt);
        let (m2, inj2) = self.hc_mix(&self.hcf, &rr, 1); tick(4, &mut tt);
        // MoE: router top-k (softmax over the picked), gated shared expert
        let rsu = mm(&self.rsu, &m2, 1); let lg = &rsu[..self.ne];
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.select_nth_unstable_by(self.top_k, |a, b| lg[*b].partial_cmp(&lg[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let top = &idx[..self.top_k]; let mx = top.iter().map(|&e| lg[e]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = top.iter().map(|&e| (lg[e] - mx).exp()).collect(); let wsum: f32 = ws.iter().sum();
        let sf = (rsu.len() - self.ne) / 2; let mut gs: Vec<f32> = (0..sf).map(|i| silu(rsu[self.ne + i]) * rsu[self.ne + sf + i]).collect();
        let mut y = mm(&self.sd, &gs, 1);
        let sgv = sigm(self.sgate.iter().zip(&m2).map(|(a, b)| a * b).sum::<f32>()); for v in y.iter_mut() { *v *= sgv; }
        let ff = self.ff; tick(5, &mut tt);
        let parts: Vec<Vec<f32>> = top.par_iter().zip(ws.par_iter()).map(|(&e, &w)| {
            let (gu, dn) = self.expert(e); let h = mm(gu, &m2, 1);
            let a: Vec<f32> = (0..ff).map(|i| silu(h[i]) * h[ff + i]).collect();
            let mut o = mm(dn, &a, 1); let w = w / wsum; for v in o.iter_mut() { *v *= w; } o
        }).collect();
        for p in parts { for i in 0..d { y[i] += p[i]; } } tick(6, &mut tt);
        self.hc_combine(&mut rr, &y, &inj2, 1);
        let (s, _) = self.hc_mix(&self.mix, &rr, 1);
        tick(7, &mut tt);
        (s, rr)
    }
}

/// PYRE_MTP_GPUSTEP: one draft step's GPU outputs (s = final mixer output for the head, rr = drafter residual, the new K/V cell)
pub struct GRing { pub k: H, pub v: H, pub cap: usize, pub end: usize, pub gid: u64, pub id: u64 }
/// PYRE_MTP_GRAPH: the persistent buffers and the two captured graphs of the GPU draft step (before / after the CPU routed experts)
pub struct GG { fx: Vec<H>, b1: Vec<H>, b2: Vec<H>, g1: Option<burn_cubecl::cubecl::client::Graph>, g2: Option<burn_cubecl::cubecl::client::Graph>, o1: Option<(H, H, H)>, key: (u64, usize, usize) }
pub static GG_S: std::sync::Mutex<Option<GG>> = std::sync::Mutex::new(None);
pub static GG_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static GG_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GG_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub struct GStep { pub s: H, pub rr: H, pub krow: Vec<f32>, pub vrow: Vec<f32> }
type H = burn_cubecl::cubecl::server::Handle;
type Cb = super::qn_cube::Cube;
type CM = super::qn_cube::CMat;
impl Mtp {
    fn gw(&self, cb: &Cb) -> &Vec<H> { self.gw.get_or_init(|| vec![cb.buf(&self.pre_e), cb.buf(&self.pre_h), cb.buf(&self.hca.norm), cb.buf(&self.hcf.norm), cb.buf(&self.mix.norm), cb.buf(&self.qn), cb.buf(&self.kn), cb.buf(&self.rope_inv)]) }
    /// the K/V cells on the GPU, with room for `extra` more; returns (k, v, cells)
    /// keep the GPU K/V ring (cell c at slot c % cap) in step with `kv`: only cells not uploaded yet are written
    pub fn kv_sync(&self, cb: &Cb, ring: &mut Option<GRing>, kv: &mut Kv) {
        static GID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1); if kv.gid == 0 { kv.gid = GID.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let row = self.row(); let cap = self.window + 32;
        if ring.as_ref().map(|g| g.cap != cap).unwrap_or(true) { static RID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1); *ring = Some(GRing { k: cb.empty(cap * row), v: cb.empty(cap * row), cap, end: 0, gid: 0, id: RID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }); }
        let g = ring.as_mut().unwrap(); let from = if g.gid == kv.gid && g.end >= kv.c0 && g.end <= kv.end() { g.end } else { kv.c0 };
        for c in from..kv.end() { let i = c - kv.c0; let o = ((c % cap) * row * 4) as u64;
            cb.client.write(&g.k.clone().offset_start(o), burn_cubecl::cubecl::bytes::Bytes::from_elems(kv.k[i * row..(i + 1) * row].to_vec()));
            cb.client.write(&g.v.clone().offset_start(o), burn_cubecl::cubecl::bytes::Bytes::from_elems(kv.v[i * row..(i + 1) * row].to_vec())); }
        g.end = kv.end(); g.gid = kv.gid;
    }
    /// compile the GPU step kernels now (one throwaway step on a 1-cell K/V)
    pub fn warm_gpu<'a>(&self, cb: &Cb, g: &(dyn Fn(&Q8Mat) -> Option<&'a CM> + Sync), head: &CM) { let mut ring = None; let row = self.row(); let mut kv = Kv { c0: 0, n: 1, k: vec![0.0; row], v: vec![0.0; row], gid: 0 }; self.kv_sync(cb, &mut ring, &mut kv); let g0 = ring.unwrap();
        let r = cb.buf(&vec![0.01f32; self.rdim()]); if self.step_gpu(cb, g, &g0.k, &g0.v, g0.cap, 0, &r, &vec![0.01f32; self.d], 1).is_none() { eprintln!("mtp: warm step returned None"); } if let Some(st) = self.step_gpu(cb, g, &g0.k, &g0.v, g0.cap, 0, &r, &vec![0.01f32; self.d], 1) { let o = cb.empty(head.rows); cb.gemv(head, &st.s, self.d, 0, &o, head.rows, 0); let _ = cb.best(&o, head.rows); }
        if std::env::var("PYRE_MTP_GRAPH").map(|v| v != "0").unwrap_or(true) { if let Some((lg, ..)) = self.step_gpu_g(cb, g, head, &g0.k, &g0.v, g0.cap, 0, &r, &vec![0.01f32; self.d], 1, g0.id) { let _ = cb.best(&lg, head.rows); } } }
    pub fn kv_reset(&self, kv: &mut Kv, c0: usize) { *kv = Kv { c0, ..Default::default() }; }
    /// one catch-up cell on the GPU: K/V of cell `pos` written to the ring and returned (k, v) for the CPU copy
    #[allow(clippy::too_many_arguments)]
    /// `rb`/`eb` = this cell's residual / embedding (offset handles into one upload); its K/V also go to `pk` at cell slot `j` (2*row each), read once by the caller
    pub fn kv_cell_gpu<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, kc: &H, vc: &H, cap: usize, rb: &H, eb: &H, pos: usize, pk: &H, pkn: usize, j: usize) -> Option<()> {
        use super::mtp_gpu as mg; let (d, hc, hd, eps) = (self.d, self.hc, self.hd, self.eps); let w = hc * d; let row = self.row(); let gw = self.gw(cb).clone();
        let eb = eb.clone(); let en = cb.empty(d); mg::rms_seg(cb, &eb, d, &gw[0], d, &en, d, d, 1, 1, eps);
        let e2 = cb.empty(d); cb.gemv(g(&self.fc_e)?, &en, d, 0, &e2, d, 0);
        let hn = cb.empty(w); mg::rms_seg(cb, rb, w, &gw[1], w, &hn, w, w, 1, 1, eps);
        let h2 = cb.empty(w); let fh = g(&self.fc_h)?; for s in 0..hc { cb.gemv(fh, &hn, w, s * d, &h2, w, s * d); }
        let rr = cb.empty(w); mg::add_b(cb, &h2, &e2, &rr, d, w);
        let (m, _) = self.hc_mix_g(cb, g, &self.hca, 2, &rr)?;
        let kf = cb.empty(row); cb.gemv(g(&self.k)?, &m, d, 0, &kf, row, 0); let vf = cb.empty(row); cb.gemv(g(&self.v)?, &m, d, 0, &vf, row, 0);
        let kn = cb.empty(row); mg::rms_seg(cb, &kf, row, &gw[6], hd, &kn, hd, hd, self.nkv, 1, eps); mg::rope_seg(cb, &kn, row, &gw[7], self.n_rot / 2, hd, self.nkv, pos);
        mg::cp(cb, &kn, row, kc, cap * row, 0, (pos % cap) * row, row); mg::cp(cb, &vf, row, vc, cap * row, 0, (pos % cap) * row, row);
        mg::cp(cb, &kn, row, pk, pkn, 0, 2 * j * row, row); mg::cp(cb, &vf, row, pk, pkn, 0, (2 * j + 1) * row, row); Some(())
    }
    pub fn kv_upload(&self, cb: &Cb, kv: &Kv, extra: usize) -> (H, H, usize) { let row = self.row(); let mut k = kv.k.clone(); k.resize((kv.n + extra) * row, 0.0); let mut v = kv.v.clone(); v.resize((kv.n + extra) * row, 0.0); (cb.buf(&k), cb.buf(&v), kv.n) }
    pub fn kv_push(&self, kv: &mut Kv, k: &[f32], v: &[f32]) { let row = self.row(); kv.k.extend_from_slice(k); kv.v.extend_from_slice(v); kv.n += 1;
        if kv.n > self.window { let drop = kv.n - self.window; kv.k.drain(..drop * row); kv.v.drain(..drop * row); kv.c0 += drop; kv.n -= drop; } }
    fn hc_mix_g<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, h: &Hc, ni: usize, x: &H) -> Option<(H, H)> { self.hc_mix_a(cb, g, h, ni, x, &mut |n| cb.empty(n)) }
    fn hc_mix_a<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, h: &Hc, ni: usize, x: &H, al: &mut dyn FnMut(usize) -> H) -> Option<(H, H)> {
        use super::mtp_gpu as mg; let (d, hc) = (self.d, self.hc); let w = hc * d; let gw = self.gw(cb);
        let xn = al(w); mg::rms_seg(cb, x, w, &gw[ni], w, &xn, d, d, hc, hc, self.eps);
        let nr = h.down.rows; let o = al(nr); cb.gemv(g(&h.down)?, &xn, w, 0, &o, nr, 0);
        let lo = al(h.lr); let inj = al(hc); mg::hc_pre(cb, &o, nr, &lo, &inj, h.lr, hc, nr > h.lr);
        let gate = al(w); cb.gemv(g(&h.up)?, &lo, h.lr, 0, &gate, w, 0);
        let mixed = al(d); mg::hc_mixed(cb, &xn, &gate, &mixed, d, hc); Some((mixed, inj))
    }
    /// routed experts of one token on the CPU (as `step`), given the MLP mix output and the router logits
    fn routed(&self, m2: &[f32], lg: &[f32]) -> Vec<f32> {
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.select_nth_unstable_by(self.top_k, |a, b| lg[*b].partial_cmp(&lg[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let top = &idx[..self.top_k]; let mx = top.iter().map(|&e| lg[e]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = top.iter().map(|&e| (lg[e] - mx).exp()).collect(); let wsum: f32 = ws.iter().sum(); let (d, ff) = (self.d, self.ff);
        let parts: Vec<Vec<f32>> = top.par_iter().zip(ws.par_iter()).map(|(&e, &w)| { let (gu, dn) = self.expert(e); let h = mm(gu, m2, 1);
            let a: Vec<f32> = (0..ff).map(|i| silu(h[i]) * h[ff + i]).collect(); let mut o = mm(dn, &a, 1); let w = w / wsum; for v in o.iter_mut() { *v *= w; } o }).collect();
        let mut y = vec![0f32; d]; for p in parts { for i in 0..d { y[i] += p[i]; } } y
    }
    /// `step` on the GPU (draft-only numerics): K/V cell `ncell` of the GPU cache (capacity `cap` cells) is written,
    /// attention runs over the last `window` cells, routed experts run on the CPU after one packed readback.
    #[allow(clippy::too_many_arguments)]
    pub fn step_gpu<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, kc: &H, vc: &H, cap: usize, kc0: usize, r: &H, emb: &[f32], pos: usize) -> Option<GStep> {
        use super::mtp_gpu as mg; let (d, hc, nh, hd, eps) = (self.d, self.hc, self.nh, self.hd, self.eps); let w = hc * d; let row = self.row(); let gw = self.gw(cb).clone();
        let mut tt = std::time::Instant::now(); let dbg = std::env::var("PYRE_MTP_TSTEP").is_ok(); let mut tq = std::time::Instant::now();
        let mut sy = |tag: &str, h: &H| { if dbg { let _ = cb.read(&h.clone()); eprintln!("tstep {tag} {:.2} ms", tq.elapsed().as_secs_f64() * 1e3); tq = std::time::Instant::now(); } };
        let eb = cb.buf(emb); let en = cb.empty(d); mg::rms_seg(cb, &eb, d, &gw[0], d, &en, d, d, 1, 1, eps);
        let e2 = cb.empty(d); cb.gemv(g(&self.fc_e)?, &en, d, 0, &e2, d, 0);
        let hn = cb.empty(w); mg::rms_seg(cb, r, w, &gw[1], w, &hn, w, w, 1, 1, eps);
        let h2 = cb.empty(w); let fh = g(&self.fc_h)?; for s in 0..hc { cb.gemv(fh, &hn, w, s * d, &h2, w, s * d); }
        let rr = cb.empty(w); mg::add_b(cb, &h2, &e2, &rr, d, w); sy("input", &rr);
        let (m, inj) = self.hc_mix_g(cb, g, &self.hca, 2, &rr)?; sy("hca", &m);
        let kf = cb.empty(row); cb.gemv(g(&self.k)?, &m, d, 0, &kf, row, 0); let vf = cb.empty(row); cb.gemv(g(&self.v)?, &m, d, 0, &vf, row, 0);
        let kn = cb.empty(row); mg::rms_seg(cb, &kf, row, &gw[6], hd, &kn, hd, hd, self.nkv, 1, eps); mg::rope_seg(cb, &kn, row, &gw[7], self.n_rot / 2, hd, self.nkv, pos);
        mg::cp(cb, &kn, row, kc, cap * row, 0, (pos % cap) * row, row); mg::cp(cb, &vf, row, vc, cap * row, 0, (pos % cap) * row, row); sy("kv", &kn);
        let n = pos + 1; let c0 = n.saturating_sub(self.window).max(kc0);
        let qr = self.q.rows; let qf = cb.empty(qr); cb.gemv(g(&self.q)?, &m, d, 0, &qf, qr, 0);
        let qn = cb.empty(qr); mg::rms_seg(cb, &qf, qr, &gw[5], hd, &qn, hd, 2 * hd, nh, 1, eps); mg::rope_seg(cb, &qn, qr, &gw[7], self.n_rot / 2, 2 * hd, nh, pos); sy("q", &qn);
        let att = cb.empty(nh * hd); mg::attn1(cb, &qn, &qf, qr, kc, vc, cap * row, &att, nh, c0, n - c0, row, hd, nh / self.nkv, (hd as f32).sqrt().recip(), cap); sy("attn", &att);
        let bo = cb.empty(d); cb.gemv(g(&self.o)?, &att, nh * hd, 0, &bo, d, 0);
        mg::hc_comb(cb, &rr, &bo, &inj, d, hc); sy("o", &rr);
        let (m2, inj2) = self.hc_mix_g(cb, g, &self.hcf, 3, &rr)?; sy("hcf", &m2);
        let rn = self.rsu.rows; let rsu = cb.empty(rn); cb.gemv(g(&self.rsu)?, &m2, d, 0, &rsu, rn, 0);
        let sf = (rn - self.ne) / 2; let gsb = cb.empty(sf); mg::silu2(cb, &rsu, rn, &gsb, self.ne, sf);
        let ysh = cb.empty(d); cb.gemv(g(&self.sd)?, &gsb, sf, 0, &ysh, d, 0); sy("shared", &ysh);
        let ne = self.ne; let pn = ne + 2 * d + 2 * row; let pk = cb.empty(pn);
        mg::cp(cb, &rsu, rn, &pk, pn, 0, 0, ne); mg::cp(cb, &m2, d, &pk, pn, 0, ne, d); mg::cp(cb, &ysh, d, &pk, pn, 0, ne + d, d);
        mg::cp(cb, &kn, row, &pk, pn, 0, ne + 2 * d, row); mg::cp(cb, &vf, row, &pk, pn, 0, ne + 2 * d + row, row);
        let hv = cb.read(&pk); tick(2, &mut tt);
        let (lg, m2c, yshc) = (&hv[..ne], &hv[ne..ne + d], &hv[ne + d..ne + 2 * d]);
        let mut y = self.routed(m2c, lg); let sgv = sigm(self.sgate.iter().zip(m2c).map(|(a, b)| a * b).sum::<f32>()); for i in 0..d { y[i] += yshc[i] * sgv; } tick(6, &mut tt);
        let yb = cb.buf(&y); mg::hc_comb(cb, &rr, &yb, &inj2, d, hc);
        let (s, _) = self.hc_mix_g(cb, g, &self.mix, 4, &rr)?; tick(7, &mut tt);
        Some(GStep { s, rr, krow: hv[ne + 2 * d..ne + 2 * d + row].to_vec(), vrow: hv[ne + 2 * d + row..].to_vec() })
    }
}
impl Mtp {
    /// PYRE_MTP_GRAPH (default on, =0 off): `step_gpu` with persistent buffers and the per-call scalars in a device buffer, captured as two
    /// CUDA graphs (before / after the CPU routed experts; the second includes the draft head GEMV). First call per key runs
    /// eagerly (allocates, compiles), the second runs eagerly then captures, later calls replay. `rid` = GPU K/V ring id.
    /// Returns (head logits handle, drafter residual handle, K row, V row).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn step_gpu_g<'a>(&self, cb: &Cb, g: &(dyn Fn(&Q8Mat) -> Option<&'a CM> + Sync), head: &CM, kc: &H, vc: &H, cap: usize, kc0: usize, r: &H, emb: &[f32], pos: usize, rid: u64) -> Option<(H, H, Vec<f32>, Vec<f32>)> {
        use super::mtp_gpu as mg; let (d, w, row, ne) = (self.d, self.hc * self.d, self.row(), self.ne); let mut tt = std::time::Instant::now();
        let mut gl = GG_S.lock().unwrap(); let key = (rid, cap, head.rows);
        if gl.as_ref().map(|x| x.key != key).unwrap_or(true) { *gl = Some(GG { fx: vec![cb.empty(4 + d), cb.empty(w), cb.empty(d), cb.empty(head.rows)], b1: vec![], b2: vec![], g1: None, g2: None, o1: None, key }); }
        let gg = gl.as_mut().unwrap(); let fx = gg.fx.clone();
        let n = pos + 1; let c0 = n.saturating_sub(self.window).max(kc0);
        let mut hv = vec![pos as f32, ((pos % cap) * row) as f32, c0 as f32, (n - c0) as f32]; hv.extend_from_slice(emb);
        let up = cb.buf(&hv); mg::cp(cb, &up, 4 + d, &fx[0], 4 + d, 0, 0, 4 + d); mg::cp(cb, r, w, &fx[1], w, 0, 0, w);
        let off = GG_OFF.load(std::sync::atomic::Ordering::Relaxed);
        // part 1
        let o1 = if let (Some(gr), Some(o)) = (gg.g1.as_ref(), gg.o1.clone()) { unsafe { gr.replay() }.expect("mtp graph replay"); GG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); o } else {
            let fresh = gg.b1.is_empty(); let o = self.g_run(cb, &mut gg.b1, &|al| self.gp1(cb, g, &fx, kc, vc, cap, al))?;
            if !fresh && !off { gg.g1 = self.g_capture(cb, &mut gg.b1, &|al| self.gp1(cb, g, &fx, kc, vc, cap, al)); }
            gg.o1 = Some(o.clone()); o };
        let (rr, inj2, pk) = o1; let pn = ne + 2 * d + 2 * row;
        let hv = cb.read(&pk); tick(2, &mut tt); let _ = pn;
        let (lg, m2c, yshc) = (&hv[..ne], &hv[ne..ne + d], &hv[ne + d..ne + 2 * d]);
        let mut y = self.routed(m2c, lg); let sgv = sigm(self.sgate.iter().zip(m2c).map(|(a, b)| a * b).sum::<f32>()); for i in 0..d { y[i] += yshc[i] * sgv; } tick(6, &mut tt);
        let yb = cb.buf(&y); mg::cp(cb, &yb, d, &fx[2], d, 0, 0, d);
        // part 2 (+ head)
        if let Some(gr) = gg.g2.as_ref() { unsafe { gr.replay() }.expect("mtp graph replay"); GG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); } else {
            let fresh = gg.b2.is_empty(); self.g_run(cb, &mut gg.b2, &|al| self.gp2(cb, g, head, &fx, &rr, &inj2, al))?;
            if !fresh && !off { gg.g2 = self.g_capture(cb, &mut gg.b2, &|al| self.gp2(cb, g, head, &fx, &rr, &inj2, al)); } }
        tick(7, &mut tt);
        Some((fx[3].clone(), rr, hv[ne + 2 * d..ne + 2 * d + row].to_vec(), hv[ne + 2 * d + row..].to_vec()))
    }
    /// run `f` eagerly with buffers from `bs` (allocated and recorded on the first run)
    fn g_run<T>(&self, cb: &Cb, bs: &mut Vec<H>, f: &(dyn Fn(&mut dyn FnMut(usize) -> H) -> Option<T> + Sync)) -> Option<T> {
        let mut i = 0usize; let mut al = |n: usize| { let h = if i < bs.len() { bs[i].clone() } else { let h = cb.empty(n); bs.push(h.clone()); h }; i += 1; h }; f(&mut al)
    }
    /// capture `f` (not executed) as a CUDA graph over the recorded buffers; None (and graphs off) on failure
    fn g_capture<T: Send>(&self, cb: &Cb, bs: &mut Vec<H>, f: &(dyn Fn(&mut dyn FnMut(usize) -> H) -> Option<T> + Sync)) -> Option<burn_cubecl::cubecl::client::Graph> {
        if cb.client.graph_prepare().is_err() { GG_OFF.store(true, std::sync::atomic::Ordering::Relaxed); return None; }
        let nb = bs.len();
        match cb.client.start_capture() { Ok(()) => { let ok = cb.client.memory_persistent_allocation((), |_| self.g_run(cb, bs, f).is_some());
                match cb.client.stop_capture() { Ok(gr) if ok && bs.len() == nb => { GG_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed); Some(gr) }
                    Ok(_) => { eprintln!("mtp graph: capture allocated or failed, off"); GG_OFF.store(true, std::sync::atomic::Ordering::Relaxed); None }
                    Err(e) => { eprintln!("mtp graph: capture failed, off: {e:?}"); GG_OFF.store(true, std::sync::atomic::Ordering::Relaxed); None } } }
            Err(e) => { eprintln!("mtp graph: start failed, off: {e:?}"); GG_OFF.store(true, std::sync::atomic::Ordering::Relaxed); let _ = cb.client.stop_capture(); None } }
    }
    #[allow(clippy::too_many_arguments)]
    fn gp1<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, fx: &[H], kc: &H, vc: &H, cap: usize, al: &mut dyn FnMut(usize) -> H) -> Option<(H, H, H)> {
        use super::mtp_gpu as mg; let (d, hc, nh, hd, eps) = (self.d, self.hc, self.nh, self.hd, self.eps); let w = hc * d; let row = self.row(); let gw = self.gw(cb).clone();
        let pb = &fx[0]; let eb = fx[0].clone().offset_start(16); let r = &fx[1];
        let en = al(d); mg::rms_seg(cb, &eb, d, &gw[0], d, &en, d, d, 1, 1, eps);
        let e2 = al(d); cb.gemv(g(&self.fc_e)?, &en, d, 0, &e2, d, 0);
        let hn = al(w); mg::rms_seg(cb, r, w, &gw[1], w, &hn, w, w, 1, 1, eps);
        let h2 = al(w); let fh = g(&self.fc_h)?; for s in 0..hc { cb.gemv(fh, &hn, w, s * d, &h2, w, s * d); }
        let rr = al(w); mg::add_b(cb, &h2, &e2, &rr, d, w);
        let (m, inj) = self.hc_mix_a(cb, g, &self.hca, 2, &rr, al)?;
        let kf = al(row); cb.gemv(g(&self.k)?, &m, d, 0, &kf, row, 0); let vf = al(row); cb.gemv(g(&self.v)?, &m, d, 0, &vf, row, 0);
        let kn = al(row); mg::rms_seg(cb, &kf, row, &gw[6], hd, &kn, hd, hd, self.nkv, 1, eps); mg::rope_p(cb, &kn, row, &gw[7], self.n_rot / 2, hd, self.nkv, pb);
        mg::cp_p(cb, &kn, row, kc, cap * row, pb, row); mg::cp_p(cb, &vf, row, vc, cap * row, pb, row);
        let qr = self.q.rows; let qf = al(qr); cb.gemv(g(&self.q)?, &m, d, 0, &qf, qr, 0);
        let qn = al(qr); mg::rms_seg(cb, &qf, qr, &gw[5], hd, &qn, hd, 2 * hd, nh, 1, eps); mg::rope_p(cb, &qn, qr, &gw[7], self.n_rot / 2, 2 * hd, nh, pb);
        let att = al(nh * hd); mg::attn1p(cb, &qn, &qf, qr, kc, vc, cap * row, &att, nh, pb, row, hd, nh / self.nkv, (hd as f32).sqrt().recip(), cap);
        let bo = al(d); cb.gemv(g(&self.o)?, &att, nh * hd, 0, &bo, d, 0);
        mg::hc_comb(cb, &rr, &bo, &inj, d, hc);
        let (m2, inj2) = self.hc_mix_a(cb, g, &self.hcf, 3, &rr, al)?;
        let rn = self.rsu.rows; let rsu = al(rn); cb.gemv(g(&self.rsu)?, &m2, d, 0, &rsu, rn, 0);
        let sf = (rn - self.ne) / 2; let gsb = al(sf); mg::silu2(cb, &rsu, rn, &gsb, self.ne, sf);
        let ysh = al(d); cb.gemv(g(&self.sd)?, &gsb, sf, 0, &ysh, d, 0);
        let ne = self.ne; let pn = ne + 2 * d + 2 * row; let pk = al(pn);
        mg::cp(cb, &rsu, rn, &pk, pn, 0, 0, ne); mg::cp(cb, &m2, d, &pk, pn, 0, ne, d); mg::cp(cb, &ysh, d, &pk, pn, 0, ne + d, d);
        mg::cp(cb, &kn, row, &pk, pn, 0, ne + 2 * d, row); mg::cp(cb, &vf, row, &pk, pn, 0, ne + 2 * d + row, row);
        Some((rr, inj2, pk))
    }
    #[allow(clippy::too_many_arguments)]
    fn gp2<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, head: &CM, fx: &[H], rr: &H, inj2: &H, al: &mut dyn FnMut(usize) -> H) -> Option<()> {
        use super::mtp_gpu as mg; let (d, hc) = (self.d, self.hc);
        mg::hc_comb(cb, rr, &fx[2], inj2, d, hc);
        let (s, _) = self.hc_mix_a(cb, g, &self.mix, 4, rr, al)?;
        cb.gemv(head, &s, d, 0, &fx[3], head.rows, 0); Some(())
    }
}
/// argmax and its softmax probability
pub fn best(lg: &[f32]) -> (usize, f32) {
    let (bi, bv) = lg.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
    let z: f32 = lg.iter().map(|v| (v - bv).exp()).sum(); (bi, 1.0 / z)
}
/// The draft vocabulary: PYRE_MTP_DRAFT_VOCAB = en (default) | cjk (Strata's wider default) | full | <file of u32 ids>.
pub fn draft_vocab(vocab: usize) -> Vec<u32> {
    static EN: &[u8] = include_bytes!("../../data/draft_vocab_en.bin");
    static CJK: &[u8] = include_bytes!("../../data/draft_vocab_cjk.bin");
    let sel = std::env::var("PYRE_MTP_DRAFT_VOCAB").unwrap_or_else(|_| "en".into());
    let raw: Vec<u8> = match sel.as_str() { "en" => EN.to_vec(), "cjk" => CJK.to_vec(), "full" => return (0..vocab as u32).collect(), p => std::fs::read(p).unwrap_or_else(|_| EN.to_vec()) };
    raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).filter(|&t| (t as usize) < vocab).collect()
}

/// The MTP file: PYRE_MTP=<path> (or 0 = off), else the HF cache copy written by scripts/mtp_fetch.py.
pub fn find() -> Option<std::path::PathBuf> {
    match std::env::var("PYRE_MTP") { Ok(v) if v == "0" => return None, Ok(v) => return Some(v.into()), _ => {} }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let snap = std::path::Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots");
    std::fs::read_dir(snap).ok()?.filter_map(|e| e.ok()).map(|e| e.path().join("mtp.safetensors")).find(|p| p.exists())
}
/// PYRE_MTP_GX (=1 on): routed experts on the GPU from a resident LRU pool of Q8 expert slots (PYRE_MTP_GX_MB, default 400)
/// and k draft steps chained on the GPU (argmax -> Q8 draft-vocab embedding -> next step) with one readback per draft run.
/// Experts not resident are skipped for that draft (weights renormalised over the resident picks); misses are uploaded
/// after the run (up to PYRE_MTP_GX_UP per run, default 16). Draft-only numerics: verify decides every token.
pub fn gx_on() -> bool { static M: OnceLock<bool> = OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_MTP_GX").map(|v| v != "0").unwrap_or(false)) }
pub static GX_ST: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
const GXK: usize = 16;
pub struct GXP { n: usize, wsl: usize, ssl: usize, wsd: usize, ssd: usize, wgu: H, sgu: H, wdn: H, sdn: H, smap: H, sgw: H, ew: H, es: H, ewn: usize, esn: usize,
    slot_of: Vec<i32>, exp_of: Vec<i32>, last: Vec<u64>, clock: u64 }
pub struct GG3 { fx: Vec<H>, b: Vec<H>, g: Option<burn_cubecl::cubecl::client::Graph>, o: Option<(H, H, H)>, key: (u64, usize, usize), ob: H }
pub static GX_S: std::sync::Mutex<Option<(Option<GXP>, Option<GG3>)>> = std::sync::Mutex::new(None);
fn q8ws(q: &Q8Mat) -> (Vec<f32>, Vec<f32>) {
    let mut w = vec![0f32; q.rows * q.cols / 4]; let mut s = vec![0f32; q.rows * q.cols / 32];
    w.par_chunks_mut(8 * 256).zip(s.par_chunks_mut(256)).zip(q.blocks().par_chunks(34 * 256)).for_each(|((w, s), bl)| {
        for (b, blk) in bl.chunks_exact(34).enumerate() { s[b] = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32(); for (i, c) in blk[2..].chunks_exact(4).enumerate() { w[b * 8 + i] = f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } } });
    (w, s)
}
impl Mtp {
    fn gx_new(&self, cb: &Cb, emb: &(dyn Fn(u32, &mut [f32]) + Sync)) -> Option<GXP> {
        let (d, ff, ne) = (self.d, self.ff, self.ne); if ne > 1024 || self.top_k > 64 || d % 32 != 0 || ff % 32 != 0 { eprintln!("mtp gx: unsupported shape, off"); return None; }
        let mb: usize = std::env::var("PYRE_MTP_GX_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
        let per = 3 * d * ff * 9 / 8; let n = (mb << 20) / per; if n < self.top_k { eprintln!("mtp gx: budget too small"); return None; } let n = n.min(ne);
        let nh = self.head.rows; let eb = nh * d * 9 / 8;
        if !cb.take(n * per + eb) { eprintln!("mtp gx: no vram budget, off"); return None; }
        let t0 = std::time::Instant::now();
        let (mut ew, mut es) = (vec![0f32; nh * d / 4], vec![0f32; nh * d / 32]);
        ew.par_chunks_mut(d / 4).zip(es.par_chunks_mut(d / 32)).zip(self.ids.par_iter()).for_each(|((ew, es), &id)| { let mut row = vec![0f32; d]; emb(id, &mut row);
            for (bi, b) in row.chunks_exact(32).enumerate() { let am = b.iter().fold(0f32, |a, v| a.max(v.abs())); let sc = am / 127.0; let iv = if sc > 0.0 { 1.0 / sc } else { 0.0 }; es[bi] = sc;
                for (ci, c) in b.chunks_exact(4).enumerate() { let q: Vec<u8> = c.iter().map(|v| ((v * iv).round().clamp(-127.0, 127.0) as i8) as u8).collect(); ew[bi * 8 + ci] = f32::from_bits(u32::from_le_bytes([q[0], q[1], q[2], q[3]])); } } });
        let (wsl, ssl, wsd, ssd) = (2 * ff * d / 4, 2 * ff * d / 32, d * ff / 4, d * ff / 32);
        let p = GXP { n, wsl, ssl, wsd, ssd, wgu: cb.empty(n * wsl), sgu: cb.empty(n * ssl), wdn: cb.empty(n * wsd), sdn: cb.empty(n * ssd), smap: cb.buf(&vec![-1f32; ne]), sgw: cb.buf(&self.sgate),
            ewn: ew.len(), esn: es.len(), ew: cb.buf(&ew), es: cb.buf(&es), slot_of: vec![-1; ne], exp_of: vec![-1; n], last: vec![0; n], clock: 0 };
        eprintln!("mtp gx: {n} expert slots ({:.0} MB) + draft-vocab embedding ({:.0} MB) on gpu, {:.0} ms", (n * per) as f64 / 1e6, eb as f64 / 1e6, t0.elapsed().as_secs_f64() * 1e3);
        Some(p)
    }
    fn gx_upload(&self, cb: &Cb, p: &mut GXP, ids: &[usize]) {
        use super::mtp_gpu as mg; let up: usize = std::env::var("PYRE_MTP_GX_UP").ok().and_then(|v| v.parse().ok()).unwrap_or(2); let mut nup = 0; let mut ch = false; let tu = std::time::Instant::now();
        for &e in ids { if e >= self.ne { continue; } p.clock += 1; let s = p.slot_of[e];
            if s >= 0 { p.last[s as usize] = p.clock; GX_ST[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed); continue; }
            GX_ST[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed); if nup >= up { continue; }
            let sl = (0..p.n).min_by_key(|&i| p.last[i]).unwrap(); if p.exp_of[sl] >= 0 { p.slot_of[p.exp_of[sl] as usize] = -1; }
            let (gu, dn) = self.expert(e); let (w1, s1) = q8ws(gu); let (w2, s2) = q8ws(dn);
            for (src, dst, tot, off) in [(&w1, &p.wgu, p.n * p.wsl, sl * p.wsl), (&s1, &p.sgu, p.n * p.ssl, sl * p.ssl), (&w2, &p.wdn, p.n * p.wsd, sl * p.wsd), (&s2, &p.sdn, p.n * p.ssd, sl * p.ssd)] {
                let t = cb.buf(src); mg::cpw(cb, &t, dst, tot, off, src.len()); }
            p.exp_of[sl] = e as i32; p.slot_of[e] = sl as i32; p.last[sl] = p.clock; nup += 1; ch = true; GX_ST[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if ch { let v: Vec<f32> = p.slot_of.iter().map(|&s| s as f32).collect(); let t = cb.buf(&v); mg::cpw(cb, &t, &p.smap, self.ne, 0, self.ne); } GX_ST[3].fetch_add(tu.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    #[allow(clippy::too_many_arguments)]
    fn gpx<'a>(&self, cb: &Cb, g: &dyn Fn(&Q8Mat) -> Option<&'a CM>, head: &CM, fx: &[H], kc: &H, vc: &H, cap: usize, p: &GXP, al: &mut dyn FnMut(usize) -> H) -> Option<(H, H, H)> {
        use super::mtp_gpu as mg; let (d, ff, ne, kk, row) = (self.d, self.ff, self.ne, self.top_k, self.row()); let w = self.hc * d; let pn = ne + 2 * d + 2 * row;
        let (rr, inj2, pk) = self.gp1(cb, g, fx, kc, vc, cap, al)?;
        let sel = al(3 * kk); mg::xsel(cb, &pk, pn, &p.smap, &sel, ne, kk);
        let m2 = pk.clone().offset_start((ne * 4) as u64);
        let h = al(kk * 2 * ff); mg::xgv(cb, &m2, d, &p.wgu, p.n * p.wsl, &p.sgu, p.n * p.ssl, &sel, kk, &h, 2 * ff, d, 0, p.wsl, p.ssl);
        let a = al(kk * ff); mg::xact(cb, &h, &a, ff, kk);
        let o = al(kk * d); mg::xgv(cb, &a, kk * ff, &p.wdn, p.n * p.wsd, &p.sdn, p.n * p.ssd, &sel, kk, &o, d, ff, ff, p.wsd, p.ssd);
        mg::xsum(cb, &o, &sel, &pk, pn, &p.sgw, &fx[2], d, kk, ne);
        self.gp2(cb, g, head, fx, &rr, &inj2, al)?;
        let bo = al(2); cb.best_into(&fx[3], head.rows, &bo);
        mg::next(cb, &fx[0], &bo, &p.ew, p.ewn, &p.es, p.esn, d, cap, row, self.window);
        mg::cp(cb, &rr, w, &fx[1], w, 0, 0, w);
        Some((pk, sel, bo))
    }
    /// k chained GPU draft steps from position `pos` (see gx_on); returns per step (draft-head index, p) and step 0's K/V row
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn draft_gx<'a>(&self, cb: &Cb, g: &(dyn Fn(&Q8Mat) -> Option<&'a CM> + Sync), head: &CM, kc: &H, vc: &H, cap: usize, kc0: usize, r: &H, emb: &[f32], pos: usize, rid: u64, k: usize, embf: &(dyn Fn(u32, &mut [f32]) + Sync)) -> Option<(Vec<(usize, f32)>, Vec<f32>, Vec<f32>)> {
        use super::mtp_gpu as mg; let (d, w, row, kk) = (self.d, self.hc * self.d, self.row(), self.top_k); let k = k.min(GXK); let pn = self.ne + 2 * d + 2 * row; let _ = pn;
        let mut gl = GX_S.lock().unwrap(); if gl.is_none() { *gl = Some((self.gx_new(cb, embf), None)); }
        let (po, go) = gl.as_mut().unwrap(); let p = po.as_mut()?; let key = (rid, cap, head.rows);
        if go.as_ref().map(|x| x.key != key).unwrap_or(true) { *go = Some(GG3 { fx: vec![cb.empty(4 + d), cb.empty(w), cb.empty(d), cb.empty(head.rows)], b: vec![], g: None, o: None, key, ob: cb.empty(2 * GXK + GXK * kk + 2 * row) }); }
        let gg = go.as_mut().unwrap(); let fx = gg.fx.clone(); let obn = 2 * GXK + GXK * kk + 2 * row;
        let n = pos + 1; let c0 = n.saturating_sub(self.window).max(kc0);
        let mut hv = vec![pos as f32, ((pos % cap) * row) as f32, c0 as f32, (n - c0) as f32]; hv.extend_from_slice(emb);
        let up = cb.buf(&hv); mg::cp(cb, &up, 4 + d, &fx[0], 4 + d, 0, 0, 4 + d); mg::cp(cb, r, w, &fx[1], w, 0, 0, w);
        let off = GG_OFF.load(std::sync::atomic::Ordering::Relaxed);
        let pr: &GXP = p;
        for j in 0..k {
            let (pk, sel, bo) = if let (Some(gr), Some(o)) = (gg.g.as_ref(), gg.o.clone()) { unsafe { gr.replay() }.expect("mtp gx graph replay"); GG_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); o } else {
                let fresh = gg.b.is_empty(); let o = self.g_run(cb, &mut gg.b, &|al| self.gpx(cb, g, head, &fx, kc, vc, cap, pr, al))?;
                if !fresh && !off { gg.g = self.g_capture(cb, &mut gg.b, &|al| self.gpx(cb, g, head, &fx, kc, vc, cap, pr, al)); }
                gg.o = Some(o.clone()); o };
            mg::cp(cb, &bo, 2, &gg.ob, obn, 0, 2 * j, 2); mg::cp(cb, &sel, 3 * kk, &gg.ob, obn, 2 * kk, 2 * GXK + j * kk, kk);
            if j == 0 { mg::cp(cb, &pk, pn, &gg.ob, obn, self.ne + 2 * d, 2 * GXK + GXK * kk, 2 * row); }
        }
        let hv = cb.read(&gg.ob);
        let bs: Vec<(usize, f32)> = (0..k).map(|j| (hv[2 * j] as usize, hv[2 * j + 1])).collect();
        let ids: Vec<usize> = (0..k * kk).map(|i| hv[2 * GXK + i] as usize).collect();
        let kvo = 2 * GXK + GXK * kk; let (kr, vr) = (hv[kvo..kvo + row].to_vec(), hv[kvo + row..kvo + 2 * row].to_vec());
        self.gx_upload(cb, p, &ids);
        if std::env::var("PYRE_MTP_DEBUG").is_ok() { eprintln!("mtp gx: hits {} misses {} uploads {} ({:.1} ms)", GX_ST[0].load(std::sync::atomic::Ordering::Relaxed), GX_ST[1].load(std::sync::atomic::Ordering::Relaxed), GX_ST[2].load(std::sync::atomic::Ordering::Relaxed), GX_ST[3].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6); }
        Some((bs, kr, vr))
    }
}
