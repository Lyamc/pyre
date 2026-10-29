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
    fn q8(&self, n: &str) -> Result<Q8Mat> { let (s, b) = self.raw(n)?; let (r, c) = (s[0], s[1]); Ok(Q8Mat::from_fn(r, c, |i, y| Self::row(b, i, c, y))) }
}

/// PYRE_MTP_DEBUG timers (ns): input, attn-hc+kv, q+attention, o+combine, mlp-hc, shared, experts, mixer
pub static TM: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
fn tick(i: usize, t: &mut std::time::Instant) { let n = t.elapsed().as_nanos() as u64; TM[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed); *t = std::time::Instant::now(); }
struct Hc { norm: Vec<f32>, down: Q8Mat, up: Q8Mat, inject: Option<Q8Mat> }
fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn silu(x: f32) -> f32 { x * sigm(x) }
fn rms(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { y[i] = x[i] * s * w[i]; } }
fn mm(m: &Q8Mat, x: &[f32], t: usize) -> Vec<f32> { let mut y = vec![0f32; t * m.rows]; m.matmul(x, t, &mut y); y }

/// The drafter's K/V: cells c0..c0+n (rope position = cell), [cell][n_kv*hd].
#[derive(Default)]
pub struct Kv { pub c0: usize, pub n: usize, k: Vec<f32>, v: Vec<f32> }
impl Kv {
    pub fn end(&self) -> usize { self.c0 + self.n }
    pub fn truncate(&mut self, end: usize, row: usize) { if end < self.end() { let n = end.saturating_sub(self.c0); self.n = n; self.k.truncate(n * row); self.v.truncate(n * row); } }
}

pub struct Mtp {
    d: usize, hc: usize, nh: usize, nkv: usize, hd: usize, n_rot: usize, eps: f32, top_k: usize, ff: usize, rope_inv: Vec<f32>,
    pre_e: Vec<f32>, pre_h: Vec<f32>, fc_e: Q8Mat, fc_h: Q8Mat, hca: Hc, hcf: Hc, mix: Hc,
    q: Q8Mat, k: Q8Mat, v: Q8Mat, o: Q8Mat, qn: Vec<f32>, kn: Vec<f32>,
    router: Q8Mat, sg: Q8Mat, su: Q8Mat, sd: Q8Mat, sgate: Vec<f32>,
    st: St, ex: Vec<OnceLock<(Q8Mat, Q8Mat)>>,
    head: Q8Mat, ids: Vec<u32>, pub window: usize,
}

pub struct Geo { pub d: usize, pub hc: usize, pub nh: usize, pub nkv: usize, pub hd: usize, pub n_rot: usize, pub eps: f32, pub top_k: usize, pub rope_base: f32 }

impl Mtp {
    /// `head_row(token, out)`: a row of the main model's output matrix; `ids`: the draft vocabulary.
    pub fn load(p: &std::path::Path, g: &Geo, ids: Vec<u32>, head_row: impl Fn(usize, &mut [f32]) + Sync) -> Result<Self> {
        let st = St::open(p)?;
        let hcm = |p: &str, inj: bool| -> Result<Hc> { Ok(Hc { norm: st.norm(&format!("{p}.hc_norm.weight"))?, down: st.q8(&format!("{p}.input_mix_weight_down.weight"))?, up: st.q8(&format!("{p}.input_mix_weight_up.weight"))?,
            inject: if inj { Some(st.q8(&format!("{p}.block_inject_weight.weight"))?) } else { None } }) };
        let (ne, ff) = { let (s, _) = st.raw("mlp.experts.down_proj")?; (s[0], s[2]) };
        let half = g.n_rot / 2;
        let head = Q8Mat::from_fn(ids.len(), g.d, |r, y| head_row(ids[r] as usize, y));
        Ok(Mtp { d: g.d, hc: g.hc, nh: g.nh, nkv: g.nkv, hd: g.hd, n_rot: g.n_rot, eps: g.eps, top_k: g.top_k, ff,
            rope_inv: (0..half).map(|i| g.rope_base.powf(-(2.0 * i as f32) / g.n_rot as f32)).collect(),
            pre_e: st.norm("pre_fc_norm_embedding.weight")?, pre_h: st.norm("pre_fc_norm_hidden.weight")?, fc_e: st.q8("fc_embedding.weight")?, fc_h: st.q8("fc_hidden.weight")?,
            hca: hcm("attn_hyper_connection", true)?, hcf: hcm("mlp_hyper_connection", true)?, mix: hcm("hyper_connection_mixer", false)?,
            q: st.q8("self_attn.q_proj.weight")?, k: st.q8("self_attn.k_proj.weight")?, v: st.q8("self_attn.v_proj.weight")?, o: st.q8("self_attn.o_proj.weight")?,
            qn: st.norm("self_attn.q_norm.weight")?, kn: st.norm("self_attn.k_norm.weight")?,
            router: st.q8("mlp.gate.weight")?, sg: st.q8("mlp.shared_expert.gate_proj.weight")?, su: st.q8("mlp.shared_expert.up_proj.weight")?, sd: st.q8("mlp.shared_expert.down_proj.weight")?,
            sgate: st.vec("mlp.shared_expert_gate.weight")?, ex: (0..ne).map(|_| OnceLock::new()).collect(), st, head, ids,
            window: std::env::var("PYRE_MTP_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(256) })
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
        let mut lo = mm(&h.down, &xn, t); for v in lo.iter_mut() { *v = silu(*v / hc as f32); }
        let inj = match &h.inject { Some(m) => { let mut v = mm(m, &xn, t); for x in v.iter_mut() { *x /= hc as f32; } v }, None => vec![0f32; t * hc] };
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
        let lg = mm(&self.router, &m2, 1);
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.select_nth_unstable_by(self.top_k, |a, b| lg[*b].partial_cmp(&lg[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let top = &idx[..self.top_k]; let mx = top.iter().map(|&e| lg[e]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = top.iter().map(|&e| (lg[e] - mx).exp()).collect(); let wsum: f32 = ws.iter().sum();
        let mut gs = mm(&self.sg, &m2, 1); let us = mm(&self.su, &m2, 1); for i in 0..gs.len() { gs[i] = silu(gs[i]) * us[i]; }
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