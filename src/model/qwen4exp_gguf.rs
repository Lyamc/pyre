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
}

struct Hc { norm: Vec<f32>, down: QMat, up: QMat, inject: Option<QMat> }
enum Mixer {
    Attn { q: QMat, k: QMat, v: QMat, o: QMat, qn: Vec<f32>, kn: Vec<f32> },
    Lin { qkv: QMat, z: QMat, beta: QMat, alpha: QMat, a: Vec<f32>, dt: Vec<f32>, conv: Vec<f32>, norm: Vec<f32>, out: QMat },
}
struct Ple { key: QMat, value: QMat, nk: Vec<f32>, nq: Vec<f32>, nc: Vec<f32>, conv: Vec<f32> }
struct Layer { hc_a: Hc, hc_f: Hc, mix: Mixer, ple: Option<Ple>, router: QMat, gate_e: QMat, up_e: QMat, down_e: QMat, sh_gate_inp: Vec<f32>, sh_gate: QMat, sh_up: QMat, sh_down: QMat }

pub struct Model { pub gpu: Option<super::qn_gpu::Gpu>, gmats: std::collections::HashMap<usize, super::qn_gpu::GMat>, gl: Vec<Option<super::qn_gpu::GLayer>>, pub cfg: Cfg, pub g: Gguf, embd: QMat, out: QMat, head: Hc, ple_tab: Option<QMat>, layers: Vec<Layer>, rope_inv: Vec<f32> }

pub struct State {
    pub tokens: Vec<u32>,
    conv: Vec<Vec<f32>>,  // per linear layer [(conv_k-1) * channels], oldest first
    ssm: Vec<Vec<f32>>,   // per linear layer [nv][dk][dv]
    k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, // per attn layer [pos][n_kv*hd]
    ple_hist: Vec<Vec<f32>>, // per PLE layer: ring of past normalized values, [(conv_k-1)*dil][hc*d]
}

fn rms(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { y[i] = x[i] * s * w[i]; } }
fn rms_inplace(x: &mut [f32], w: &[f32], eps: f32) { let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt().recip(); for i in 0..x.len() { x[i] *= s * w[i]; } }
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
            max_ctx: k("attention.indexer.top_k").unwrap_or(2048) as usize,
        };
        let hc = |p: &str, inj: bool| -> Result<Hc> { Ok(Hc { norm: g.vec(&format!("{p}_norm.weight"))?, down: g.mat(&format!("{p}_down.weight"))?, up: g.mat(&format!("{p}_up.weight"))?, inject: if inj { Some(g.mat(&format!("{p}_inject.weight"))?) } else { None } }) };
        let mut layers = vec![];
        for il in 0..n_layers {
            let p = format!("blk.{il}");
            let m = |s: &str| g.mat(&format!("{p}.{s}"));
            let v = |s: &str| g.vec(&format!("{p}.{s}"));
            let full = (il + 1) % cfg.full_every == 0;
            let mix = if full {
                Mixer::Attn { q: m("attn_q.weight")?, k: m("attn_k.weight")?, v: m("attn_v.weight")?, o: m("attn_output.weight")?, qn: v("attn_q_norm.weight")?, kn: v("attn_k_norm.weight")? }
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
        Ok(Model { gpu: None, gmats: Default::default(), gl: vec![], embd: g.mat("token_embd.weight")?, out, head, ple_tab, layers, rope_inv, cfg, g })
    }

    pub fn max_ctx(&self) -> usize { self.cfg.max_ctx }

    /// Bring every routed-expert page into this process's working set once, up front (the idea of
    /// Strata's resident expert arena, without a copy: the pages stay the OS file cache's). Without
    /// it, each decode step soft-faults on ~10k first-touched 4 KiB pages. Skipped when free RAM is
    /// short (keeps 16 GiB spare); `PYRE_QN_PREFAULT=0` turns it off. Returns GiB touched.
    pub fn prefault_experts(&self) -> f64 {
        if std::env::var("PYRE_QN_PREFAULT").map(|v| v == "0").unwrap_or(false) { return 0.0; }
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
        let (mut on, mut off) = (0, 0); let b0 = g.budget_left();
        let mut gl = vec![];
        for l in &self.layers {
            let min = match &l.mix { Mixer::Attn { q, k, v, .. } => vec![q, k, v], Mixer::Lin { qkv, z, beta, alpha, .. } => vec![qkv, z, beta, alpha] };
            let mout = match &l.mix { Mixer::Attn { o, .. } => o, Mixer::Lin { out, .. } => out };
            let parts = (|| -> Result<Option<GLayer>> { Ok(Some(GLayer {
                hca: match ghc(&l.hc_a)? { Some(x) => x, None => return Ok(None) },
                min: match g.upload(&min)? { Some(x) => x, None => return Ok(None) },
                mout: match g.upload(&[mout])? { Some(x) => x, None => return Ok(None) },
                hcf: match ghc(&l.hc_f)? { Some(x) => x, None => return Ok(None) },
                fin: match g.upload(&[&l.router, &l.sh_gate, &l.sh_up])? { Some(x) => x, None => return Ok(None) },
                sdown: match g.upload(&[&l.sh_down])? { Some(x) => x, None => return Ok(None) },
            })) })()?;
            if parts.is_some() { on += 1 } else { off += 1 }
            gl.push(parts);
        }
        let mut m = std::collections::HashMap::new();
        let mut extra: Vec<Vec<&QMat>> = vec![];
        for l in &self.layers { if let Some(p) = &l.ple { extra.push(vec![&p.key, &p.value]); } }
        extra.push(vec![&self.head.down]); extra.push(vec![&self.head.up]); extra.push(vec![&self.out]);
        for grp in extra { if let Some(gm) = g.upload(&grp)? { m.insert(grp[0].key(), gm); } }
        let gb = (b0 - g.budget_left()) as f64 / (1u64 << 30) as f64;
        self.gmats = m; self.gl = gl; self.gpu = Some(g);
        Ok((on, off, gb))
    }

    /// The layer with its dense half on the GPU: two device->host copies (mixer inputs; then the
    /// MoE input, router logits, shared-expert output and the updated residual).
    fn layer_gpu(&self, l: &Layer, gl: &super::qn_gpu::GLayer, x: &mut [f32], st: &mut State, ai: &mut usize, li: &mut usize, start: usize, t: usize) {
        let g = self.gpu.as_ref().unwrap(); let c = &self.cfg; let (d, hc, ne) = (c.d, c.hc, c.n_exp);
        let xg = g.up(x, [t, hc * d]);
        let (mixed, inj_a) = gl.hca.mix(xg.clone(), t, hc, d, c.eps);
        let mi = g.read(gl.min.fwd(mixed));
        let proj = gl.min.split_host(&mi, t);
        let core = match l.mix { Mixer::Attn { .. } => { let o = self.attn_core(l, *ai, st, proj, start, t); *ai += 1; o }, Mixer::Lin { .. } => { let o = self.linear_core(l, *li, st, proj, t); *li += 1; o } };
        let o = gl.mout.fwd(g.up(&core, [t, core.len() / t]));
        let x2 = super::qn_gpu::combine(xg, o, inj_a.unwrap(), t, hc, d);
        let (mf, inj_f) = gl.hcf.mix(x2.clone(), t, hc, d, c.eps);
        let fi = gl.fin.fwd(mf.clone()); let sf = l.sh_gate.rows;
        let lg = fi.clone().slice([0..t, 0..ne]);
        let sh = gl.sdown.fwd(burn::tensor::activation::silu(fi.clone().slice([0..t, ne..ne + sf])) * fi.slice([0..t, ne + sf..ne + 2 * sf]));
        let all = g.read(burn::tensor::Tensor::cat(vec![x2, mf, inj_f.unwrap(), lg, sh], 1));
        let w = hc * d + d + hc + ne + d;
        let (mut mfc, mut injc, mut lgc, mut y) = (Vec::with_capacity(t * d), Vec::with_capacity(t * hc), Vec::with_capacity(t * ne), Vec::with_capacity(t * d));
        for ti in 0..t { let r = &all[ti * w..(ti + 1) * w]; let mut o = 0;
            x[ti * hc * d..(ti + 1) * hc * d].copy_from_slice(&r[..hc * d]); o += hc * d;
            mfc.extend_from_slice(&r[o..o + d]); o += d; injc.extend_from_slice(&r[o..o + hc]); o += hc;
            lgc.extend_from_slice(&r[o..o + ne]); o += ne; y.extend_from_slice(&r[o..o + d]); }
        self.routed(l, &mfc, &lgc, t, &mut y);
        self.hc_combine(x, &y, &injc, t);
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
        let mut st = State { tokens: vec![], conv: vec![], ssm: vec![], k: vec![], v: vec![], ple_hist: vec![] };
        for l in &self.layers {
            match l.mix { Mixer::Lin { .. } => { st.conv.push(vec![0.0; (c.conv_k - 1) * ch]); st.ssm.push(vec![0.0; c.lin_nv * c.lin_dk * c.lin_dv]); }, Mixer::Attn { .. } => { st.k.push(vec![]); st.v.push(vec![]); } }
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
        let Mixer::Attn { q, k, v, o, .. } = &l.mix else { unreachable!() };
        let att = self.attn_core(l, ai, st, self.mm(&[q, k, v], x, t), start, t);
        self.mm(&[o], &att, t).pop().unwrap()
    }

    /// Attention after the q/k/v projections; returns the gated heads before the output projection.
    fn attn_core(&self, l: &Layer, ai: usize, st: &mut State, proj: Vec<Vec<f32>>, start: usize, t: usize) -> Vec<f32> {
        let Mixer::Attn { qn, kn, .. } = &l.mix else { unreachable!() };
        let c = &self.cfg; let (nh, nkv, hd) = (c.n_head, c.n_kv, c.hd);
        let mut r = proj.into_iter(); let qf = r.next().unwrap(); let mut kf = r.next().unwrap(); let vf = r.next().unwrap();
        for ti in 0..t { for h in 0..nkv { let kh = &mut kf[ti * nkv * hd + h * hd..ti * nkv * hd + (h + 1) * hd]; rms_inplace(kh, kn, c.eps); self.rope(kh, start + ti); } }
        st.k[ai].extend_from_slice(&kf); st.v[ai].extend_from_slice(&vf);
        let (kc, vc) = (&st.k[ai], &st.v[ai]);
        let scale = (hd as f32).sqrt().recip(); let grp = nh / nkv;
        let mut att = vec![0f32; t * nh * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
            let (ti, h) = (idx / nh, idx % nh);
            let base = ti * nh * 2 * hd + h * 2 * hd;
            let mut qh = qf[base..base + hd].to_vec(); rms_inplace(&mut qh, qn, c.eps); self.rope(&mut qh, start + ti);
            let kvh = h / grp; let n = start + ti + 1;
            let mut sc: Vec<f32> = (0..n).map(|p| dot(&qh, &kc[p * nkv * hd + kvh * hd..p * nkv * hd + (kvh + 1) * hd]) * scale).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            for (p, s) in sc.iter().enumerate() { let w = s / sum; let vv = &vc[p * nkv * hd + kvh * hd..]; for i in 0..hd { out[i] += w * vv[i]; } }
            let gate = &qf[base + hd..base + 2 * hd]; for i in 0..hd { out[i] *= sigm(gate[i]); }
        });
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
        for ti in 0..t {
            let cur = &m[ti * ch..(ti + 1) * ch];
            let mut co = vec![0f32; ch];
            for cc in 0..ch { let mut o = conv[cc * kk + kk - 1] * cur[cc]; for k in 0..kk - 1 { o += conv[cc * kk + k] * cs[k * ch + cc]; } co[cc] = silu(o); }
            cs.drain(..ch); cs.extend_from_slice(cur);
            let (qp, rest) = co.split_at_mut(nk * dk); let (kp, vp) = rest.split_at_mut(nk * dk);
            for h in 0..nk { for (buf, sc) in [(&mut qp[h * dk..(h + 1) * dk], (dk as f32).sqrt().recip()), (&mut kp[h * dk..(h + 1) * dk], 1.0)] {
                let n = (buf.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip() * sc; for v in buf.iter_mut() { *v *= n; } } }
            let (qp, kp, vp) = (&*qp, &*kp, &*vp);
            let o_t = &mut o_all[ti * nv * dv..(ti + 1) * nv * dv];
            s.par_chunks_mut(dk * dv).zip(o_t.par_chunks_mut(dv)).enumerate().for_each(|(h, (sh, oh))| {
                let kh = h % nk; let kv = &kp[kh * dk..(kh + 1) * dk]; let qv = &qp[kh * dk..(kh + 1) * dk]; let vv = &vp[h * dv..(h + 1) * dv];
                let g = (softplus(aa[ti * nv + h] + dt[h]) * a[h]).exp(); let b = sigm(bb[ti * nv + h]);
                for v in sh.iter_mut() { *v *= g; }
                let mut mem = vec![0f32; dv]; for i in 0..dk { let ki = kv[i]; let row = &sh[i * dv..(i + 1) * dv]; for j in 0..dv { mem[j] += row[j] * ki; } }
                let delta: Vec<f32> = (0..dv).map(|j| (vv[j] - mem[j]) * b).collect();
                for i in 0..dk { let ki = kv[i]; let qi = qv[i]; let row = &mut sh[i * dv..(i + 1) * dv]; for j in 0..dv { row[j] += ki * delta[j]; oh[j] += row[j] * qi; } }
                rms_inplace(oh, norm, c.eps); let zh = &zz[ti * nv * dv + h * dv..ti * nv * dv + (h + 1) * dv]; for j in 0..dv { oh[j] *= sigm(zh[j]); }
            });
        }
        o_all
    }

    fn ffn(&self, l: &Layer, x: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg; let d = c.d; let ff = c.ff_exp;
        let mut r = self.mm(&[&l.router, &l.sh_gate, &l.sh_up], x, t).into_iter();
        let (lg, mut g, u) = (r.next().unwrap(), r.next().unwrap(), r.next().unwrap());
        // shared expert (batched)
        for i in 0..g.len() { g[i] = silu(g[i]) * u[i]; }
        let mut y = self.mm(&[&l.sh_down], &g, t).pop().unwrap();
        self.routed(l, x, &lg, t, &mut y);
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
    fn routed(&self, l: &Layer, x: &[f32], lg: &[f32], t: usize, y: &mut [f32]) {
        let c = &self.cfg; let d = c.d; let ff = c.ff_exp; let t_r = std::time::Instant::now();
        for ti in 0..t { let sg = sigm(dot(&l.sh_gate_inp, &x[ti * d..(ti + 1) * d])); for v in &mut y[ti * d..(ti + 1) * d] { *v *= sg; } }
        // routing: softmax -> top-k -> renormalise
        let mut by_exp: Vec<Vec<(usize, f32)>> = vec![vec![]; c.n_exp];
        for ti in 0..t {
            let r = &lg[ti * c.n_exp..(ti + 1) * c.n_exp];
            let mut idx: Vec<usize> = (0..c.n_exp).collect();
            idx.select_nth_unstable_by(c.top_k, |a, b| r[*b].partial_cmp(&r[*a]).unwrap());
            let top = &idx[..c.top_k]; let mx = top.iter().map(|&e| r[e]).fold(f32::NEG_INFINITY, f32::max);
            let ws: Vec<f32> = top.iter().map(|&e| (r[e] - mx).exp()).collect(); let sum: f32 = ws.iter().sum();
            for (&e, w) in top.iter().zip(ws) { by_exp[e].push((ti, w / sum)); }
        }
        let work: Vec<(usize, &Vec<(usize, f32)>)> = by_exp.iter().enumerate().filter(|(_, v)| !v.is_empty()).collect();
        if t * c.top_k >= work.len() * env_usize("PYRE_QN_GEMM_MIN").unwrap_or(8) { // >= 8 tokens per expert on average
            self.routed_gemm(l, x, &work, t, y);
            ROUTED_NS.fetch_add(t_r.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        // one task per (expert, block of rows): keeps all cores busy for decode (10 experts) and prefill alike
        let blk: usize = env_usize("PYRE_QN_BLK").unwrap_or(if t == 1 { 64 } else { 64 });
        use super::qdot;
        let qon = std::env::var("PYRE_QN_QDOT").map(|v| v != "0").unwrap_or(true);
        let qgu = qon && qdot::supports(l.gate_e.ty) && qdot::supports(l.up_e.ty);
        let qdn = qon && qdot::supports(l.down_e.ty);
        let acts: Vec<qdot::Q8Act> = if qgu { (0..t).map(|ti| qdot::quantize(&x[ti * d..(ti + 1) * d])).collect() } else { vec![] }; let acts = &acts;
        let parts: Vec<(usize, usize, Vec<f32>)> = work.par_iter().flat_map(|&(e, toks)| (0..ff.div_ceil(blk)).into_par_iter().map(move |b| {
            let (r0, r1) = (b * blk, ((b + 1) * blk).min(ff));
            let gm = l.gate_e.slice_rows(e * ff + r0, r1 - r0); let um = l.up_e.slice_rows(e * ff + r0, r1 - r0);
            let mut hbuf = vec![0f32; toks.len() * (r1 - r0)];
            if qgu { let mut sc = qdot::Scratch::default();
                let mut s2 = qdot::Scratch::default();
                for r in 0..r1 - r0 { let (gr, ur) = (gm.row_raw(r), um.row_raw(r));
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
        ROUTED_NS.fetch_add(t_r.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Feed `ids` (appended to `st.tokens`) and return the logits of the last one.
    pub fn forward(&self, ids: &[u32], st: &mut State) -> Vec<f32> {
        let c = &self.cfg; let (d, hc) = (c.d, c.hc); let t = ids.len(); let start = st.tokens.len();
        st.tokens.extend_from_slice(ids);
        let mut x = vec![0f32; t * hc * d];
        let mut e = vec![0f32; d];
        for (ti, &id) in ids.iter().enumerate() { self.embd.row(id as usize, &mut e); for s in 0..hc { x[(ti * hc + s) * d..(ti * hc + s + 1) * d].copy_from_slice(&e); } }
        let emb = if self.ple_tab.is_some() { Some(self.ple_rows(st, start, t)) } else { None };
        let (mut li, mut ai, mut pi) = (0, 0, 0);
        let dbg = std::env::var("PYRE_QN_DEBUG").is_ok();
        let prof = std::env::var("PYRE_QN_PROFILE").is_ok(); let mut prof_acc = [0f64; 5]; let t_all = std::time::Instant::now();
        for (il, l) in self.layers.iter().enumerate() {
            let tt = std::time::Instant::now();
            if let Some(p) = &l.ple { let mut h = std::mem::take(&mut st.ple_hist[pi]); self.ple(p, &mut h, emb.as_ref().unwrap(), &mut x, t); st.ple_hist[pi] = h; pi += 1; }
            if let Some(Some(gl)) = self.gl.get(il) {
                self.layer_gpu(l, gl, &mut x, st, &mut ai, &mut li, start, t);
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
        let last = &x[(t - 1) * hc * d..t * hc * d];
        let (h, _) = self.hc_mix(&self.head, last, 1);
        let th = std::time::Instant::now();
        let lg = self.mm(&[&self.out], &h, 1).pop().unwrap();
        if prof { prof_acc[4] = th.elapsed().as_secs_f64(); let (gn, gs) = super::qn_gpu::take_stats(); let rs = ROUTED_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9; eprint!("routed {rs:.3}s (gate/up {:.3}) ", PH[0].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9); eprintln!("prof T={t} total {:.3}s | hc {:.3} attn {:.3} gdn {:.3} ffn {:.3} head {:.3} | gpu {gn} calls {gs:.3}s", t_all.elapsed().as_secs_f64(), prof_acc[0], prof_acc[1], prof_acc[2], prof_acc[3], prof_acc[4]); }
        lg
    }
}

static PH: [std::sync::atomic::AtomicU64; 1] = [std::sync::atomic::AtomicU64::new(0)];
static ROUTED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn env_usize(k: &str) -> Option<usize> { std::env::var(k).ok().and_then(|v| v.parse().ok()) }
