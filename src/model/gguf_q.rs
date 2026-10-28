//! Memory-mapped multi-shard GGUF reader plus CPU dequantization and multi-threaded
//! quantized matmul for the ggml block formats used by the Qwen3.8-Flash-Next GGUFs.
//!
//! The block layouts and decode expressions are transcribed from ggml-org/llama.cpp
//! `ggml/src/ggml-common.h` / `ggml/src/ggml-quants.c` (MIT licence, (c) the ggml authors);
//! Q2_0 (type 42) also matches Strata's `tools/gguf_writer.py::dequantize_q2_0` (MIT, Niko1221/Strata).
//! Weights are never expanded in memory: each worker dequantizes one row at a time into a
//! small scratch buffer and dots it against the activations, so routed experts stay quantized
//! in RAM (file-backed pages from the mmap).
use super::iq_tables::*;
use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType { F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_0, Q3K, Q4K, Q5K, Q6K, IQ4NL, IQ4XS, IQ2XXS, IQ2XS, IQ2S, IQ3S, IQ1M }

impl QType {
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32, 1 => Self::F16, 30 => Self::BF16, 2 => Self::Q4_0, 6 => Self::Q5_0, 8 => Self::Q8_0,
            42 => Self::Q2_0, 11 => Self::Q3K, 12 => Self::Q4K, 13 => Self::Q5K, 14 => Self::Q6K,
            20 => Self::IQ4NL, 23 => Self::IQ4XS, 16 => Self::IQ2XXS, 17 => Self::IQ2XS, 22 => Self::IQ2S,
            21 => Self::IQ3S, 29 => Self::IQ1M,
            _ => bail!("unsupported ggml type {id}"),
        })
    }
    /// (elements per block, bytes per block)
    pub fn geom(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4), Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18), Self::Q5_0 => (32, 22), Self::Q8_0 => (32, 34), Self::Q2_0 => (64, 18),
            Self::Q3K => (256, 110), Self::Q4K => (256, 144), Self::Q5K => (256, 176), Self::Q6K => (256, 210),
            Self::IQ4NL => (32, 18), Self::IQ4XS => (256, 136), Self::IQ2XXS => (256, 66), Self::IQ2XS => (256, 74),
            Self::IQ2S => (256, 82), Self::IQ3S => (256, 110), Self::IQ1M => (256, 56),
        }
    }
    pub fn row_bytes(self, n: usize) -> usize { let (e, b) = self.geom(); n / e * b }
}

struct Lut { q2: Vec<[f32; 4]>, sign8: Vec<[f32; 8]>, iq2xxs: Vec<[f32; 8]>, iq2xs: Vec<[f32; 8]>, iq2s: Vec<[f32; 8]>, iq3s: Vec<[f32; 4]> }
fn lut() -> &'static Lut {
    static L: std::sync::OnceLock<Lut> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let g8 = |t: &[u64]| t.iter().map(|v| { let b = v.to_le_bytes(); std::array::from_fn(|j| b[j] as f32) }).collect::<Vec<[f32; 8]>>();
        Lut {
            q2: (0..256u32).map(|b| std::array::from_fn(|j| (((b >> (2 * j)) & 3) as i32 - 1) as f32)).collect(),
            sign8: (0..256u32).map(|s| std::array::from_fn(|j| if s & (1 << j) != 0 { -1.0 } else { 1.0 })).collect(),
            iq2xxs: g8(&IQ2XXS_GRID), iq2xs: g8(&IQ2XS_GRID), iq2s: g8(&IQ2S_GRID),
            iq3s: IQ3S_GRID.iter().map(|v| { let b = v.to_le_bytes(); std::array::from_fn(|j| b[j] as f32) }).collect(),
        }
    })
}
#[inline(always)] fn put8(y: &mut [f32], d: f32, g: &[f32; 8], s: &[f32; 8]) { let y = &mut y[..8]; for j in 0..8 { y[j] = d * g[j] * s[j]; } }

#[inline] fn h2f(b: &[u8], o: usize) -> f32 { f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32() }

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) } else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

/// Dequantize `y.len()` elements (a whole number of blocks) from `x`.
pub fn dequant(t: QType, x: &[u8], y: &mut [f32]) {
    let (qk, bs) = t.geom();
    let nb = y.len() / qk;
    match t {
        QType::F32 => for (i, v) in y.iter_mut().enumerate() { *v = f32::from_le_bytes([x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]]) },
        QType::F16 => for (i, v) in y.iter_mut().enumerate() { *v = h2f(x, 2 * i) },
        QType::BF16 => { let x = &x[..2 * y.len()]; for (v, c) in y.iter_mut().zip(x.chunks_exact(2)) { *v = f32::from_bits(((c[0] as u32) | ((c[1] as u32) << 8)) << 16) } },
        QType::Q4_0 => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let y = &mut y[i * 32..];
            for j in 0..16 { y[j] = ((b[2 + j] & 0xF) as i32 - 8) as f32 * d; y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d; } },
        QType::Q5_0 => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]); let qs = &b[6..]; let y = &mut y[i * 32..];
            for j in 0..16 { let h0 = (((qh >> j) << 4) & 0x10) as u8; let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                y[j] = (((qs[j] & 0xF) | h0) as i32 - 16) as f32 * d; y[j + 16] = (((qs[j] >> 4) | h1) as i32 - 16) as f32 * d; } },
        QType::Q8_0 => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); for j in 0..32 { y[i * 32 + j] = (b[2 + j] as i8) as f32 * d; } },
        QType::Q2_0 => { let lt = lut(); for i in 0..nb { let b = &x[i * bs..i * bs + 18]; let d = h2f(b, 0); let y = &mut y[i * 64..i * 64 + 64];
            for (k, yy) in y.chunks_exact_mut(4).enumerate() { let q = &lt.q2[b[2 + k] as usize]; for j in 0..4 { yy[j] = q[j] * d; } } } },
        QType::IQ4NL => { let kv: [f32; 16] = std::array::from_fn(|i| KVALUES_IQ4NL[i] as f32); for i in 0..nb { let b = &x[i * bs..i * bs + 18]; let d = h2f(b, 0); let (lo, hi) = y[i * 32..i * 32 + 32].split_at_mut(16);
            for j in 0..16 { lo[j] = d * kv[(b[2 + j] & 0xF) as usize]; hi[j] = d * kv[(b[2 + j] >> 4) as usize]; } } },
        QType::IQ4XS => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let sh = u16::from_le_bytes([b[2], b[3]]); let sl = &b[4..8]; let mut qs = &b[8..];
            let mut y = &mut y[i * 256..];
            for ib in 0..8 { let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4); let dl = d * (ls - 32) as f32;
                let kv: [f32; 16] = std::array::from_fn(|i| KVALUES_IQ4NL[i] as f32 * dl); let q16 = &qs[..16]; let (lo, hi) = y[..32].split_at_mut(16);
                for j in 0..16 { lo[j] = kv[(q16[j] & 0xF) as usize]; hi[j] = kv[(q16[j] >> 4) as usize]; }
                y = &mut y[32..]; qs = &qs[16..]; } },
        QType::Q3K => for i in 0..nb { let b = &x[i * bs..]; let hm = &b[0..32]; let mut q = &b[32..96]; let sc = &b[96..108]; let d_all = h2f(b, 108);
            let a = |k: usize| u32::from_le_bytes([sc[4 * k], sc[4 * k + 1], sc[4 * k + 2], sc[4 * k + 3]]);
            let (a0, a1, tmp) = (a(0), a(1), a(2)); let (k1, k2) = (0x03030303u32, 0x0f0f0f0fu32);
            let aux = [(a0 & k2) | ((tmp & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4), ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)];
            let scales: Vec<i8> = aux.iter().flat_map(|v| v.to_le_bytes()).map(|v| v as i8).collect();
            let mut y = &mut y[i * 256..]; let mut is = 0; let mut m = 1u8;
            for _n in 0..2 { let mut shift = 0;
                for _j in 0..4 {
                    for half in 0..2 { let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 { let o = l + 16 * half; y[l] = dl * (((q[o] >> shift) & 3) as i32 - if hm[o] & m != 0 { 0 } else { 4 }) as f32; }
                        y = &mut y[16..]; }
                    shift += 2; m <<= 1; }
                q = &q[32..]; } },
        QType::Q4K => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let mn = h2f(b, 2); let sc = &b[4..16]; let mut q = &b[16..]; let mut y = &mut y[i * 256..];
            for j in 0..4 { let (s1, m1) = scale_min_k4(2 * j, sc); let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                let (d1, m1, d2, m2) = (d * s1 as f32, mn * m1 as f32, d * s2 as f32, mn * m2 as f32);
                for l in 0..32 { y[l] = d1 * (q[l] & 0xF) as f32 - m1; y[l + 32] = d2 * (q[l] >> 4) as f32 - m2; }
                y = &mut y[64..]; q = &q[32..]; } },
        QType::Q5K => for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let mn = h2f(b, 2); let sc = &b[4..16]; let qh = &b[16..48]; let mut ql = &b[48..]; let mut y = &mut y[i * 256..];
            let (mut u1, mut u2) = (1u8, 2u8);
            for j in 0..4 { let (s1, m1) = scale_min_k4(2 * j, sc); let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                let (d1, m1, d2, m2) = (d * s1 as f32, mn * m1 as f32, d * s2 as f32, mn * m2 as f32);
                for l in 0..32 { y[l] = d1 * ((ql[l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32 - m1; y[l + 32] = d2 * ((ql[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32 - m2; }
                y = &mut y[64..]; ql = &ql[32..]; u1 <<= 2; u2 <<= 2; } },
        QType::Q6K => for i in 0..nb { let b = &x[i * bs..]; let mut ql = &b[0..128]; let mut qh = &b[128..192]; let mut sc = &b[192..208]; let d = h2f(b, 208); let mut y = &mut y[i * 256..];
            for _n in 0..2 {
                for l in 0..32 { let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32; let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32; let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    y[l] = d * (sc[is] as i8) as f32 * q1 as f32; y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32; y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32; }
                y = &mut y[128..]; ql = &ql[64..]; qh = &qh[32..]; sc = &sc[8..]; } },
        QType::IQ2XXS => { let lt = lut(); for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let qs = &b[2..66]; let mut y = &mut y[i * 256..];
            for ib in 0..8 { let a0 = &qs[8 * ib..8 * ib + 4]; let a1 = u32::from_le_bytes([qs[8 * ib + 4], qs[8 * ib + 5], qs[8 * ib + 6], qs[8 * ib + 7]]);
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for l in 0..4 { put8(y, db, &lt.iq2xxs[a0[l] as usize], &lt.sign8[KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize] as usize]); y = &mut y[8..]; } } } },
        QType::IQ2XS => { let lt = lut(); for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let qs = &b[2..66]; let sc = &b[66..74]; let mut y = &mut y[i * 256..];
            for ib in 0..8 { let db = [d * (0.5 + (sc[ib] & 0xF) as f32) * 0.25, d * (0.5 + (sc[ib] >> 4) as f32) * 0.25];
                for l in 0..4 { let q = u16::from_le_bytes([qs[2 * (4 * ib + l)], qs[2 * (4 * ib + l) + 1]]);
                    put8(y, db[l / 2], &lt.iq2xs[(q & 511) as usize], &lt.sign8[KSIGNS_IQ2XS[(q >> 9) as usize] as usize]); y = &mut y[8..]; } } } },
        QType::IQ2S => { let lt = lut(); for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let mut qs = &b[2..34]; let mut signs = &b[34..66]; let qh = &b[66..74]; let sc = &b[74..82];
            let mut y = &mut y[i * 256..];
            for ib in 0..8 { let db = [d * (0.5 + (sc[ib] & 0xF) as f32) * 0.25, d * (0.5 + (sc[ib] >> 4) as f32) * 0.25];
                for l in 0..4 { let idx = qs[l] as usize | (((qh[ib] as usize) << (8 - 2 * l)) & 0x300);
                    put8(y, db[l / 2], &lt.iq2s[idx], &lt.sign8[signs[l] as usize]); y = &mut y[8..]; }
                qs = &qs[4..]; signs = &signs[4..]; } } },
        QType::IQ3S => { let lt = lut(); for i in 0..nb { let b = &x[i * bs..]; let d = h2f(b, 0); let mut qs = &b[2..66]; let mut qh = &b[66..74]; let mut signs = &b[74..106]; let sc = &b[106..110];
            let mut y = &mut y[i * 256..];
            for ib in (0..8).step_by(2) {
                for half in 0..2 { let s = sc[ib / 2]; let db = d * (1 + 2 * if half == 0 { s & 0xF } else { s >> 4 }) as f32;
                    for l in 0..4 {
                        let g1 = &lt.iq3s[qs[2 * l] as usize | (((qh[half] as usize) << (8 - 2 * l)) & 256)];
                        let g2 = &lt.iq3s[qs[2 * l + 1] as usize | (((qh[half] as usize) << (7 - 2 * l)) & 256)];
                        let g: [f32; 8] = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                        put8(y, db, &g, &lt.sign8[signs[l] as usize]); y = &mut y[8..]; }
                    qs = &qs[8..]; signs = &signs[4..]; }
                qh = &qh[2..]; } } },
        QType::IQ1M => for i in 0..nb { let b = &x[i * bs..]; let mut qs = &b[0..32]; let mut qh = &b[32..48]; let scb = &b[48..56];
            let sc = [u16::from_le_bytes([scb[0], scb[1]]), u16::from_le_bytes([scb[2], scb[3]]), u16::from_le_bytes([scb[4], scb[5]]), u16::from_le_bytes([scb[6], scb[7]])];
            let su = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            let d = f16::from_bits(su).to_f32(); let mut y = &mut y[i * 256..];
            for ib in 0..8 {
                let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as f32; let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as f32;
                let idx = [qs[0] as usize | (((qh[0] as usize) << 8) & 0x700), qs[1] as usize | (((qh[0] as usize) << 4) & 0x700),
                           qs[2] as usize | (((qh[1] as usize) << 8) & 0x700), qs[3] as usize | (((qh[1] as usize) << 4) & 0x700)];
                let dl = [if qh[0] & 0x08 != 0 { -0.125 } else { 0.125 }, if qh[0] & 0x80 != 0 { -0.125 } else { 0.125 }, if qh[1] & 0x08 != 0 { -0.125 } else { 0.125 }, if qh[1] & 0x80 != 0 { -0.125f32 } else { 0.125 }];
                for l in 0..4 { let g = IQ1S_GRID[idx[l]].to_le_bytes(); let s = if l < 2 { dl1 } else { dl2 };
                    for j in 0..8 { y[j] = s * ((g[j] as i8) as f32 + dl[l]); }
                    y = &mut y[8..]; }
                qs = &qs[4..]; qh = &qh[2..]; } },
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { return unsafe { dot_avx2(a, b) }; } }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut s0, mut s1, mut s2, mut s3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), s3);
        i += 32;
    }
    while i + 8 <= n { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0); i += 8; }
    let s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), s);
    let mut r: f32 = t.iter().sum();
    while i < n { r += a[i] * b[i]; i += 1; }
    r
}

/// A quantized row-major matrix view: `rows` rows of `cols` elements, inside an mmap.
#[derive(Clone)]
pub struct QMat { pub ty: QType, pub rows: usize, pub cols: usize, mm: Arc<Mmap>, off: usize }

impl QMat {
    pub fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols) }
    /// The raw bytes of row `r`.
    pub fn row_raw(&self, r: usize) -> &[u8] { let rb = self.row_bytes(); &self.mm[self.off + r * rb..self.off + (r + 1) * rb] }
    /// Identity of the weight data (stable for the mmap's lifetime).
    pub fn key(&self) -> usize { self.mm.as_ptr() as usize + self.off }
    pub fn bytes(&self) -> &[u8] { &self.mm[self.off..self.off + self.rows * self.row_bytes()] }
    pub fn row(&self, r: usize, y: &mut [f32]) { let rb = self.row_bytes(); dequant(self.ty, &self.mm[self.off + r * rb..self.off + (r + 1) * rb], &mut y[..self.cols]); }
    pub fn to_f32(&self) -> Vec<f32> { let mut v = vec![0f32; self.rows * self.cols]; for r in 0..self.rows { self.row(r, &mut v[r * self.cols..]); } v }
    /// Sub-matrix of `n` rows starting at `r0` (e.g. one expert out of a fused [E*out, in] tensor).
    /// Raw bytes of rows r0..r0+n (contiguous).
    pub fn raw_rows(&self, r0: usize, n: usize) -> &[u8] { let rb = self.row_bytes(); &self.mm[self.off + r0 * rb..self.off + (r0 + n) * rb] }
    pub fn slice_rows(&self, r0: usize, n: usize) -> QMat { QMat { ty: self.ty, rows: n, cols: self.cols, mm: self.mm.clone(), off: self.off + r0 * self.row_bytes() } }
    /// y[t*rows + r] = sum_c W[r,c] * x[t*cols + c] for t in 0..T. Rows split across the rayon pool.
    pub fn matmul(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        debug_assert!(x.len() >= t * cols && y.len() >= t * rows);
        if t >= 16 {
            // prompt chunks: dequantize a block of rows once, then a cache-blocked SGEMM against all tokens
            const RB: usize = 64;
            let outs: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(RB)).into_par_iter().map(|ci| {
                let r0 = ci * RB; let nr = (r0 + RB).min(rows) - r0;
                let mut w = vec![0f32; nr * cols]; for r in 0..nr { self.row(r0 + r, &mut w[r * cols..(r + 1) * cols]); }
                let mut o = vec![0f32; t * nr];
                unsafe { matrixmultiply::sgemm(t, cols, nr, 1.0, x.as_ptr(), cols as isize, 1, w.as_ptr(), 1, cols as isize, 0.0, o.as_mut_ptr(), nr as isize, 1); }
                (r0, o)
            }).collect();
            for (r0, o) in outs { let nr = o.len() / t; for ti in 0..t { y[ti * rows + r0..ti * rows + r0 + nr].copy_from_slice(&o[ti * nr..(ti + 1) * nr]); } }
            return;
        }
        let chunk = (rows / (rayon::current_num_threads() * 4)).clamp(4, 256);
        let outs: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(chunk)).into_par_iter().map(|ci| {
            let r0 = ci * chunk; let r1 = (r0 + chunk).min(rows);
            let mut buf = vec![0f32; cols];
            let mut o = vec![0f32; (r1 - r0) * t];
            for r in r0..r1 { self.row(r, &mut buf); for ti in 0..t { o[(r - r0) * t + ti] = dot(&buf, &x[ti * cols..(ti + 1) * cols]); } }
            (r0, o)
        }).collect();
        for (r0, o) in outs { let n = o.len() / t; for i in 0..n { for ti in 0..t { y[ti * rows + r0 + i] = o[i * t + ti]; } } }
    }
    /// Single-threaded matvec (for use inside an outer parallel loop).
    pub fn matvec_st(&self, x: &[f32], y: &mut [f32], buf: &mut Vec<f32>) {
        buf.resize(self.cols, 0.0);
        for r in 0..self.rows { self.row(r, buf); y[r] = dot(buf, x); }
    }
}

#[derive(Debug, Clone)]
pub enum Meta { U(u64), I(i64), F(f64), B(bool), S(String), A(Vec<Meta>) }
impl Meta {
    pub fn as_u64(&self) -> Option<u64> { match self { Meta::U(v) => Some(*v), Meta::I(v) => Some(*v as u64), Meta::B(b) => Some(*b as u64), _ => None } }
    pub fn as_f64(&self) -> Option<f64> { match self { Meta::F(v) => Some(*v), Meta::U(v) => Some(*v as f64), Meta::I(v) => Some(*v as f64), _ => None } }
    pub fn as_str(&self) -> Option<&str> { if let Meta::S(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Meta]> { if let Meta::A(a) = self { Some(a) } else { None } }
}

pub struct TInfo { pub shape: Vec<usize>, pub ty: QType, shard: usize, off: usize }

pub struct Gguf { pub paths: Vec<PathBuf>, maps: Vec<Arc<Mmap>>, pub meta: HashMap<String, Meta>, pub tensors: HashMap<String, TInfo> }

struct Rd<'a> { b: &'a [u8], p: usize }
impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> { anyhow::ensure!(self.p + n <= self.b.len(), "truncated GGUF"); let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> Result<String> { let n = self.u64()? as usize; Ok(String::from_utf8_lossy(self.take(n)?).into_owned()) }
    fn val(&mut self, t: u32) -> Result<Meta> {
        Ok(match t {
            0 => Meta::U(self.take(1)?[0] as u64), 1 => Meta::I(self.take(1)?[0] as i8 as i64),
            2 => Meta::U(u16::from_le_bytes(self.take(2)?.try_into()?) as u64), 3 => Meta::I(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Meta::U(self.u32()? as u64), 5 => Meta::I(self.u32()? as i32 as i64), 6 => Meta::F(f32::from_bits(self.u32()?) as f64),
            7 => Meta::B(self.take(1)?[0] != 0), 8 => Meta::S(self.str()?),
            9 => { let et = self.u32()?; let n = self.u64()? as usize; let mut v = Vec::with_capacity(n); for _ in 0..n { v.push(self.val(et)?); } Meta::A(v) }
            10 => Meta::U(self.u64()?), 11 => Meta::I(self.u64()? as i64), 12 => Meta::F(f64::from_bits(self.u64()?)),
            _ => bail!("bad GGUF value type {t}"),
        })
    }
}

/// All shards of a split GGUF given any one of them (`-NNNNN-of-MMMMM.gguf`), or the single file.
pub fn shard_paths(p: &Path) -> Result<Vec<PathBuf>> {
    let name = p.file_name().and_then(|n| n.to_str()).context("gguf path")?;
    if let Some(i) = name.rfind("-of-") {
        if i >= 6 && name.ends_with(".gguf") {
            let total: usize = name[i + 4..name.len() - 5].parse().context("shard count")?;
            let stem = &name[..i - 6];
            let v: Vec<PathBuf> = (1..=total).map(|k| p.with_file_name(format!("{stem}-{k:05}-of-{total:05}.gguf"))).collect();
            for s in &v { anyhow::ensure!(s.exists(), "missing shard {}", s.display()); }
            return Ok(v);
        }
    }
    Ok(vec![p.to_path_buf()])
}

impl Gguf {
    pub fn open(first: &Path) -> Result<Self> {
        let paths = shard_paths(first)?;
        let mut g = Gguf { paths: paths.clone(), maps: vec![], meta: HashMap::new(), tensors: HashMap::new() };
        for (si, p) in paths.iter().enumerate() {
            let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
            let mm = Arc::new(unsafe { Mmap::map(&f)? });
            let mut r = Rd { b: &mm, p: 0 };
            anyhow::ensure!(r.take(4)? == b"GGUF", "{} is not GGUF", p.display());
            let _ver = r.u32()?; let nt = r.u64()? as usize; let nkv = r.u64()? as usize;
            let mut align = 32usize;
            for _ in 0..nkv { let k = r.str()?; let t = r.u32()?; let v = r.val(t)?; if k == "general.alignment" { align = v.as_u64().unwrap_or(32) as usize; } if si == 0 || !g.meta.contains_key(&k) { g.meta.insert(k, v); } }
            let mut infos = vec![];
            for _ in 0..nt {
                let name = r.str()?; let nd = r.u32()? as usize; let mut shape = vec![]; for _ in 0..nd { shape.push(r.u64()? as usize); }
                let ty = QType::from_id(r.u32()?).with_context(|| name.clone())?; let off = r.u64()? as usize; infos.push((name, shape, ty, off));
            }
            let data = r.p.div_ceil(align) * align;
            for (name, shape, ty, off) in infos { g.tensors.insert(name, TInfo { shape, ty, shard: si, off: data + off }); }
            g.maps.push(mm);
        }
        Ok(g)
    }
    pub fn arch(&self) -> &str { self.meta.get("general.architecture").and_then(|m| m.as_str()).unwrap_or("") }
    pub fn u(&self, k: &str) -> Result<u64> { self.meta.get(k).and_then(|m| m.as_u64()).with_context(|| format!("gguf key {k}")) }
    pub fn f(&self, k: &str) -> Result<f64> { self.meta.get(k).and_then(|m| m.as_f64()).with_context(|| format!("gguf key {k}")) }
    pub fn arr_u64(&self, k: &str) -> Result<Vec<u64>> { Ok(self.meta.get(k).and_then(|m| m.as_arr()).with_context(|| format!("gguf key {k}"))?.iter().filter_map(|m| m.as_u64()).collect()) }
    pub fn has(&self, n: &str) -> bool { self.tensors.contains_key(n) }
    /// A 2-D tensor (or N-D flattened to [prod(shape[1..]), shape[0]]) as a matrix view.
    pub fn mat(&self, n: &str) -> Result<QMat> {
        let t = self.tensors.get(n).with_context(|| format!("missing tensor {n}"))?;
        let cols = t.shape[0]; let rows: usize = t.shape[1..].iter().product::<usize>().max(1);
        let (e, _) = t.ty.geom(); anyhow::ensure!(cols % e == 0, "{n}: row {cols} not a multiple of block {e}");
        Ok(QMat { ty: t.ty, rows, cols, mm: self.maps[t.shard].clone(), off: t.off })
    }
    pub fn shape(&self, n: &str) -> Option<&[usize]> { self.tensors.get(n).map(|t| t.shape.as_slice()) }
    /// A small tensor fully dequantized to f32.
    pub fn vec(&self, n: &str) -> Result<Vec<f32>> { let m = self.mat(n)?; Ok(m.to_f32()) }
    pub fn file_bytes(&self) -> u64 { self.maps.iter().map(|m| m.len() as u64).sum() }
}
