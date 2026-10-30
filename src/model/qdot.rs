//! (AVX-VNNI row copies and the gathered IQ2_S decode follow Strata's iq_avx2 CPU kernels, MIT.)
//! Integer dot products of i-quant / Q2_0 weight rows against Q8-quantized activations, for the
//! routed experts on the CPU. The approach is ggml's `vec_dot_*_q8_K` idea (and Strata's CPU
//! expert pool, which uses ggml's kernels): never expand weights to floats; quantize the activation
//! once per token to int8 blocks of 32, decode each weight row to unsigned grid magnitudes + signs +
//! one scale per 16 values, and let AVX2 `maddubs` do 32 multiply-adds per instruction.
//! This file is pyre's own implementation; the block layouts are ggml's (MIT, see gguf_q.rs).
use super::gguf_q::QType;
use super::iq_tables::*;
use half::f16;

/// One activation vector quantized to int8 blocks of 32 (+ per-8 sums for offset corrections).
pub struct Q8Act { pub q: Vec<i8>, pub d: Vec<f32>, pub s8: Vec<i32> }

pub fn quantize(x: &[f32]) -> Q8Act {
    let n = x.len(); debug_assert!(n % 32 == 0);
    let mut a = Q8Act { q: vec![0; n], d: vec![0.0; n / 32], s8: vec![0; n / 8] };
    quantize_into(x, &mut a); a
}
pub fn quantize_into(x: &[f32], a: &mut Q8Act) {
    let n = x.len(); a.q.resize(n, 0); a.d.resize(n / 32, 0.0); a.s8.resize(n / 8, 0);
    for b in 0..n / 32 {
        let xb = &x[b * 32..(b + 1) * 32];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        a.d[b] = d;
        for j in 0..32 { a.q[b * 32 + j] = (xb[j] * id).round() as i8; }
        for k in 0..4 { a.s8[b * 4 + k] = a.q[b * 32 + 8 * k..b * 32 + 8 * k + 8].iter().map(|&v| v as i32).sum(); }
    }
}

pub fn supports(t: QType) -> bool { matches!(t, QType::IQ2XXS | QType::IQ2S | QType::IQ1M | QType::Q2_0) }

/// Per-thread decode buffers (reused across rows: no allocation per row).
#[derive(Default)]
pub struct Scratch { g: Vec<u8>, s: Vec<u8>, sc: Vec<f32>, c8: Vec<f32> }

fn sign8() -> &'static [u64; 256] {
    static T: std::sync::OnceLock<[u64; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|m| { let mut v = 0u64; for j in 0..8 { v |= (if m & (1 << j) != 0 { 0xFFu64 } else { 0x01 }) << (8 * j); } v }))
}
#[inline] fn h2f(b: &[u8], o: usize) -> f32 { f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32() }
#[inline] fn put(v: &mut [u8], o: usize, w: u64) { v[o..o + 8].copy_from_slice(&w.to_le_bytes()); }

/// sum_c W[c] * x[c] for one quantized row (`row` = the row's bytes, `a.q.len()` columns).
pub fn row_dot(t: QType, row: &[u8], a: &Q8Act, sc: &mut Scratch) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if avx2() { return unsafe { match t { QType::IQ2S => iq2s_avx2(row, a), QType::IQ2XXS => iq2xxs_avx2(row, a), QType::IQ1M => iq1m_avx2(row, a), QType::Q2_0 => q2_0_avx2(row, a), _ => unreachable!() } }; } }
    row_dot_generic(t, row, a, sc)
}

fn avx2() -> bool { static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *A.get_or_init(|| { #[cfg(target_arch = "x86_64")] { is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && std::env::var("PYRE_QN_QDOT_GENERIC").is_err() } #[cfg(not(target_arch = "x86_64"))] { false } }) }

fn q2tab() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|b| { let mut v = 0u32; for j in 0..4 { v |= (((b >> (2 * j)) & 3) as u32) << (8 * j); } v }))
}

#[cfg(target_arch = "x86_64")]
mod k {
    use std::arch::x86_64::*;
    #[inline(always)] pub unsafe fn hsum(v: __m256) -> f32 { let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); let h = _mm_add_ps(h, _mm_movehl_ps(h, h)); let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1)); _mm_cvtss_f32(h) }
    /// acc += (sum of g*sign(s)*x over each 16 bytes) * (lo | hi)
    #[inline(always)] pub unsafe fn step(acc: __m256, g: __m256i, s: __m256i, x: *const i8, lo: f32, hi: f32) -> __m256 {
        let xv = _mm256_loadu_si256(x as *const __m256i);
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(xv, s)), _mm256_set1_epi16(1));
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set_m128(_mm_set1_ps(hi), _mm_set1_ps(lo)), acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2s_avx2(row: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let s8 = sign8(); let mut acc = _mm256_setzero_ps(); let n = a.q.len();
    for i in 0..n / 256 { let b = &row[i * 82..i * 82 + 82]; let d = h2f(b, 0) * 0.25; let (qs, sg, qh, sb) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
        for ib in 0..8 { let o = i * 256 + ib * 32; let h = qh[ib] as usize;
            let q = |l: usize| IQ2S_GRID[qs[4 * ib + l] as usize | ((h << (8 - 2 * l)) & 0x300)] as i64;
            let g = _mm256_set_epi64x(q(3), q(2), q(1), q(0));
            let s = _mm256_set_epi64x(s8[sg[4 * ib + 3] as usize] as i64, s8[sg[4 * ib + 2] as usize] as i64, s8[sg[4 * ib + 1] as usize] as i64, s8[sg[4 * ib] as usize] as i64);
            let dx = d * *a.d.get_unchecked(o / 32);
            acc = k::step(acc, g, s, a.q.as_ptr().add(o), dx * (0.5 + (sb[ib] & 0xF) as f32), dx * (0.5 + (sb[ib] >> 4) as f32)); } }
    k::hsum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2xxs_avx2(row: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let s8 = sign8(); let mut acc = _mm256_setzero_ps(); let n = a.q.len();
    for i in 0..n / 256 { let b = &row[i * 66..i * 66 + 66]; let d = h2f(b, 0) * 0.25; let qs = &b[2..66];
        for ib in 0..8 { let o = i * 256 + ib * 32; let a1 = u32::from_le_bytes([qs[8 * ib + 4], qs[8 * ib + 5], qs[8 * ib + 6], qs[8 * ib + 7]]);
            let g = _mm256_set_epi64x(IQ2XXS_GRID[qs[8 * ib + 3] as usize] as i64, IQ2XXS_GRID[qs[8 * ib + 2] as usize] as i64, IQ2XXS_GRID[qs[8 * ib + 1] as usize] as i64, IQ2XXS_GRID[qs[8 * ib] as usize] as i64);
            let sl = |l: u32| s8[KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize] as usize] as i64;
            let s = _mm256_set_epi64x(sl(3), sl(2), sl(1), sl(0));
            let db = d * (0.5 + (a1 >> 28) as f32) * *a.d.get_unchecked(o / 32);
            acc = k::step(acc, g, s, a.q.as_ptr().add(o), db, db); } }
    k::hsum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq1m_avx2(row: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps(); let mut corr = 0f32; let n = a.q.len(); const M: u64 = 0x0101010101010101;
    for i in 0..n / 256 { let b = &row[i * 56..i * 56 + 56]; let (qs, qh, sb) = (&b[0..32], &b[32..48], &b[48..56]);
        let sc4 = [u16::from_le_bytes([sb[0], sb[1]]), u16::from_le_bytes([sb[2], sb[3]]), u16::from_le_bytes([sb[4], sb[5]]), u16::from_le_bytes([sb[6], sb[7]])];
        let d = f16::from_bits((sc4[0] >> 12) | ((sc4[1] >> 8) & 0x00f0) | ((sc4[2] >> 4) & 0x0f00) | (sc4[3] & 0xf000)).to_f32();
        for ib in 0..8 { let o = i * 256 + ib * 32; let (q4, h2) = (&qs[4 * ib..4 * ib + 4], &qh[2 * ib..2 * ib + 2]);
            let dl = [d * (2 * ((sc4[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as f32, d * (2 * ((sc4[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as f32];
            let w = [IQ1S_GRID[q4[0] as usize | (((h2[0] as usize) << 8) & 0x700)], IQ1S_GRID[q4[1] as usize | (((h2[0] as usize) << 4) & 0x700)],
                     IQ1S_GRID[q4[2] as usize | (((h2[1] as usize) << 8) & 0x700)], IQ1S_GRID[q4[3] as usize | (((h2[1] as usize) << 4) & 0x700)]];
            let g = _mm256_set_epi64x((w[3] & M) as i64, (w[2] & M) as i64, (w[1] & M) as i64, (w[0] & M) as i64);
            let sg = |v: u64| (M | (((v >> 7) & M) * 0xFE)) as i64;
            let s = _mm256_set_epi64x(sg(w[3]), sg(w[2]), sg(w[1]), sg(w[0]));
            let dx = *a.d.get_unchecked(o / 32);
            acc = k::step(acc, g, s, a.q.as_ptr().add(o), dl[0] * dx, dl[1] * dx);
            let del = [if h2[0] & 0x08 != 0 { -0.125 } else { 0.125 }, if h2[0] & 0x80 != 0 { -0.125 } else { 0.125 }, if h2[1] & 0x08 != 0 { -0.125 } else { 0.125 }, if h2[1] & 0x80 != 0 { -0.125f32 } else { 0.125 }];
            let s8 = &a.s8[o / 8..o / 8 + 4];
            corr += dx * (dl[0] * (del[0] * s8[0] as f32 + del[1] * s8[1] as f32) + dl[1] * (del[2] * s8[2] as f32 + del[3] * s8[3] as f32)); } }
    corr + k::hsum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2_0_avx2(row: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let t = q2tab(); let mut acc = _mm256_setzero_ps(); let mut corr = 0f32; let n = a.q.len(); let ones = _mm256_set1_epi8(1);
    for i in 0..n / 64 { let b = &row[i * 18..i * 18 + 18]; let d = h2f(b, 0); let q = &b[2..18];
        for h in 0..2 { let o = i * 64 + h * 32; let qq = &q[8 * h..8 * h + 8];
            let g = _mm256_set_epi32(t[qq[7] as usize] as i32, t[qq[6] as usize] as i32, t[qq[5] as usize] as i32, t[qq[4] as usize] as i32, t[qq[3] as usize] as i32, t[qq[2] as usize] as i32, t[qq[1] as usize] as i32, t[qq[0] as usize] as i32);
            let dx = d * *a.d.get_unchecked(o / 32);
            acc = k::step(acc, g, ones, a.q.as_ptr().add(o), dx, dx);
            let s8 = &a.s8[o / 8..o / 8 + 4]; corr -= dx * (s8[0] + s8[1] + s8[2] + s8[3]) as f32; } }
    corr + k::hsum(acc)
}

fn row_dot_generic(t: QType, row: &[u8], a: &Q8Act, sc: &mut Scratch) -> f32 { decode(t, row, a.q.len(), sc); dot_decoded(sc, a) }

/// One decoded row against an activation (prefill: decode once, dot against every token).
pub fn dot_decoded(sc: &Scratch, a: &Q8Act) -> f32 {
    let mut corr = 0f32; for (gi, &c) in sc.c8.iter().enumerate() { if c != 0.0 { corr += c * a.d[gi / 4] * a.s8[gi] as f32; } }
    corr + dot_i8(&sc.g, &sc.s, &sc.sc, a)
}

/// Decode a row of `n` values into unsigned magnitudes, signs, per-16 scales and per-8 offsets.
pub fn decode(t: QType, row: &[u8], n: usize, sc: &mut Scratch) {
    sc.g.resize(n, 0); sc.s.resize(n, 1); sc.sc.resize(n / 16, 0.0); sc.c8.clear(); sc.c8.resize(n / 8, 0.0);
    let (g, s, scl, c8) = (&mut sc.g[..], &mut sc.s[..], &mut sc.sc[..], &mut sc.c8[..]);
    let s8t = sign8();
    match t {
        QType::IQ2XXS => for i in 0..n / 256 { let b = &row[i * 66..]; let d = h2f(b, 0); let qs = &b[2..66];
            for ib in 0..8 { let a1 = u32::from_le_bytes([qs[8 * ib + 4], qs[8 * ib + 5], qs[8 * ib + 6], qs[8 * ib + 7]]);
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25; let o = i * 256 + ib * 32;
                scl[o / 16] = db; scl[o / 16 + 1] = db;
                for l in 0..4 { put(g, o + 8 * l, IQ2XXS_GRID[qs[8 * ib + l] as usize]); put(s, o + 8 * l, s8t[KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize] as usize]); } } },
        QType::IQ2S => for i in 0..n / 256 { let b = &row[i * 82..]; let d = h2f(b, 0); let qs = &b[2..34]; let sg = &b[34..66]; let qh = &b[66..74]; let sb = &b[74..82];
            for ib in 0..8 { let o = i * 256 + ib * 32;
                scl[o / 16] = d * (0.5 + (sb[ib] & 0xF) as f32) * 0.25; scl[o / 16 + 1] = d * (0.5 + (sb[ib] >> 4) as f32) * 0.25;
                for l in 0..4 { let idx = qs[4 * ib + l] as usize | (((qh[ib] as usize) << (8 - 2 * l)) & 0x300);
                    put(g, o + 8 * l, IQ2S_GRID[idx]); put(s, o + 8 * l, s8t[sg[4 * ib + l] as usize]); } } },
        QType::IQ1M => for i in 0..n / 256 { let b = &row[i * 56..]; let qs = &b[0..32]; let qh = &b[32..48]; let sb = &b[48..56];
            let sc4 = [u16::from_le_bytes([sb[0], sb[1]]), u16::from_le_bytes([sb[2], sb[3]]), u16::from_le_bytes([sb[4], sb[5]]), u16::from_le_bytes([sb[6], sb[7]])];
            let d = f16::from_bits((sc4[0] >> 12) | ((sc4[1] >> 8) & 0x00f0) | ((sc4[2] >> 4) & 0x0f00) | (sc4[3] & 0xf000)).to_f32();
            for ib in 0..8 { let o = i * 256 + ib * 32; let (q4, h2) = (&qs[4 * ib..4 * ib + 4], &qh[2 * ib..2 * ib + 2]);
                let dl = [d * (2 * ((sc4[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as f32, d * (2 * ((sc4[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as f32];
                scl[o / 16] = dl[0]; scl[o / 16 + 1] = dl[1];
                let idx = [q4[0] as usize | (((h2[0] as usize) << 8) & 0x700), q4[1] as usize | (((h2[0] as usize) << 4) & 0x700),
                           q4[2] as usize | (((h2[1] as usize) << 8) & 0x700), q4[3] as usize | (((h2[1] as usize) << 4) & 0x700)];
                let del = [if h2[0] & 0x08 != 0 { -0.125 } else { 0.125 }, if h2[0] & 0x80 != 0 { -0.125 } else { 0.125 }, if h2[1] & 0x08 != 0 { -0.125 } else { 0.125 }, if h2[1] & 0x80 != 0 { -0.125f32 } else { 0.125 }];
                for l in 0..4 {
                    // grid bytes are i8 in {-1,0,1}: magnitude to g, sign to s
                    let w = IQ1S_GRID[idx[l]]; let mag = w & 0x0101010101010101; let neg = (w >> 7) & 0x0101010101010101;
                    put(g, o + 8 * l, mag); put(s, o + 8 * l, 0x0101010101010101 | (neg * 0xFE));
                    c8[(o + 8 * l) / 8] = dl[l / 2] * del[l];
                } } },
        QType::Q2_0 => for i in 0..n / 64 { let b = &row[i * 18..]; let d = h2f(b, 0); let o = i * 64;
            for k in 0..16 { let q = b[2 + k]; for j in 0..4 { g[o + 4 * k + j] = (q >> (2 * j)) & 3; } }
            for k in 0..4 { scl[o / 16 + k] = d; }
            s[o..o + 64].fill(1);
            // value = q - 1: subtract d * sum(x) over the block
            c8[o / 8..o / 8 + 8].fill(-d); },
        _ => unreachable!("qdot: unsupported {t:?}"),
    }
}

fn dot_i8(g: &[u8], s: &[u8], scl: &[f32], a: &Q8Act) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { return unsafe { dot_i8_avx2(g, s, scl, a) }; } }
    let mut r = 0f32;
    for b in 0..a.q.len() / 32 { for h in 0..2 { let o = b * 32 + h * 16; let mut acc = 0i32;
        for j in 0..16 { acc += g[o + j] as i32 * (s[o + j] as i8) as i32 * a.q[o + j] as i32; }
        r += acc as f32 * scl[o / 16] * a.d[b]; } }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_avx2(g: &[u8], s: &[u8], scl: &[f32], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1); let mut acc = _mm256_setzero_ps();
    let (pg, ps, pq) = (g.as_ptr(), s.as_ptr(), a.q.as_ptr());
    for b in 0..a.q.len() / 32 {
        let gv = _mm256_loadu_si256(pg.add(b * 32) as *const __m256i);
        let sv = _mm256_loadu_si256(ps.add(b * 32) as *const __m256i);
        let xv = _mm256_loadu_si256(pq.add(b * 32) as *const __m256i);
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(gv, _mm256_sign_epi8(xv, sv)), ones);
        let dx = *a.d.get_unchecked(b);
        let sc = _mm256_set_m128(_mm_set1_ps(*scl.get_unchecked(2 * b + 1) * dx), _mm_set1_ps(*scl.get_unchecked(2 * b) * dx));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sc, acc);
    }
    let mut t = [0f32; 8]; _mm256_storeu_ps(t.as_mut_ptr(), acc); t.iter().sum()
}

/// An owned Q8_0 matrix (ggml block layout: f16 scale + 32 x i8), for small dense weights the CPU
/// multiplies every token (the hyper-connection projections, stored as BF16 in the GGUF).
pub struct Q8Mat { pub rows: usize, pub cols: usize, data: Vec<u8>, sc: Vec<f32> }

impl Q8Mat {
    /// The Q8_0 blocks (34 bytes: f16 scale, 32 i8) row after row.
    pub fn blocks(&self) -> &[u8] { &self.data }
    pub fn from_rows(ms: &[&super::gguf_q::QMat]) -> Self {
        use rayon::prelude::*;
        let cols = ms[0].cols; let nb = cols / 32; let rb = nb * 34;
        let rows: Vec<(&super::gguf_q::QMat, usize)> = ms.iter().flat_map(|m| (0..m.rows).map(move |r| (*m, r))).collect();
        let mut data = vec![0u8; rows.len() * rb];
        data.par_chunks_mut(rb).zip(rows.par_iter()).for_each_init(|| vec![0f32; cols], |buf, (o, &(m, r))| {
            m.row(r, buf);
            for b in 0..nb { let x = &buf[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                o[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                for j in 0..32 { o[b * 34 + 2 + j] = ((x[j] * id).round().clamp(-127.0, 127.0) as i8) as u8; } }
        });
        let sc = (0..data.len() / 34).map(|b| h2f(&data, b * 34)).collect();
        Q8Mat { rows: rows.len(), cols, data, sc }
    }
    /// Same quantization from rows produced by `f(row, out)` (the MTP drafter's BF16 weights).
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, &mut [f32]) + Sync) -> Self {
        use rayon::prelude::*;
        let nb = cols / 32; let rb = nb * 34;
        let mut data = vec![0u8; rows * rb];
        data.par_chunks_mut(rb).enumerate().for_each_init(|| vec![0f32; cols], |buf, (r, o)| {
            f(r, buf);
            for b in 0..nb { let x = &buf[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                o[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                for j in 0..32 { o[b * 34 + 2 + j] = ((x[j] * id).round().clamp(-127.0, 127.0) as i8) as u8; } }
        });
        let sc = (0..data.len() / 34).map(|b| h2f(&data, b * 34)).collect();
        Q8Mat { rows, cols, data, sc }
    }
    pub fn bytes(&self) -> usize { self.data.len() }
    /// y[t][rows] = x[t][cols] W^T with int8 dots; rows split across the rayon pool.
    pub fn matmul(&self, x: &[f32], t: usize, y: &mut [f32]) {
        use rayon::prelude::*;
        let (rows, cols) = (self.rows, self.cols); let rb = cols / 32 * 34;
        let acts: Vec<Q8Act> = (0..t).map(|ti| quantize(&x[ti * cols..(ti + 1) * cols])).collect();
        let ch = (rows / (rayon::current_num_threads() * 2)).max(8);
        let outs: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(ch)).into_par_iter().map(|ci| {
            let r0 = ci * ch; let r1 = (r0 + ch).min(rows); let mut o = vec![0f32; (r1 - r0) * t];
            for r in r0..r1 { let row = &self.data[r * rb..(r + 1) * rb]; for ti in 0..t { o[ti * (r1 - r0) + r - r0] = dot_q8(row, &self.sc[r * (cols / 32)..(r + 1) * (cols / 32)], &acts[ti]); } }
            (r0, o)
        }).collect();
        for (r0, o) in outs { let n = o.len() / t; for ti in 0..t { y[ti * rows + r0..ti * rows + r0 + n].copy_from_slice(&o[ti * n..(ti + 1) * n]); } }
    }
}

fn dot_q8(row: &[u8], sc: &[f32], a: &Q8Act) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if avx2() { return unsafe { dot_q8_avx2(row, sc, a) }; } }
    let mut r = 0f32;
    for b in 0..a.q.len() / 32 { let d = sc[b]; let mut s = 0i32; for j in 0..32 { s += (row[b * 34 + 2 + j] as i8) as i32 * a.q[b * 32 + j] as i32; } r += s as f32 * d * a.d[b]; }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_avx2(row: &[u8], sc: &[f32], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps(); let ones = _mm256_set1_epi16(1);
    for b in 0..a.q.len() / 32 {
        let w = _mm256_loadu_si256(row.as_ptr().add(b * 34 + 2) as *const __m256i);
        let x = _mm256_loadu_si256(a.q.as_ptr().add(b * 32) as *const __m256i);
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(*sc.get_unchecked(b) * *a.d.get_unchecked(b)), acc);
    }
    k::hsum(acc)
}

// ---- Q8_K-style activations (one scale per 256) with integer accumulation per super-block ----
// Same idea as ggml's q8_K + vec_dot_iq2_s_q8_K: the per-16/32 weight scales are small odd integers
// (2s+1), so they can be applied with `madd_epi16` and the whole 256-block summed in int32; only one
// float multiply-add per 256 values remains. pyre's own implementation.
pub struct Q8K { pub q: Vec<i8>, pub d: Vec<f32>, pub s8: Vec<i32> }
impl Default for Q8K { fn default() -> Self { Q8K { q: vec![], d: vec![], s8: vec![] } } }
pub fn quantize_k_into(x: &[f32], a: &mut Q8K) {
    let n = x.len(); a.q.resize(n, 0); a.d.resize(n / 256, 0.0); a.s8.resize(n / 8, 0);
    for b in 0..n / 256 {
        let xb = &x[b * 256..(b + 1) * 256];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 }; a.d[b] = d;
        for j in 0..256 { a.q[b * 256 + j] = (xb[j] * id).round() as i8; }
    }
    for k in 0..n / 8 { a.s8[k] = a.q[8 * k..8 * k + 8].iter().map(|&v| v as i32).sum(); }
}
/// AVX-VNNI present (Alder Lake and later) and not disabled with PYRE_QN_VNNI=0.
/// PYRE_QN_GATHER=1: IQ2_S grid lookups with vpgatherdq (opt-in until measured).
/// PYRE_QN_GATHER: IQ2_S grid words with one vpgatherqq (1), scalar loads (0), or "auto": gathered only on threads
/// running on a hybrid CPU's performance core (CPUID leaf 1Ah core type 40h), scalar on E-cores, whose gathers are
/// slow (Strata, MIT, github.com/Niko1221/Strata: cpu_gather_fast_here, src/kernels/cpu/expert_layout.cpp). Same
/// grid words either way (bit-identical). Per thread, decided on first use (pool workers are pinned).
pub fn gather() -> bool {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let m = *M.get_or_init(|| match std::env::var("PYRE_QN_GATHER").as_deref() { Ok("1") => 1, Ok("auto") => 2, _ => 0 });
    if m < 2 { return m == 1; }
    thread_local! { static P: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
    P.with(|p| { if p.get() == 0 { p.set(if pcore_here() { 2 } else { 1 }); } p.get() == 2 })
}
/// The calling thread runs on a performance core of a hybrid Intel CPU (CPUID 7.EDX[15] hybrid, 1Ah EAX[31:24] = 40h).
fn pcore_here() -> bool {
    #[cfg(target_arch = "x86_64")] unsafe { use std::arch::x86_64::__cpuid_count;
        if __cpuid_count(0, 0).eax < 0x1A || (__cpuid_count(7, 0).edx >> 15) & 1 == 0 { return false; }
        return (__cpuid_count(0x1A, 0).eax >> 24) == 0x40; }
    #[allow(unreachable_code)] false
}
pub fn vnni() -> bool { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *V.get_or_init(|| { #[cfg(target_arch = "x86_64")] { std::env::var("PYRE_QN_VNNI").map(|v| v != "0").unwrap_or(true) && is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") } #[cfg(not(target_arch = "x86_64"))] { false } }) }
pub fn supports_k(t: QType) -> bool { matches!(t, QType::IQ2XXS | QType::IQ2S | QType::IQ1M) }
// ---- gate+up pair: one IQ2 super-block of each row per step, sharing the activation; each row's own
// operations are the same, in the same order, as iq2s_k_b / iq2xxs_k_b (bit-identical results)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2_sb<const V: bool>(t: QType, rp: *const u8, i: usize, a: &Q8K, acc: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let xq = a.q.as_ptr();
    if t == QType::IQ2S {
        let b = rp.add(i * 82); kk::pf(b); let d = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() * 0.125 * *a.d.get_unchecked(i);
        let (qs, sg, qh, sb) = (b.add(2), b.add(34), b.add(66), b.add(74));
        let mut ia = _mm256_setzero_si256();
        for ib in 0..8 { let h = *qh.add(ib) as usize;
            let q = |l: usize| *IQ2S_GRID.get_unchecked(*qs.add(4 * ib + l) as usize | ((h << (8 - 2 * l)) & 0x300)) as i64;
            let g = if gather() { kk::g2s(qs.add(4 * ib), h) } else if gidx() { kk::g2i(qs.add(4 * ib), h) } else { _mm256_set_epi64x(q(3), q(2), q(1), q(0)) };
            let s = kk::signs((sg.add(4 * ib) as *const u32).read_unaligned());
            let sbv = *sb.add(ib);
            ia = kk::istep::<V>(ia, g, s, xq.add(i * 256 + ib * 32), kk::sc16(sbv)); }
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_set1_ps(d), acc)
    } else {
        let s8 = sign8(); let (es, eo) = (esigns(), esign_on());
        let b = rp.add(i * 66); kk::pf(b); let d = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() * 0.125 * *a.d.get_unchecked(i); let qs = b.add(2);
        let mut ia = _mm256_setzero_si256();
        for ib in 0..8 { let a1 = (qs.add(8 * ib + 4) as *const u32).read_unaligned();
            let gq = |l: usize| *IQ2XXS_GRID.get_unchecked(*qs.add(8 * ib + l) as usize) as i64;
            let g = _mm256_set_epi64x(gq(3), gq(2), gq(1), gq(0));
            let sl = |l: u32| if eo { *es.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as i64 } else { *s8.get_unchecked(*KSIGNS_IQ2XS.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as usize) as i64 };
            let s = _mm256_set_epi64x(sl(3), sl(2), sl(1), sl(0));
            let sc = 2 * (a1 >> 28) as i32 + 1;
            ia = kk::istep::<V>(ia, g, s, xq.add(i * 256 + ib * 32), kk::sc2(sc, sc)); }
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_set1_ps(d), acc)
    }
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2_pair_b<const V: bool>(t: QType, ra: &[u8], rb: &[u8], a: &Q8K) -> (f32, f32) {
    use std::arch::x86_64::*;
    let (mut x, mut y) = (_mm256_setzero_ps(), _mm256_setzero_ps()); let n = a.q.len();
    for i in 0..n / 256 { x = iq2_sb::<V>(t, ra.as_ptr(), i, a, x); y = iq2_sb::<V>(t, rb.as_ptr(), i, a, y); }
    (k::hsum(x), k::hsum(y))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2_pair(t: QType, ra: &[u8], rb: &[u8], a: &Q8K) -> (f32, f32) { iq2_pair_b::<false>(t, ra, rb, a) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq2_pair_v(t: QType, ra: &[u8], rb: &[u8], a: &Q8K) -> (f32, f32) { iq2_pair_b::<true>(t, ra, rb, a) }
/// Two rows of the same IQ2_S / IQ2_XXS type against one activation, interleaved per super-block;
/// the same results as two row_dot_k calls. None when the type or CPU is not covered.
pub fn row_dot_k_pair(t: QType, ra: &[u8], rb: &[u8], a: &Q8K) -> Option<(f32, f32)> {
    #[cfg(target_arch = "x86_64")]
    { if matches!(t, QType::IQ2S | QType::IQ2XXS) { if vnni() { return Some(unsafe { iq2_pair_v(t, ra, rb, a) }); } if avx2() { return Some(unsafe { iq2_pair(t, ra, rb, a) }); } } }
    let _ = (t, ra, rb, a); None
}

/// IQ2_XXS sign masks indexed directly by the 7-bit sign index: sign8()[KSIGNS_IQ2XS[i]] precomputed (Strata's
/// even_signs table, MIT, github.com/Niko1221/Strata src/kernels/cpu/iq_avx2.cpp): one load per 8 values instead of two
/// dependent ones; the same 64-bit words, so bit-identical. PYRE_QN_ESIGN=0 restores the double lookup.
pub fn esigns() -> &'static [u64; 128] { static T: std::sync::OnceLock<[u64; 128]> = std::sync::OnceLock::new(); T.get_or_init(|| { let s8 = sign8(); std::array::from_fn(|i| s8[KSIGNS_IQ2XS[i] as usize]) }) }
pub fn esign_on() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_ESIGN").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_ISPLIT: independent integer accumulator chains per token in the IQ2 row kernels (4 for one token, 2 for
/// two; Strata's EX accumulators, iq_avx2_rows.inl): shorter vpdpwssd/add dependency chains, the same exact i32 sums
/// (no overflow: |sum| < 2^22), so bit-identical.
pub fn isplit() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_ISPLIT").map(|v| v != "0").unwrap_or(false)) }

pub fn row_dot_k(t: QType, row: &[u8], a: &Q8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if vnni() { return unsafe { match t { QType::IQ2S => iq2s_k_v(row, a), QType::IQ2XXS => iq2xxs_k_v(row, a), QType::IQ1M => iq1m_k_v(row, a), _ => unreachable!() } }; }
      if avx2() { return unsafe { match t { QType::IQ2S => iq2s_k(row, a), QType::IQ2XXS => iq2xxs_k(row, a), QType::IQ1M => iq1m_k(row, a), _ => unreachable!() } }; } }
    // scalar fallback through the float decoder
    let mut v = vec![0f32; a.q.len()]; super::gguf_q::dequant(t, row, &mut v);
    let mut s = 0f32; for b in 0..a.d.len() { let mut si = 0f32; for j in 0..256 { si += v[b * 256 + j] * a.q[b * 256 + j] as f32; } s += si * a.d[b]; } s
}

#[cfg(target_arch = "x86_64")]
mod kk {
    use std::arch::x86_64::*;
    /// int32 lanes += sum over 4-byte groups of g*sign(s)*x, weighted by (lo for bytes 0..16, hi for 16..32)
    /// V: AVX-VNNI `vpdpwssd` does the madd and the add in one uop (the same integer sums; maddubs cannot
    /// saturate here since |grid| <= 43). Using AVX-VNNI for the i-quant rows follows Strata's iq_avx2 kernels.
    #[inline(always)] pub unsafe fn istep<const V: bool>(acc: __m256i, g: __m256i, s: __m256i, x: *const i8, sc: __m256i) -> __m256i {
        let xv = _mm256_loadu_si256(x as *const __m256i);
        let p = _mm256_maddubs_epi16(g, _mm256_sign_epi8(xv, s));
        if V { _mm256_dpwssd_avx_epi32(acc, p, sc) } else { _mm256_add_epi32(acc, _mm256_madd_epi16(p, sc)) }
    }
    /// IQ2_S grid words for one 32-value chunk with one vpgatherdq (the indices built in a register: Strata's
    /// gathered decode, iq_avx2.cpp). Same words in the same lanes as the scalar loads.
    #[inline(always)] pub unsafe fn g2s(qs: *const u8, h: usize) -> __m256i {
        let q = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((qs as *const i32).read_unaligned()));
        let hi = _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(h as i32), _mm_setr_epi32(8, 6, 4, 2)), _mm_set1_epi32(0x300));
        _mm256_i32gather_epi64::<8>(super::IQ2S_GRID.as_ptr() as *const i64, _mm_or_si128(q, hi))
    }
    /// IQ2_S scale byte -> the sc2 vector (lanes 0-7 = 2*(b&15)+1, 8-15 = 2*(b>>4)+1) by one load from a const table
    /// (Strata, MIT: scale_vecs.s16, src/kernels/cpu/iq_avx2.cpp); the same lanes as sc2, so bit-identical.
    #[inline(always)] pub unsafe fn sc16(b: u8) -> __m256i { if super::sc16_tab() { _mm256_loadu_si256(super::SC16.as_ptr().add(b as usize) as *const __m256i) } else { sc2(2 * (b & 0xF) as i32 + 1, 2 * (b >> 4) as i32 + 1) } }
    /// IQ2_S grid words via 16-bit indices built with one punpcklbw: the 8 low index bytes interleaved with the qh
    /// byte's 2-bit groups spread one per byte (HI2S); only lanes 0-3 are used. Same indices as the scalar shift/mask/or
    /// (bit-identical). Strata (MIT, github.com/Niko1221/Strata, src/kernels/cpu/iq_avx2.cpp: hi_spread + grid_indices).
    /// Reads 8 bytes at qs (stays inside the 82-byte block for every chunk).
    #[inline(always)] pub unsafe fn g2i(qs: *const u8, h: usize) -> __m256i {
        let v = _mm_unpacklo_epi8(_mm_cvtsi64_si128((qs as *const i64).read_unaligned()), _mm_cvtsi64_si128(*super::HI2S.get_unchecked(h) as i64));
        let mut sp = [0u16; 8]; _mm_storeu_si128(sp.as_mut_ptr() as *mut __m128i, v);
        let gq = |l: usize| *super::IQ2S_GRID.get_unchecked(sp[l] as usize) as i64;
        _mm256_set_epi64x(gq(3), gq(2), gq(1), gq(0))
    }
    /// Prefetch PYRE_QN_PFB bytes (default see pfb()) ahead of an IQ2 super-block, two lines (Strata rows_ahead, MIT).
    #[inline(always)] pub unsafe fn pf(b: *const u8) { let d = super::pfb(); if d > 0 { _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(d) as *const i8); _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(d + 64) as *const i8); } }
    #[inline(always)] pub unsafe fn sc2(lo: i32, hi: i32) -> __m256i { _mm256_set_m128i(_mm_set1_epi16(hi as i16), _mm_set1_epi16(lo as i16)) }
    /// 32 sign bytes (+1/-1) from 32 bits
    #[inline(always)] pub unsafe fn signs(bits: u32) -> __m256i {
        let sv = _mm256_set1_epi32(bits as i32);
        let sh = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
        let bm = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
        let m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sv, sh), bm), bm);
        _mm256_or_si256(m, _mm256_set1_epi8(1))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2s_k(row: &[u8], a: &Q8K) -> f32 { if isplit() { iq2s_k_b::<false, true>(row, a) } else { iq2s_k_b::<false, false>(row, a) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq2s_k_v(row: &[u8], a: &Q8K) -> f32 { if isplit() { iq2s_k_b::<true, true>(row, a) } else { iq2s_k_b::<true, false>(row, a) } }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2s_k_b<const V: bool, const S: bool>(row: &[u8], a: &Q8K) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps(); let n = a.q.len(); let rp = row.as_ptr(); let xq = a.q.as_ptr();
    for i in 0..n / 256 { let b = rp.add(i * 82); kk::pf(b); let d = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() * 0.125 * *a.d.get_unchecked(i);
        let (qs, sg, qh, sb) = (b.add(2), b.add(34), b.add(66), b.add(74));
        let mut ia = [_mm256_setzero_si256(); 4];
        for ib in 0..8 { let h = *qh.add(ib) as usize;
            let q = |l: usize| *IQ2S_GRID.get_unchecked(*qs.add(4 * ib + l) as usize | ((h << (8 - 2 * l)) & 0x300)) as i64;
            let g = if gather() { kk::g2s(qs.add(4 * ib), h) } else if gidx() { kk::g2i(qs.add(4 * ib), h) } else { _mm256_set_epi64x(q(3), q(2), q(1), q(0)) };
            let s = kk::signs((sg.add(4 * ib) as *const u32).read_unaligned());
            let sbv = *sb.add(ib);
            let c = if S { ib & 3 } else { 0 }; ia[c] = kk::istep::<V>(ia[c], g, s, xq.add(i * 256 + ib * 32), kk::sc16(sbv)); }
        let ia = if S { _mm256_add_epi32(_mm256_add_epi32(ia[0], ia[1]), _mm256_add_epi32(ia[2], ia[3])) } else { ia[0] };
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_set1_ps(d), acc); }
    k::hsum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2xxs_k(row: &[u8], a: &Q8K) -> f32 { if isplit() { iq2xxs_k_b::<false, true>(row, a) } else { iq2xxs_k_b::<false, false>(row, a) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq2xxs_k_v(row: &[u8], a: &Q8K) -> f32 { if isplit() { iq2xxs_k_b::<true, true>(row, a) } else { iq2xxs_k_b::<true, false>(row, a) } }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2xxs_k_b<const V: bool, const S: bool>(row: &[u8], a: &Q8K) -> f32 {
    use std::arch::x86_64::*;
    let s8 = sign8(); let (es, eo) = (esigns(), esign_on()); let mut acc = _mm256_setzero_ps(); let n = a.q.len(); let rp = row.as_ptr(); let xq = a.q.as_ptr();
    for i in 0..n / 256 { let b = rp.add(i * 66); kk::pf(b); let d = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() * 0.125 * *a.d.get_unchecked(i); let qs = b.add(2);
        let mut ia = [_mm256_setzero_si256(); 4];
        for ib in 0..8 { let a1 = (qs.add(8 * ib + 4) as *const u32).read_unaligned();
            let gq = |l: usize| *IQ2XXS_GRID.get_unchecked(*qs.add(8 * ib + l) as usize) as i64;
            let g = _mm256_set_epi64x(gq(3), gq(2), gq(1), gq(0));
            let sl = |l: u32| if eo { *es.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as i64 } else { *s8.get_unchecked(*KSIGNS_IQ2XS.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as usize) as i64 };
            let s = _mm256_set_epi64x(sl(3), sl(2), sl(1), sl(0));
            let sc = 2 * (a1 >> 28) as i32 + 1;
            let c = if S { ib & 3 } else { 0 }; ia[c] = kk::istep::<V>(ia[c], g, s, xq.add(i * 256 + ib * 32), kk::sc2(sc, sc)); }
        let ia = if S { _mm256_add_epi32(_mm256_add_epi32(ia[0], ia[1]), _mm256_add_epi32(ia[2], ia[3])) } else { ia[0] };
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_set1_ps(d), acc); }
    k::hsum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq1m_k(row: &[u8], a: &Q8K) -> f32 { iq1m_k_b::<false>(row, a) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq1m_k_v(row: &[u8], a: &Q8K) -> f32 { iq1m_k_b::<true>(row, a) }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq1m_k_b<const V: bool>(row: &[u8], a: &Q8K) -> f32 {
    use std::arch::x86_64::*;
    let mut tot = 0f32; let n = a.q.len(); const M: u64 = 0x0101010101010101; let xq = a.q.as_ptr();
    for i in 0..n / 256 { let b = &row[i * 56..i * 56 + 56]; let (qs, qh, sb) = (&b[0..32], &b[32..48], &b[48..56]);
        let sc4 = [u16::from_le_bytes([sb[0], sb[1]]), u16::from_le_bytes([sb[2], sb[3]]), u16::from_le_bytes([sb[4], sb[5]]), u16::from_le_bytes([sb[6], sb[7]])];
        let d = f16::from_bits((sc4[0] >> 12) | ((sc4[1] >> 8) & 0x00f0) | ((sc4[2] >> 4) & 0x0f00) | (sc4[3] & 0xf000)).to_f32() * *a.d.get_unchecked(i);
        let mut ia = _mm256_setzero_si256(); let mut corr = 0i32;
        for ib in 0..8 { let o = i * 256 + ib * 32; let (q4, h2) = (&qs[4 * ib..4 * ib + 4], &qh[2 * ib..2 * ib + 2]);
            let dl = [(2 * ((sc4[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as i32, (2 * ((sc4[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as i32];
            let w = [IQ1S_GRID[q4[0] as usize | (((h2[0] as usize) << 8) & 0x700)], IQ1S_GRID[q4[1] as usize | (((h2[0] as usize) << 4) & 0x700)],
                     IQ1S_GRID[q4[2] as usize | (((h2[1] as usize) << 8) & 0x700)], IQ1S_GRID[q4[3] as usize | (((h2[1] as usize) << 4) & 0x700)]];
            let g = _mm256_set_epi64x((w[3] & M) as i64, (w[2] & M) as i64, (w[1] & M) as i64, (w[0] & M) as i64);
            let sg = |v: u64| (M | (((v >> 7) & M) * 0xFE)) as i64;
            let s = _mm256_set_epi64x(sg(w[3]), sg(w[2]), sg(w[1]), sg(w[0]));
            ia = kk::istep::<V>(ia, g, s, xq.add(o), kk::sc2(dl[0], dl[1]));
            let del = |m: u8| if m != 0 { -1i32 } else { 1 };
            let s8 = &a.s8[o / 8..o / 8 + 4];
            corr += dl[0] * (del(h2[0] & 0x08) * s8[0] + del(h2[0] & 0x80) * s8[1]) + dl[1] * (del(h2[1] & 0x08) * s8[2] + del(h2[1] & 0x80) * s8[3]); }
        // values are dl*(grid + delta), delta = +-1/8: integer part + corr/8
        let mut l = [0i32; 8]; _mm256_storeu_si256(l.as_mut_ptr() as *mut __m256i, ia);
        tot += d * (l.iter().sum::<i32>() as f32 + 0.125 * corr as f32); }
    tot
}

// ---- Q2_0 with activations quantized per 64 (the weight block) and permuted to the weight's
// bit-plane order: byte j of a block holds elements 4j..4j+3 in bit pairs, so (q >> 2k) & 3 over
// all 16 bytes gives elements 4j+k for j = 0..15; storing x in that order turns the unpack into
// two shifts and one AND per 32 values (no table lookups), with int32 accumulation per block.
pub struct Q8D { pub q: Vec<i8>, pub d: Vec<f32>, pub s: Vec<i32> }
impl Default for Q8D { fn default() -> Self { Q8D { q: vec![], d: vec![], s: vec![] } } }
pub fn quantize_d_into(x: &[f32], a: &mut Q8D) {
    let n = x.len(); a.q.resize(n, 0); a.d.resize(n / 64, 0.0); a.s.resize(n / 64, 0);
    for b in 0..n / 64 {
        let xb = &x[b * 64..(b + 1) * 64];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 }; a.d[b] = d;
        let mut s = 0i32;
        for j in 0..16 { for k in 0..4 { let v = (xb[4 * j + k] * id).round() as i8; a.q[b * 64 + 16 * k + j] = v; s += v as i32; } }
        a.s[b] = s;
    }
}
pub fn q2_0_dot_d(row: &[u8], a: &Q8D) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if avx2() { return unsafe { q2_0_d_avx2(row, a) }; } }
    let mut s = 0f32;
    for i in 0..a.d.len() { let b = &row[i * 18..i * 18 + 18]; let d = h2f(b, 0); let mut is = 0i32;
        for j in 0..16 { for k in 0..4 { is += (((b[2 + j] >> (2 * k)) & 3) as i32) * a.q[i * 64 + 16 * k + j] as i32; } }
        s += d * a.d[i] * (is - a.s[i]) as f32; }
    s
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2_0_d_avx2(row: &[u8], a: &Q8D) -> f32 { if dvnni() { q2_0_d_v(row, a) } else { q2_0_d_b::<false>(row, a) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q2_0_d_v(row: &[u8], a: &Q8D) -> f32 { q2_0_d_b::<true>(row, a) }
/// PYRE_QN_DVNNI: Q2_0 down dots with AVX-VNNI vpdpbusd (u2 weights x s8 activations, 4 products per i32 lane);
/// the same exact i32 sums as maddubs+madd(1) (no saturation: 2*3*127 < 32767), so bit-identical.
/// PYRE_QN_DVNNI=p: VNNI down dots only on threads running on a P-core (E-cores keep maddubs; same sums, bit-identical).
pub fn dvnni() -> bool {
    static P: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let m = *P.get_or_init(|| if !vnni() { 0 } else { match std::env::var("PYRE_QN_DVNNI").as_deref() { Ok("p") | Ok("P") => 2, Ok("0") | Err(_) => 0, Ok(_) => 1 } });
    if m < 2 { return m == 1; }
    thread_local! { static C: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
    C.with(|c| { if c.get() == 0 { c.set(if pcore_here() { 2 } else { 1 }); } c.get() == 2 })
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q2_0_d_b<const V: bool>(row: &[u8], a: &Q8D) -> f32 {
    use std::arch::x86_64::*;
    let m3 = _mm256_set1_epi8(3); let one = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps(); let mut corr = 0f32; let rp = row.as_ptr(); let xq = a.q.as_ptr();
    for i in 0..a.d.len() { let b = rp.add(i * 18);
        let dd = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() * *a.d.get_unchecked(i);
        let qv = _mm_loadu_si128(b.add(2) as *const __m128i);
        let lo = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 2), qv), m3);
        let hi = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 6), _mm_srli_epi16(qv, 4)), m3);
        let x0 = _mm256_loadu_si256(xq.add(i * 64) as *const __m256i); let x1 = _mm256_loadu_si256(xq.add(i * 64 + 32) as *const __m256i);
        let p = if V { _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), lo, x0), hi, x1) } else { _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), one), _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), one)) };
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dd), acc);
        corr -= dd * *a.s.get_unchecked(i) as f32; }
    corr + k::hsum(acc)
}

/// PYRE_QN_KBENCH: CPU IQ2_S row-dot speed, one expert (cache-hot) vs many (streamed), 1 and 8 threads.
pub fn kbench() {
    let (d, rows) = (2560usize, 1280usize); let rb = d / 256 * 82;
    let mut x = vec![0f32; d]; for i in 0..d { x[i] = ((i * 7919 % 1000) as f32 - 500.0) / 300.0; }
    let mut a = Q8K::default(); quantize_k_into(&x, &mut a);
    let mk = |ne: usize| { let mut v = vec![0u8; ne * rows * rb]; let mut s = 0x9e3779b97f4a7c15u64;
        for b in v.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; *b = s as u8; }
        for r in 0..ne * rows * d / 256 { v[r * 82] = 0x00; v[r * 82 + 1] = 0x1c; } v };
    let pool = super::spin_pool::Pool::new(8, std::time::Duration::from_micros(2000));
    { let w = mk(8); let mut hsh = 0u64; for r in 0..8 * rows { hsh = hsh.rotate_left(5) ^ row_dot_k(QType::IQ2S, &w[r * rb..(r + 1) * rb], &a).to_bits() as u64; } eprintln!("kbench checksum {hsh:016x}"); }
    { let (ff, dr) = (512usize, 2560usize); let rbq = ff / 64 * 18; let mut hx = vec![0f32; ff]; for i in 0..ff { hx[i] = ((i * 7919 % 1000) as f32 - 500.0) / 300.0; }
      let mut ad = Q8D::default(); quantize_d_into(&hx, &mut ad); let ne = 300; let mut w = vec![0u8; ne * dr * rbq]; let mut sd = 0x9e3779b97f4a7c15u64;
      for b in w.iter_mut() { sd ^= sd << 13; sd ^= sd >> 7; sd ^= sd << 17; *b = sd as u8; } for r in 0..ne * dr * ff / 64 { w[r * 18] = 0x00; w[r * 18 + 1] = 0x1c; }
      let nrow = ne * dr; let mut hsh = 0u64; let mut hb = 0u64;
      for r in 0..nrow { let v = unsafe { q2_0_d_b::<false>(&w[r * rbq..(r + 1) * rbq], &ad) }; hb = hb.rotate_left(5) ^ v.to_bits() as u64; hsh = hsh.rotate_left(5) ^ q2_0_dot_d(&w[r * rbq..(r + 1) * rbq], &ad).to_bits() as u64; }
      let t0 = std::time::Instant::now(); let mut sk = 0f32; for _ in 0..3 { for r in 0..nrow { sk += q2_0_dot_d(&w[r * rbq..(r + 1) * rbq], &ad); } }
      eprintln!("kbench Q2_0 down checksum {hsh:016x} base {hb:016x} identical={} dvnni={} {:.2} ns/row (sink {})", hsh == hb, dvnni(), t0.elapsed().as_secs_f64() * 1e9 / (3 * nrow) as f64, sk.to_bits() & 1); }
    for &(ne, label) in &[(1usize, "hot 1 expert"), (300, "stream 300 experts")] {
        let w = mk(ne); let nrow = ne * rows; let reps = if ne == 1 { 200 } else { 2 };
        for &th in &[1usize, 8] {
            let t0 = std::time::Instant::now(); let sink = std::sync::atomic::AtomicU32::new(0);
            for _ in 0..reps {
                if th == 1 { let mut s = 0f32; for r in 0..nrow { s += row_dot_k(QType::IQ2S, &w[r * rb..(r + 1) * rb], &a); } sink.fetch_add(s.to_bits() & 1, std::sync::atomic::Ordering::Relaxed); }
                else { let nt = nrow.div_ceil(32); pool.run_side(nt, &|t| { let mut s = 0f32; for r in t * 32..((t + 1) * 32).min(nrow) { s += row_dot_k(QType::IQ2S, &w[r * rb..(r + 1) * rb], &a); } sink.fetch_add(s.to_bits() & 1, std::sync::atomic::Ordering::Relaxed); }, &mut || {}); }
            }
            let el = t0.elapsed().as_secs_f64(); let per = el / (reps * ne) as f64;
            eprintln!("kbench {label} threads={th}: {:.1} us per expert gate+up, {:.1} GB/s, {:.1} ns per superblock-thread", per * 1e6, (rb * rows) as f64 / per / 1e9, el * 1e9 * th as f64 / (reps * nrow * d / 256) as f64);
            // the same rows as interleaved pairs (gate+up style); checks bit-identity against row_dot_k
            let t1 = std::time::Instant::now(); let half = nrow / 2;
            for _ in 0..reps {
                let f = |t: usize| { let mut s = 0f32; for r in t * 16..((t + 1) * 16).min(half) { let (ra, rb2) = (&w[r * rb..(r + 1) * rb], &w[(half + r) * rb..(half + r + 1) * rb]); let (g, u) = row_dot_k_pair(QType::IQ2S, ra, rb2, &a).unwrap(); s += g + u; } sink.fetch_add(s.to_bits() & 1, std::sync::atomic::Ordering::Relaxed); };
                if th == 1 { for t in 0..half.div_ceil(16) { f(t); } } else { pool.run_side(half.div_ceil(16), &f, &mut || {}); } }
            let el1 = t1.elapsed().as_secs_f64();
            let same = (0..half.min(64)).all(|r| { let (ra, rb2) = (&w[r * rb..(r + 1) * rb], &w[(half + r) * rb..(half + r + 1) * rb]); let (g, u) = row_dot_k_pair(QType::IQ2S, ra, rb2, &a).unwrap(); g.to_bits() == row_dot_k(QType::IQ2S, ra, &a).to_bits() && u.to_bits() == row_dot_k(QType::IQ2S, rb2, &a).to_bits() });
            eprintln!("kbench {label} threads={th} PAIR: {:.1} us per expert gate+up ({:.2}x), identical={same}", el1 / (reps * ne) as f64 * 1e6, el / el1);
        }
    }
}

/// `row_dot_k` of one row against several activations (`out[j]` = row . acts[j]), decoding each 256-block's
/// grid and signs once for all of them. Integer sums and float steps are the same as `row_dot_k`, so the
/// results are bit-identical; only the decode work is shared (prompt chunks / verify windows).
pub fn row_dot_k_multi(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    { if avx2() && matches!(t, QType::IQ2S | QType::IQ2XXS) && acts.len() <= 16 { unsafe { if knm() && (2..=4).contains(&acts.len()) { match (vnni(), acts.len()) { (true, 2) => iq2_kn_v::<2>(t, row, acts, out), (true, 3) => iq2_kn_v::<3>(t, row, acts, out), (true, _) => iq2_kn_v::<4>(t, row, acts, out), (false, 2) => iq2_kn::<2>(t, row, acts, out), (false, 3) => iq2_kn::<3>(t, row, acts, out), (false, _) => iq2_kn::<4>(t, row, acts, out) } } else if vnni() { iq2_k_multi_v(t, row, acts, out) } else { iq2_k_multi(t, row, acts, out) } }; return; } }
    for (j, a) in acts.iter().enumerate() { out[j] = row_dot_k(t, row, a); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2_k_multi(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) { iq2_k_multi_b::<false>(t, row, acts, out) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq2_k_multi_v(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) { iq2_k_multi_b::<true>(t, row, acts, out) }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2_k_multi_b<const V: bool>(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let na = acts.len(); let n = acts[0].q.len(); let rp = row.as_ptr();
    let mut acc = [_mm256_setzero_ps(); 16];
    let mut g = [_mm256_setzero_si256(); 8]; let mut sg = [_mm256_setzero_si256(); 8]; let mut sc = [_mm256_setzero_si256(); 8];
    let s8 = sign8(); let (es, eo) = (esigns(), esign_on());
    for i in 0..n / 256 {
        let dw = match t {
            QType::IQ2S => { let b = rp.add(i * 82); kk::pf(b); let (qs, sgp, qh, sb) = (b.add(2), b.add(34), b.add(66), b.add(74));
                for ib in 0..8 { let h = *qh.add(ib) as usize;
                    if gather() { g[ib] = kk::g2s(qs.add(4 * ib), h); } else if gidx() { g[ib] = kk::g2i(qs.add(4 * ib), h); } else {
                    let q = |l: usize| *IQ2S_GRID.get_unchecked(*qs.add(4 * ib + l) as usize | ((h << (8 - 2 * l)) & 0x300)) as i64;
                    g[ib] = _mm256_set_epi64x(q(3), q(2), q(1), q(0)); }
                    sg[ib] = kk::signs((sgp.add(4 * ib) as *const u32).read_unaligned());
                    let sbv = *sb.add(ib); sc[ib] = kk::sc16(sbv); }
                f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() }
            _ => { let b = rp.add(i * 66); kk::pf(b); let qs = b.add(2);
                for ib in 0..8 { let a1 = (qs.add(8 * ib + 4) as *const u32).read_unaligned();
                    let gq = |l: usize| *IQ2XXS_GRID.get_unchecked(*qs.add(8 * ib + l) as usize) as i64;
                    g[ib] = _mm256_set_epi64x(gq(3), gq(2), gq(1), gq(0));
                    let sl = |l: u32| if eo { *es.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as i64 } else { *s8.get_unchecked(*KSIGNS_IQ2XS.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as usize) as i64 };
                    sg[ib] = _mm256_set_epi64x(sl(3), sl(2), sl(1), sl(0));
                    let s = 2 * (a1 >> 28) as i32 + 1; sc[ib] = kk::sc2(s, s); }
                f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() }
        };
        for j in 0..na { let a = acts[j]; let xq = a.q.as_ptr();
            let mut ia = _mm256_setzero_si256();
            for ib in 0..8 { ia = kk::istep::<V>(ia, g[ib], sg[ib], xq.add(i * 256 + ib * 32), sc[ib]); }
            let d = dw * 0.125 * *a.d.get_unchecked(i);
            acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia), _mm256_set1_ps(d), acc[j]); }
    }
    for j in 0..na { out[j] = k::hsum(acc[j]); }
}
/// PYRE_QN_KNM (default on): row_dot_k_multi for 2..4 activations with a const token count, decoding one 32-value
/// chunk at a time and feeding it to every token's integer accumulator right away (no 8-chunk decode arrays, no
/// spills). Integer sums are exact and the per-super-block float steps are unchanged -> bit-identical.
pub fn knm() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_KNM").map(|v| v != "0").unwrap_or(true)) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq2_kn<const N: usize>(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) { if isplit() { iq2_kn_b::<false, N, true>(t, row, acts, out) } else { iq2_kn_b::<false, N, false>(t, row, acts, out) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn iq2_kn_v<const N: usize>(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) { if isplit() { iq2_kn_b::<true, N, true>(t, row, acts, out) } else { iq2_kn_b::<true, N, false>(t, row, acts, out) } }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn iq2_kn_b<const V: bool, const N: usize, const S: bool>(t: QType, row: &[u8], acts: &[&Q8K], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = acts[0].q.len(); let rp = row.as_ptr();
    let mut acc = [_mm256_setzero_ps(); N]; let mut xq = [std::ptr::null::<i8>(); N]; for j in 0..N { xq[j] = acts[j].q.as_ptr(); }
    let s8 = sign8(); let (es, eo) = (esigns(), esign_on());
    for i in 0..n / 256 {
        let mut ia = [_mm256_setzero_si256(); N]; let mut ib2 = [_mm256_setzero_si256(); N];
        let dw = match t {
            QType::IQ2S => { let b = rp.add(i * 82); kk::pf(b); let (qs, sgp, qh, sb) = (b.add(2), b.add(34), b.add(66), b.add(74));
                for ib in 0..8 { let h = *qh.add(ib) as usize;
                    let g = if gather() { kk::g2s(qs.add(4 * ib), h) } else if gidx() { kk::g2i(qs.add(4 * ib), h) } else {
                    let q = |l: usize| *IQ2S_GRID.get_unchecked(*qs.add(4 * ib + l) as usize | ((h << (8 - 2 * l)) & 0x300)) as i64;
                    _mm256_set_epi64x(q(3), q(2), q(1), q(0)) };
                    let sg = kk::signs((sgp.add(4 * ib) as *const u32).read_unaligned()); let sc = kk::sc16(*sb.add(ib));
                    for j in 0..N { if S && N <= 2 && ib & 1 == 1 { ib2[j] = kk::istep::<V>(ib2[j], g, sg, xq[j].add(i * 256 + ib * 32), sc); } else { ia[j] = kk::istep::<V>(ia[j], g, sg, xq[j].add(i * 256 + ib * 32), sc); } } }
                f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() }
            _ => { let b = rp.add(i * 66); kk::pf(b); let qs = b.add(2);
                for ib in 0..8 { let a1 = (qs.add(8 * ib + 4) as *const u32).read_unaligned();
                    let gq = |l: usize| *IQ2XXS_GRID.get_unchecked(*qs.add(8 * ib + l) as usize) as i64;
                    let g = _mm256_set_epi64x(gq(3), gq(2), gq(1), gq(0));
                    let sl = |l: u32| if eo { *es.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as i64 } else { *s8.get_unchecked(*KSIGNS_IQ2XS.get_unchecked(((a1 >> (7 * l)) & 127) as usize) as usize) as i64 };
                    let sg = _mm256_set_epi64x(sl(3), sl(2), sl(1), sl(0));
                    let s = 2 * (a1 >> 28) as i32 + 1; let sc = kk::sc2(s, s);
                    for j in 0..N { if S && N <= 2 && ib & 1 == 1 { ib2[j] = kk::istep::<V>(ib2[j], g, sg, xq[j].add(i * 256 + ib * 32), sc); } else { ia[j] = kk::istep::<V>(ia[j], g, sg, xq[j].add(i * 256 + ib * 32), sc); } } }
                f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32() }
        };
        if S && N <= 2 { for j in 0..N { ia[j] = _mm256_add_epi32(ia[j], ib2[j]); } }
        for j in 0..N { let d = dw * 0.125 * *acts[j].d.get_unchecked(i); acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[j]), _mm256_set1_ps(d), acc[j]); }
    }
    for j in 0..N { out[j] = k::hsum(acc[j]); }
}
/// `q2_0_dot_d` of one Q2_0 row against up to 8 activations (`out[j]` = row . acts[j]): the 2-bit unpack of each
/// block is done once for all of them; each activation's integer sums, float steps and order are the same as
/// `q2_0_dot_d` (bit-identical). Used by the verify down projection for experts shared by several tokens.
pub fn q2_0_dot_d_multi(row: &[u8], acts: &[&Q8D], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    { if avx2() && acts.len() <= 8 { unsafe { if dkn() && (2..=4).contains(&acts.len()) { match (dvnni(), acts.len()) { (true, 2) => q2_0_dn_v::<2>(row, acts, out), (true, 3) => q2_0_dn_v::<3>(row, acts, out), (true, _) => q2_0_dn_v::<4>(row, acts, out), (false, 2) => q2_0_dn::<2>(row, acts, out), (false, 3) => q2_0_dn::<3>(row, acts, out), (false, _) => q2_0_dn::<4>(row, acts, out) } } else { q2_0_d_multi_avx2(row, acts, out) } }; return; } }
    for (j, a) in acts.iter().enumerate() { out[j] = q2_0_dot_d(row, a); }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2_0_d_multi_avx2(row: &[u8], acts: &[&Q8D], out: &mut [f32]) { if dvnni() { q2_0_d_multi_v(row, acts, out) } else { q2_0_d_multi_b::<false>(row, acts, out) } }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q2_0_d_multi_v(row: &[u8], acts: &[&Q8D], out: &mut [f32]) { q2_0_d_multi_b::<true>(row, acts, out) }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q2_0_d_multi_b<const V: bool>(row: &[u8], acts: &[&Q8D], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m3 = _mm256_set1_epi8(3); let one = _mm256_set1_epi16(1); let n = acts.len();
    let mut acc = [_mm256_setzero_ps(); 8]; let mut corr = [0f32; 8]; let rp = row.as_ptr();
    for i in 0..acts[0].d.len() { let b = rp.add(i * 18);
        let h = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32();
        let qv = _mm_loadu_si128(b.add(2) as *const __m128i);
        let lo = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 2), qv), m3);
        let hi = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 6), _mm_srli_epi16(qv, 4)), m3);
        for j in 0..n { let a = acts.get_unchecked(j); let xq = a.q.as_ptr();
            let dd = h * *a.d.get_unchecked(i);
            let x0 = _mm256_loadu_si256(xq.add(i * 64) as *const __m256i); let x1 = _mm256_loadu_si256(xq.add(i * 64 + 32) as *const __m256i);
            let p = if V { _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), lo, x0), hi, x1) } else { _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), one), _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), one)) };
            acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dd), acc[j]);
            corr[j] -= dd * *a.s.get_unchecked(i) as f32; } }
    for j in 0..n { out[j] = corr[j] + k::hsum(acc[j]); }
}
/// PYRE_QN_DKN (default on): q2_0_dot_d_multi for 2..4 activations with a const token count (acc/corr in
/// registers, activation pointers hoisted). Same per-activation integer sums, float steps and order -> bit-identical.
pub fn dkn() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_DKN").map(|v| v != "0").unwrap_or(true)) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2_0_dn<const N: usize>(row: &[u8], acts: &[&Q8D], out: &mut [f32]) { q2_0_dn_b::<false, N>(row, acts, out) }
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn q2_0_dn_v<const N: usize>(row: &[u8], acts: &[&Q8D], out: &mut [f32]) { q2_0_dn_b::<true, N>(row, acts, out) }
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn q2_0_dn_b<const V: bool, const N: usize>(row: &[u8], acts: &[&Q8D], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m3 = _mm256_set1_epi8(3); let one = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); N]; let mut corr = [0f32; N]; let rp = row.as_ptr();
    let mut xq = [std::ptr::null::<i8>(); N]; let mut ad = [std::ptr::null::<f32>(); N]; let mut asp = [std::ptr::null::<i32>(); N];
    for j in 0..N { xq[j] = acts[j].q.as_ptr(); ad[j] = acts[j].d.as_ptr(); asp[j] = acts[j].s.as_ptr(); }
    for i in 0..acts[0].d.len() { let b = rp.add(i * 18);
        let h = f16::from_bits(u16::from_le_bytes([*b, *b.add(1)])).to_f32();
        let qv = _mm_loadu_si128(b.add(2) as *const __m128i);
        let lo = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 2), qv), m3);
        let hi = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(qv, 6), _mm_srli_epi16(qv, 4)), m3);
        for j in 0..N { let dd = h * *ad[j].add(i);
            let x0 = _mm256_loadu_si256(xq[j].add(i * 64) as *const __m256i); let x1 = _mm256_loadu_si256(xq[j].add(i * 64 + 32) as *const __m256i);
            let p = if V { _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), lo, x0), hi, x1) } else { _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), one), _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), one)) };
            acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dd), acc[j]);
            corr[j] -= dd * *asp[j].add(i) as f32; } }
    for j in 0..N { out[j] = corr[j] + k::hsum(acc[j]); }
}
/// (2s+1) scale vectors of an IQ2_S scale byte, see kk::sc16
pub static SC16: [[i16; 16]; 256] = { let mut t = [[0i16; 16]; 256]; let mut b = 0; while b < 256 { let mut l = 0; while l < 8 { t[b][l] = 2 * (b & 15) as i16 + 1; t[b][l + 8] = 2 * (b >> 4) as i16 + 1; l += 1; } b += 1; } t };
/// PYRE_QN_SC16 (default on): IQ2_S scale vectors from SC16
pub fn sc16_tab() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_SC16").map(|v| v != "0").unwrap_or(true)) }

/// IQ2_S: qh byte -> its four 2-bit groups one per byte (bits 8-9 of index k in byte k), see kk::g2i (Strata hi_spread.iq2s, MIT)
pub static HI2S: [u64; 256] = { let mut t = [0u64; 256]; let mut h = 0; while h < 256 { let mut k = 0; while k < 4 { t[h] |= (((h >> (2 * k)) & 3) as u64) << (8 * k); k += 1; } h += 1; } t };
/// PYRE_QN_GIDX: IQ2_S grid indices via HI2S + punpcklbw on the non-gather path (bit-identical), default on; PYRE_QN_GIDX=0 disables
pub fn gidx() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_GIDX").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_PFB: IQ2 super-block prefetch distance in bytes, default 2048 like Strata (0 = off)
pub fn pfb() -> usize { static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_PFB").ok().and_then(|v| v.parse().ok()).unwrap_or(2048)) }