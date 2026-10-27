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
