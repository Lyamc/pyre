//! Hand-written CubeCL kernels for the qwen4exp decode step (one token): the dense half of every
//! layer runs as ~14 launches on one stream with two readbacks, instead of ~60 small Burn ops.
//! Idea (fused per-layer GPU step, routed experts on the CPU) from Strata (MIT,
//! github.com/Niko1221/Strata); the kernels and layout here are pyre's own.
//! Weights are Q8_0 (32 x i8 per f32 scale), packed 4 per u32 word, one row after another.
use super::gguf_q::QMat;
use burn_cubecl::cubecl;
use cubecl::prelude::*;
use cubecl::server::Handle;

const ROWS_PER_CUBE: u32 = 8;
/// warps per cube of k_gemv4r: 8, or fewer for short matrices (the HC down projections' 324 rows filled only 11 of the
/// 3060's 28 SMs) so every SM gets work; each warp's rows and lane order are unchanged, so the bits are too.
/// PYRE_QN_GEMV_RPC=0 keeps 8 everywhere.
fn gemv_rpc(rows: usize) -> u32 { static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); let thr = *T.get_or_init(|| std::env::var("PYRE_QN_GEMV_RPC").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)); if rows >= thr { ROWS_PER_CUBE } else if rows >= 1024 { 4 } else { 2 } }
/// dense IQ4_XS kernel: PYRE_QN_NATIVE=0 Q8_0 repack, 1 f32 IQ4 kernel, 2 dp4a IQ4 kernel (int8 activations), 3 = 2 with x quantised in-kernel
/// PYRE_QN_NATS=0: the per-8-values k_gemv_iq4d_m instead of the sub-block k_gemv_iq4s_m
pub fn nats() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATS").map(|v| v == "1").unwrap_or(false)) }
/// Test override for nats(): 0 = env, 1 = iq4d_m, 2 = iq4s_m.
pub static NATS_OV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn nats_on() -> bool { match NATS_OV.load(std::sync::atomic::Ordering::Relaxed) { 1 | 3 => false, 2 => true, _ => nats() } }
pub fn natr() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR").map(|v| v != "0").unwrap_or(true)) }
pub fn natm() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATM").map(|v| v != "0").unwrap_or(true)) }
pub fn native_mode() -> u32 { static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATIVE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) }
pub static NATIVE_STAT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cube]
fn dot4(word: u32, x: &[f32], base: u32) -> f32 {
    let mut part = 0.0f32;
    #[unroll]
    for j in 0..4u32 {
        let v = dec(word, j);
        part += v * x[(base + j) as usize];
    }
    part
}

#[cube]
fn dec(word: u32, j: u32) -> f32 {
    // signed byte j as f32 through the 2^23 mantissa trick: (b ^ 0x80) + 2^23 is exact, minus 2^23 + 128 gives b
    // as an i8 (the same value the int convert gave, with fewer instructions)
    let raw = ((word ^ 0x80808080u32) >> (j * 8)) & 255u32;
    f32::reinterpret(raw | 0x4B000000u32) - 8388736.0f32
}
#[cube]
fn dotv(v0: f32, v1: f32, v2: f32, v3: f32, x: &[f32], base: u32) -> f32 {
    let mut part = 0.0f32;
    part += v0 * x[base as usize];
    part += v1 * x[(base + 1) as usize];
    part += v2 * x[(base + 2) as usize];
    part += v3 * x[(base + 3) as usize];
    part
}

#[cube(launch_unchecked)]
fn k_gemv(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let words = cols / 4;
        let wb = row * words;
        let sb = row * (cols / 32);
        let mut acc = 0.0f32;
        let mut wi = UNIT_POS_X;
        // four independent loads in flight per lane
        while wi + 96 < words {
            let w0 = w[(wb + wi) as usize];
            let w1 = w[(wb + wi + 32) as usize];
            let w2 = w[(wb + wi + 64) as usize];
            let w3 = w[(wb + wi + 96) as usize];
            let k = wi * 4;
            acc += dot4(w0, x, xoff + k) * s[(sb + k / 32) as usize];
            acc += dot4(w1, x, xoff + k + 128) * s[(sb + k / 32 + 4) as usize];
            acc += dot4(w2, x, xoff + k + 256) * s[(sb + k / 32 + 8) as usize];
            acc += dot4(w3, x, xoff + k + 384) * s[(sb + k / 32 + 12) as usize];
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            acc += dot4(w[(wb + wi) as usize], x, xoff + k) * s[(sb + k / 32) as usize];
            wi += 32;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}

/// k_gemv for up to 4 activation columns (x at xoff + c*xst, out at ooff + c*ost): each weight word is
/// loaded once for all columns; every column accumulates in exactly k_gemv's order (same results).
#[cube(launch_unchecked)]
fn k_gemv4(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let words = cols / 4;
        let wb = row * words;
        let sb = row * (cols / 32);
        let mut a0 = 0.0f32; let mut a1 = 0.0f32; let mut a2 = 0.0f32; let mut a3 = 0.0f32;
        let (x1, x2, x3) = (xoff + xst, xoff + 2 * xst, xoff + 3 * xst);
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let w0 = w[(wb + wi) as usize];
            let w1 = w[(wb + wi + 32) as usize];
            let w2 = w[(wb + wi + 64) as usize];
            let w3 = w[(wb + wi + 96) as usize];
            let k = wi * 4;
            let s0 = s[(sb + k / 32) as usize]; let s1 = s[(sb + k / 32 + 4) as usize]; let s2 = s[(sb + k / 32 + 8) as usize]; let s3 = s[(sb + k / 32 + 12) as usize];
            // each weight byte decoded once for all columns (same values and summation order as dot4)
            let (p0, p1, p2, p3) = (dec(w0, 0u32), dec(w0, 1u32), dec(w0, 2u32), dec(w0, 3u32));
            let (q0, q1, q2, q3) = (dec(w1, 0u32), dec(w1, 1u32), dec(w1, 2u32), dec(w1, 3u32));
            let (r0, r1, r2, r3) = (dec(w2, 0u32), dec(w2, 1u32), dec(w2, 2u32), dec(w2, 3u32));
            let (u0, u1, u2, u3) = (dec(w3, 0u32), dec(w3, 1u32), dec(w3, 2u32), dec(w3, 3u32));
            a0 += dotv(p0, p1, p2, p3, x, xoff + k) * s0; a0 += dotv(q0, q1, q2, q3, x, xoff + k + 128) * s1; a0 += dotv(r0, r1, r2, r3, x, xoff + k + 256) * s2; a0 += dotv(u0, u1, u2, u3, x, xoff + k + 384) * s3;
            if nt > 1 { a1 += dotv(p0, p1, p2, p3, x, x1 + k) * s0; a1 += dotv(q0, q1, q2, q3, x, x1 + k + 128) * s1; a1 += dotv(r0, r1, r2, r3, x, x1 + k + 256) * s2; a1 += dotv(u0, u1, u2, u3, x, x1 + k + 384) * s3; }
            if nt > 2 { a2 += dotv(p0, p1, p2, p3, x, x2 + k) * s0; a2 += dotv(q0, q1, q2, q3, x, x2 + k + 128) * s1; a2 += dotv(r0, r1, r2, r3, x, x2 + k + 256) * s2; a2 += dotv(u0, u1, u2, u3, x, x2 + k + 384) * s3; }
            if nt > 3 { a3 += dotv(p0, p1, p2, p3, x, x3 + k) * s0; a3 += dotv(q0, q1, q2, q3, x, x3 + k + 128) * s1; a3 += dotv(r0, r1, r2, r3, x, x3 + k + 256) * s2; a3 += dotv(u0, u1, u2, u3, x, x3 + k + 384) * s3; }
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let w0 = w[(wb + wi) as usize]; let s0 = s[(sb + k / 32) as usize];
            a0 += dot4(w0, x, xoff + k) * s0;
            if nt > 1 { a1 += dot4(w0, x, x1 + k) * s0; }
            if nt > 2 { a2 += dot4(w0, x, x2 + k) * s0; }
            if nt > 3 { a3 += dot4(w0, x, x3 + k) * s0; }
            wi += 32;
        }
        let t0 = plane_sum(a0); let t1 = plane_sum(a1); let t2 = plane_sum(a2); let t3 = plane_sum(a3);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t0;
            if nt > 1 { out[(ooff + ost + row) as usize] = t1; }
            if nt > 2 { out[(ooff + 2 * ost + row) as usize] = t2; }
            if nt > 3 { out[(ooff + 3 * ost + row) as usize] = t3; }
        }
    }
}

/// k_gemv4 with each warp on 4 consecutive rows: the activations are loaded once per k-step for all rows
/// (a quarter of k_gemv4's loads per weight word). Each (row, column) still accumulates in exactly k_gemv's order.
#[cube(launch_unchecked)]
fn k_gemv4r(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32, #[comptime] c1: bool, #[comptime] c2: bool, #[comptime] c3: bool) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 4u32;
    if row0 < rows {
        let words = cols / 4; let nb = cols / 32;
        let mut rr0 = row0 + 0u32; if rr0 >= rows { rr0 = rows - 1u32; } let wb0 = rr0 * words; let sb0 = rr0 * nb;
        let mut rr1 = row0 + 1u32; if rr1 >= rows { rr1 = rows - 1u32; } let wb1 = rr1 * words; let sb1 = rr1 * nb;
        let mut rr2 = row0 + 2u32; if rr2 >= rows { rr2 = rows - 1u32; } let wb2 = rr2 * words; let sb2 = rr2 * nb;
        let mut rr3 = row0 + 3u32; if rr3 >= rows { rr3 = rows - 1u32; } let wb3 = rr3 * words; let sb3 = rr3 * nb;
        let mut cc0 = 0u32; if cc0 >= nt { cc0 = nt - 1u32; } let xb0 = xoff + cc0 * xst;
        let mut cc1 = 1u32; if cc1 >= nt { cc1 = nt - 1u32; } let xb1 = xoff + cc1 * xst;
        let mut cc2 = 2u32; if cc2 >= nt { cc2 = nt - 1u32; } let xb2 = xoff + cc2 * xst;
        let mut cc3 = 3u32; if cc3 >= nt { cc3 = nt - 1u32; } let xb3 = xoff + cc3 * xst;
        let mut a0_0 = 0.0f32; let mut a0_1 = 0.0f32; let mut a0_2 = 0.0f32; let mut a0_3 = 0.0f32;
        let mut a1_0 = 0.0f32; let mut a1_1 = 0.0f32; let mut a1_2 = 0.0f32; let mut a1_3 = 0.0f32;
        let mut a2_0 = 0.0f32; let mut a2_1 = 0.0f32; let mut a2_2 = 0.0f32; let mut a2_3 = 0.0f32;
        let mut a3_0 = 0.0f32; let mut a3_1 = 0.0f32; let mut a3_2 = 0.0f32; let mut a3_3 = 0.0f32;
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let k = wi * 4;
            let x0_0_0 = x[(xb0 + k + 0u32) as usize]; let x0_0_1 = x[(xb0 + k + 1u32) as usize]; let x0_0_2 = x[(xb0 + k + 2u32) as usize]; let x0_0_3 = x[(xb0 + k + 3u32) as usize];
            let x0_1_0 = x[(xb0 + k + 128u32) as usize]; let x0_1_1 = x[(xb0 + k + 129u32) as usize]; let x0_1_2 = x[(xb0 + k + 130u32) as usize]; let x0_1_3 = x[(xb0 + k + 131u32) as usize];
            let x0_2_0 = x[(xb0 + k + 256u32) as usize]; let x0_2_1 = x[(xb0 + k + 257u32) as usize]; let x0_2_2 = x[(xb0 + k + 258u32) as usize]; let x0_2_3 = x[(xb0 + k + 259u32) as usize];
            let x0_3_0 = x[(xb0 + k + 384u32) as usize]; let x0_3_1 = x[(xb0 + k + 385u32) as usize]; let x0_3_2 = x[(xb0 + k + 386u32) as usize]; let x0_3_3 = x[(xb0 + k + 387u32) as usize];
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            let w0_1 = w[(wb0 + wi + 32u32) as usize]; let s0_1 = s[(sb0 + k / 32 + 4u32) as usize];
            let v0_1_0 = dec(w0_1, 0u32); let v0_1_1 = dec(w0_1, 1u32); let v0_1_2 = dec(w0_1, 2u32); let v0_1_3 = dec(w0_1, 3u32);
            let w0_2 = w[(wb0 + wi + 64u32) as usize]; let s0_2 = s[(sb0 + k / 32 + 8u32) as usize];
            let v0_2_0 = dec(w0_2, 0u32); let v0_2_1 = dec(w0_2, 1u32); let v0_2_2 = dec(w0_2, 2u32); let v0_2_3 = dec(w0_2, 3u32);
            let w0_3 = w[(wb0 + wi + 96u32) as usize]; let s0_3 = s[(sb0 + k / 32 + 12u32) as usize];
            let v0_3_0 = dec(w0_3, 0u32); let v0_3_1 = dec(w0_3, 1u32); let v0_3_2 = dec(w0_3, 2u32); let v0_3_3 = dec(w0_3, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0; a0_0 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s0_1; a0_0 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s0_2; a0_0 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s0_3;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            let w1_1 = w[(wb1 + wi + 32u32) as usize]; let s1_1 = s[(sb1 + k / 32 + 4u32) as usize];
            let v1_1_0 = dec(w1_1, 0u32); let v1_1_1 = dec(w1_1, 1u32); let v1_1_2 = dec(w1_1, 2u32); let v1_1_3 = dec(w1_1, 3u32);
            let w1_2 = w[(wb1 + wi + 64u32) as usize]; let s1_2 = s[(sb1 + k / 32 + 8u32) as usize];
            let v1_2_0 = dec(w1_2, 0u32); let v1_2_1 = dec(w1_2, 1u32); let v1_2_2 = dec(w1_2, 2u32); let v1_2_3 = dec(w1_2, 3u32);
            let w1_3 = w[(wb1 + wi + 96u32) as usize]; let s1_3 = s[(sb1 + k / 32 + 12u32) as usize];
            let v1_3_0 = dec(w1_3, 0u32); let v1_3_1 = dec(w1_3, 1u32); let v1_3_2 = dec(w1_3, 2u32); let v1_3_3 = dec(w1_3, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0; a1_0 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s1_1; a1_0 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s1_2; a1_0 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s1_3;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            let w2_1 = w[(wb2 + wi + 32u32) as usize]; let s2_1 = s[(sb2 + k / 32 + 4u32) as usize];
            let v2_1_0 = dec(w2_1, 0u32); let v2_1_1 = dec(w2_1, 1u32); let v2_1_2 = dec(w2_1, 2u32); let v2_1_3 = dec(w2_1, 3u32);
            let w2_2 = w[(wb2 + wi + 64u32) as usize]; let s2_2 = s[(sb2 + k / 32 + 8u32) as usize];
            let v2_2_0 = dec(w2_2, 0u32); let v2_2_1 = dec(w2_2, 1u32); let v2_2_2 = dec(w2_2, 2u32); let v2_2_3 = dec(w2_2, 3u32);
            let w2_3 = w[(wb2 + wi + 96u32) as usize]; let s2_3 = s[(sb2 + k / 32 + 12u32) as usize];
            let v2_3_0 = dec(w2_3, 0u32); let v2_3_1 = dec(w2_3, 1u32); let v2_3_2 = dec(w2_3, 2u32); let v2_3_3 = dec(w2_3, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0; a2_0 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s2_1; a2_0 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s2_2; a2_0 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s2_3;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            let w3_1 = w[(wb3 + wi + 32u32) as usize]; let s3_1 = s[(sb3 + k / 32 + 4u32) as usize];
            let v3_1_0 = dec(w3_1, 0u32); let v3_1_1 = dec(w3_1, 1u32); let v3_1_2 = dec(w3_1, 2u32); let v3_1_3 = dec(w3_1, 3u32);
            let w3_2 = w[(wb3 + wi + 64u32) as usize]; let s3_2 = s[(sb3 + k / 32 + 8u32) as usize];
            let v3_2_0 = dec(w3_2, 0u32); let v3_2_1 = dec(w3_2, 1u32); let v3_2_2 = dec(w3_2, 2u32); let v3_2_3 = dec(w3_2, 3u32);
            let w3_3 = w[(wb3 + wi + 96u32) as usize]; let s3_3 = s[(sb3 + k / 32 + 12u32) as usize];
            let v3_3_0 = dec(w3_3, 0u32); let v3_3_1 = dec(w3_3, 1u32); let v3_3_2 = dec(w3_3, 2u32); let v3_3_3 = dec(w3_3, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0; a3_0 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s3_1; a3_0 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s3_2; a3_0 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s3_3;
            if c1 {
                let x1_0_0 = x[(xb1 + k + 0u32) as usize]; let x1_0_1 = x[(xb1 + k + 1u32) as usize]; let x1_0_2 = x[(xb1 + k + 2u32) as usize]; let x1_0_3 = x[(xb1 + k + 3u32) as usize];
                let x1_1_0 = x[(xb1 + k + 128u32) as usize]; let x1_1_1 = x[(xb1 + k + 129u32) as usize]; let x1_1_2 = x[(xb1 + k + 130u32) as usize]; let x1_1_3 = x[(xb1 + k + 131u32) as usize];
                let x1_2_0 = x[(xb1 + k + 256u32) as usize]; let x1_2_1 = x[(xb1 + k + 257u32) as usize]; let x1_2_2 = x[(xb1 + k + 258u32) as usize]; let x1_2_3 = x[(xb1 + k + 259u32) as usize];
                let x1_3_0 = x[(xb1 + k + 384u32) as usize]; let x1_3_1 = x[(xb1 + k + 385u32) as usize]; let x1_3_2 = x[(xb1 + k + 386u32) as usize]; let x1_3_3 = x[(xb1 + k + 387u32) as usize];
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0; a0_1 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s0_1; a0_1 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s0_2; a0_1 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s0_3;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0; a1_1 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s1_1; a1_1 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s1_2; a1_1 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s1_3;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0; a2_1 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s2_1; a2_1 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s2_2; a2_1 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s2_3;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0; a3_1 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s3_1; a3_1 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s3_2; a3_1 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s3_3;
            }
            if c2 {
                let x2_0_0 = x[(xb2 + k + 0u32) as usize]; let x2_0_1 = x[(xb2 + k + 1u32) as usize]; let x2_0_2 = x[(xb2 + k + 2u32) as usize]; let x2_0_3 = x[(xb2 + k + 3u32) as usize];
                let x2_1_0 = x[(xb2 + k + 128u32) as usize]; let x2_1_1 = x[(xb2 + k + 129u32) as usize]; let x2_1_2 = x[(xb2 + k + 130u32) as usize]; let x2_1_3 = x[(xb2 + k + 131u32) as usize];
                let x2_2_0 = x[(xb2 + k + 256u32) as usize]; let x2_2_1 = x[(xb2 + k + 257u32) as usize]; let x2_2_2 = x[(xb2 + k + 258u32) as usize]; let x2_2_3 = x[(xb2 + k + 259u32) as usize];
                let x2_3_0 = x[(xb2 + k + 384u32) as usize]; let x2_3_1 = x[(xb2 + k + 385u32) as usize]; let x2_3_2 = x[(xb2 + k + 386u32) as usize]; let x2_3_3 = x[(xb2 + k + 387u32) as usize];
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0; a0_2 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s0_1; a0_2 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s0_2; a0_2 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s0_3;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0; a1_2 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s1_1; a1_2 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s1_2; a1_2 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s1_3;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0; a2_2 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s2_1; a2_2 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s2_2; a2_2 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s2_3;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0; a3_2 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s3_1; a3_2 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s3_2; a3_2 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s3_3;
            }
            if c3 {
                let x3_0_0 = x[(xb3 + k + 0u32) as usize]; let x3_0_1 = x[(xb3 + k + 1u32) as usize]; let x3_0_2 = x[(xb3 + k + 2u32) as usize]; let x3_0_3 = x[(xb3 + k + 3u32) as usize];
                let x3_1_0 = x[(xb3 + k + 128u32) as usize]; let x3_1_1 = x[(xb3 + k + 129u32) as usize]; let x3_1_2 = x[(xb3 + k + 130u32) as usize]; let x3_1_3 = x[(xb3 + k + 131u32) as usize];
                let x3_2_0 = x[(xb3 + k + 256u32) as usize]; let x3_2_1 = x[(xb3 + k + 257u32) as usize]; let x3_2_2 = x[(xb3 + k + 258u32) as usize]; let x3_2_3 = x[(xb3 + k + 259u32) as usize];
                let x3_3_0 = x[(xb3 + k + 384u32) as usize]; let x3_3_1 = x[(xb3 + k + 385u32) as usize]; let x3_3_2 = x[(xb3 + k + 386u32) as usize]; let x3_3_3 = x[(xb3 + k + 387u32) as usize];
                a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0; a0_3 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s0_1; a0_3 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s0_2; a0_3 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s0_3;
                a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0; a1_3 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s1_1; a1_3 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s1_2; a1_3 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s1_3;
                a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0; a2_3 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s2_1; a2_3 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s2_2; a2_3 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s2_3;
                a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0; a3_3 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s3_1; a3_3 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s3_2; a3_3 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s3_3;
            }
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let x0_0_0 = x[(xb0 + k + 0u32) as usize]; let x0_0_1 = x[(xb0 + k + 1u32) as usize]; let x0_0_2 = x[(xb0 + k + 2u32) as usize]; let x0_0_3 = x[(xb0 + k + 3u32) as usize];
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0;
            if c1 {
                let x1_0_0 = x[(xb1 + k + 0u32) as usize]; let x1_0_1 = x[(xb1 + k + 1u32) as usize]; let x1_0_2 = x[(xb1 + k + 2u32) as usize]; let x1_0_3 = x[(xb1 + k + 3u32) as usize];
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0;
            }
            if c2 {
                let x2_0_0 = x[(xb2 + k + 0u32) as usize]; let x2_0_1 = x[(xb2 + k + 1u32) as usize]; let x2_0_2 = x[(xb2 + k + 2u32) as usize]; let x2_0_3 = x[(xb2 + k + 3u32) as usize];
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0;
            }
            if c3 {
                let x3_0_0 = x[(xb3 + k + 0u32) as usize]; let x3_0_1 = x[(xb3 + k + 1u32) as usize]; let x3_0_2 = x[(xb3 + k + 2u32) as usize]; let x3_0_3 = x[(xb3 + k + 3u32) as usize];
                a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0;
                a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0;
                a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0;
                a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0;
            }
            wi += 32;
        }
        let t0_0 = plane_sum(a0_0); let t0_1 = plane_sum(a0_1); let t0_2 = plane_sum(a0_2); let t0_3 = plane_sum(a0_3);
        let t1_0 = plane_sum(a1_0); let t1_1 = plane_sum(a1_1); let t1_2 = plane_sum(a1_2); let t1_3 = plane_sum(a1_3);
        let t2_0 = plane_sum(a2_0); let t2_1 = plane_sum(a2_1); let t2_2 = plane_sum(a2_2); let t2_3 = plane_sum(a2_3);
        let t3_0 = plane_sum(a3_0); let t3_1 = plane_sum(a3_1); let t3_2 = plane_sum(a3_2); let t3_3 = plane_sum(a3_3);
        if UNIT_POS_X == 0 {
            if row0 + 0u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 0u32) as usize] = t0_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 0u32) as usize] = t0_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 0u32) as usize] = t0_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 0u32) as usize] = t0_3; }
            }
            if row0 + 1u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 1u32) as usize] = t1_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 1u32) as usize] = t1_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 1u32) as usize] = t1_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 1u32) as usize] = t1_3; }
            }
            if row0 + 2u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 2u32) as usize] = t2_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 2u32) as usize] = t2_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 2u32) as usize] = t2_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 2u32) as usize] = t2_3; }
            }
            if row0 + 3u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 3u32) as usize] = t3_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 3u32) as usize] = t3_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 3u32) as usize] = t3_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 3u32) as usize] = t3_3; }
            }
        }
    }
}

/// k_gemv4r with the activations read as 4-wide vectors (one 16-byte load instead of four); same arithmetic.
#[cube(launch_unchecked)]
fn k_gemv4v<N: Size>(x: &[Vector<f32, N>], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32, #[comptime] c1: bool, #[comptime] c2: bool, #[comptime] c3: bool) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 4u32;
    if row0 < rows {
        let words = cols / 4; let nb = cols / 32;
        let mut rr0 = row0 + 0u32; if rr0 >= rows { rr0 = rows - 1u32; } let wb0 = rr0 * words; let sb0 = rr0 * nb;
        let mut rr1 = row0 + 1u32; if rr1 >= rows { rr1 = rows - 1u32; } let wb1 = rr1 * words; let sb1 = rr1 * nb;
        let mut rr2 = row0 + 2u32; if rr2 >= rows { rr2 = rows - 1u32; } let wb2 = rr2 * words; let sb2 = rr2 * nb;
        let mut rr3 = row0 + 3u32; if rr3 >= rows { rr3 = rows - 1u32; } let wb3 = rr3 * words; let sb3 = rr3 * nb;
        let mut cc0 = 0u32; if cc0 >= nt { cc0 = nt - 1u32; } let xb0 = xoff + cc0 * xst;
        let mut cc1 = 1u32; if cc1 >= nt { cc1 = nt - 1u32; } let xb1 = xoff + cc1 * xst;
        let mut cc2 = 2u32; if cc2 >= nt { cc2 = nt - 1u32; } let xb2 = xoff + cc2 * xst;
        let mut cc3 = 3u32; if cc3 >= nt { cc3 = nt - 1u32; } let xb3 = xoff + cc3 * xst;
        let mut a0_0 = 0.0f32; let mut a0_1 = 0.0f32; let mut a0_2 = 0.0f32; let mut a0_3 = 0.0f32;
        let mut a1_0 = 0.0f32; let mut a1_1 = 0.0f32; let mut a1_2 = 0.0f32; let mut a1_3 = 0.0f32;
        let mut a2_0 = 0.0f32; let mut a2_1 = 0.0f32; let mut a2_2 = 0.0f32; let mut a2_3 = 0.0f32;
        let mut a3_0 = 0.0f32; let mut a3_1 = 0.0f32; let mut a3_2 = 0.0f32; let mut a3_3 = 0.0f32;
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let k = wi * 4;
            let vx0_0 = x[((xb0 + k + 0u32) / 4u32) as usize]; let x0_0_0 = vx0_0.extract(0usize); let x0_0_1 = vx0_0.extract(1usize); let x0_0_2 = vx0_0.extract(2usize); let x0_0_3 = vx0_0.extract(3usize);
            let vx0_1 = x[((xb0 + k + 128u32) / 4u32) as usize]; let x0_1_0 = vx0_1.extract(0usize); let x0_1_1 = vx0_1.extract(1usize); let x0_1_2 = vx0_1.extract(2usize); let x0_1_3 = vx0_1.extract(3usize);
            let vx0_2 = x[((xb0 + k + 256u32) / 4u32) as usize]; let x0_2_0 = vx0_2.extract(0usize); let x0_2_1 = vx0_2.extract(1usize); let x0_2_2 = vx0_2.extract(2usize); let x0_2_3 = vx0_2.extract(3usize);
            let vx0_3 = x[((xb0 + k + 384u32) / 4u32) as usize]; let x0_3_0 = vx0_3.extract(0usize); let x0_3_1 = vx0_3.extract(1usize); let x0_3_2 = vx0_3.extract(2usize); let x0_3_3 = vx0_3.extract(3usize);
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            let w0_1 = w[(wb0 + wi + 32u32) as usize]; let s0_1 = s[(sb0 + k / 32 + 4u32) as usize];
            let v0_1_0 = dec(w0_1, 0u32); let v0_1_1 = dec(w0_1, 1u32); let v0_1_2 = dec(w0_1, 2u32); let v0_1_3 = dec(w0_1, 3u32);
            let w0_2 = w[(wb0 + wi + 64u32) as usize]; let s0_2 = s[(sb0 + k / 32 + 8u32) as usize];
            let v0_2_0 = dec(w0_2, 0u32); let v0_2_1 = dec(w0_2, 1u32); let v0_2_2 = dec(w0_2, 2u32); let v0_2_3 = dec(w0_2, 3u32);
            let w0_3 = w[(wb0 + wi + 96u32) as usize]; let s0_3 = s[(sb0 + k / 32 + 12u32) as usize];
            let v0_3_0 = dec(w0_3, 0u32); let v0_3_1 = dec(w0_3, 1u32); let v0_3_2 = dec(w0_3, 2u32); let v0_3_3 = dec(w0_3, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0; a0_0 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s0_1; a0_0 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s0_2; a0_0 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s0_3;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            let w1_1 = w[(wb1 + wi + 32u32) as usize]; let s1_1 = s[(sb1 + k / 32 + 4u32) as usize];
            let v1_1_0 = dec(w1_1, 0u32); let v1_1_1 = dec(w1_1, 1u32); let v1_1_2 = dec(w1_1, 2u32); let v1_1_3 = dec(w1_1, 3u32);
            let w1_2 = w[(wb1 + wi + 64u32) as usize]; let s1_2 = s[(sb1 + k / 32 + 8u32) as usize];
            let v1_2_0 = dec(w1_2, 0u32); let v1_2_1 = dec(w1_2, 1u32); let v1_2_2 = dec(w1_2, 2u32); let v1_2_3 = dec(w1_2, 3u32);
            let w1_3 = w[(wb1 + wi + 96u32) as usize]; let s1_3 = s[(sb1 + k / 32 + 12u32) as usize];
            let v1_3_0 = dec(w1_3, 0u32); let v1_3_1 = dec(w1_3, 1u32); let v1_3_2 = dec(w1_3, 2u32); let v1_3_3 = dec(w1_3, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0; a1_0 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s1_1; a1_0 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s1_2; a1_0 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s1_3;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            let w2_1 = w[(wb2 + wi + 32u32) as usize]; let s2_1 = s[(sb2 + k / 32 + 4u32) as usize];
            let v2_1_0 = dec(w2_1, 0u32); let v2_1_1 = dec(w2_1, 1u32); let v2_1_2 = dec(w2_1, 2u32); let v2_1_3 = dec(w2_1, 3u32);
            let w2_2 = w[(wb2 + wi + 64u32) as usize]; let s2_2 = s[(sb2 + k / 32 + 8u32) as usize];
            let v2_2_0 = dec(w2_2, 0u32); let v2_2_1 = dec(w2_2, 1u32); let v2_2_2 = dec(w2_2, 2u32); let v2_2_3 = dec(w2_2, 3u32);
            let w2_3 = w[(wb2 + wi + 96u32) as usize]; let s2_3 = s[(sb2 + k / 32 + 12u32) as usize];
            let v2_3_0 = dec(w2_3, 0u32); let v2_3_1 = dec(w2_3, 1u32); let v2_3_2 = dec(w2_3, 2u32); let v2_3_3 = dec(w2_3, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0; a2_0 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s2_1; a2_0 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s2_2; a2_0 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s2_3;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            let w3_1 = w[(wb3 + wi + 32u32) as usize]; let s3_1 = s[(sb3 + k / 32 + 4u32) as usize];
            let v3_1_0 = dec(w3_1, 0u32); let v3_1_1 = dec(w3_1, 1u32); let v3_1_2 = dec(w3_1, 2u32); let v3_1_3 = dec(w3_1, 3u32);
            let w3_2 = w[(wb3 + wi + 64u32) as usize]; let s3_2 = s[(sb3 + k / 32 + 8u32) as usize];
            let v3_2_0 = dec(w3_2, 0u32); let v3_2_1 = dec(w3_2, 1u32); let v3_2_2 = dec(w3_2, 2u32); let v3_2_3 = dec(w3_2, 3u32);
            let w3_3 = w[(wb3 + wi + 96u32) as usize]; let s3_3 = s[(sb3 + k / 32 + 12u32) as usize];
            let v3_3_0 = dec(w3_3, 0u32); let v3_3_1 = dec(w3_3, 1u32); let v3_3_2 = dec(w3_3, 2u32); let v3_3_3 = dec(w3_3, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0; a3_0 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s3_1; a3_0 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s3_2; a3_0 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s3_3;
            if c1 {
                let vx1_0 = x[((xb1 + k + 0u32) / 4u32) as usize]; let x1_0_0 = vx1_0.extract(0usize); let x1_0_1 = vx1_0.extract(1usize); let x1_0_2 = vx1_0.extract(2usize); let x1_0_3 = vx1_0.extract(3usize);
                let vx1_1 = x[((xb1 + k + 128u32) / 4u32) as usize]; let x1_1_0 = vx1_1.extract(0usize); let x1_1_1 = vx1_1.extract(1usize); let x1_1_2 = vx1_1.extract(2usize); let x1_1_3 = vx1_1.extract(3usize);
                let vx1_2 = x[((xb1 + k + 256u32) / 4u32) as usize]; let x1_2_0 = vx1_2.extract(0usize); let x1_2_1 = vx1_2.extract(1usize); let x1_2_2 = vx1_2.extract(2usize); let x1_2_3 = vx1_2.extract(3usize);
                let vx1_3 = x[((xb1 + k + 384u32) / 4u32) as usize]; let x1_3_0 = vx1_3.extract(0usize); let x1_3_1 = vx1_3.extract(1usize); let x1_3_2 = vx1_3.extract(2usize); let x1_3_3 = vx1_3.extract(3usize);
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0; a0_1 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s0_1; a0_1 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s0_2; a0_1 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s0_3;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0; a1_1 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s1_1; a1_1 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s1_2; a1_1 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s1_3;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0; a2_1 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s2_1; a2_1 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s2_2; a2_1 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s2_3;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0; a3_1 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s3_1; a3_1 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s3_2; a3_1 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s3_3;
            }
            if c2 {
                let vx2_0 = x[((xb2 + k + 0u32) / 4u32) as usize]; let x2_0_0 = vx2_0.extract(0usize); let x2_0_1 = vx2_0.extract(1usize); let x2_0_2 = vx2_0.extract(2usize); let x2_0_3 = vx2_0.extract(3usize);
                let vx2_1 = x[((xb2 + k + 128u32) / 4u32) as usize]; let x2_1_0 = vx2_1.extract(0usize); let x2_1_1 = vx2_1.extract(1usize); let x2_1_2 = vx2_1.extract(2usize); let x2_1_3 = vx2_1.extract(3usize);
                let vx2_2 = x[((xb2 + k + 256u32) / 4u32) as usize]; let x2_2_0 = vx2_2.extract(0usize); let x2_2_1 = vx2_2.extract(1usize); let x2_2_2 = vx2_2.extract(2usize); let x2_2_3 = vx2_2.extract(3usize);
                let vx2_3 = x[((xb2 + k + 384u32) / 4u32) as usize]; let x2_3_0 = vx2_3.extract(0usize); let x2_3_1 = vx2_3.extract(1usize); let x2_3_2 = vx2_3.extract(2usize); let x2_3_3 = vx2_3.extract(3usize);
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0; a0_2 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s0_1; a0_2 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s0_2; a0_2 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s0_3;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0; a1_2 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s1_1; a1_2 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s1_2; a1_2 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s1_3;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0; a2_2 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s2_1; a2_2 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s2_2; a2_2 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s2_3;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0; a3_2 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s3_1; a3_2 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s3_2; a3_2 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s3_3;
            }
            if c3 {
                let vx3_0 = x[((xb3 + k + 0u32) / 4u32) as usize]; let x3_0_0 = vx3_0.extract(0usize); let x3_0_1 = vx3_0.extract(1usize); let x3_0_2 = vx3_0.extract(2usize); let x3_0_3 = vx3_0.extract(3usize);
                let vx3_1 = x[((xb3 + k + 128u32) / 4u32) as usize]; let x3_1_0 = vx3_1.extract(0usize); let x3_1_1 = vx3_1.extract(1usize); let x3_1_2 = vx3_1.extract(2usize); let x3_1_3 = vx3_1.extract(3usize);
                let vx3_2 = x[((xb3 + k + 256u32) / 4u32) as usize]; let x3_2_0 = vx3_2.extract(0usize); let x3_2_1 = vx3_2.extract(1usize); let x3_2_2 = vx3_2.extract(2usize); let x3_2_3 = vx3_2.extract(3usize);
                let vx3_3 = x[((xb3 + k + 384u32) / 4u32) as usize]; let x3_3_0 = vx3_3.extract(0usize); let x3_3_1 = vx3_3.extract(1usize); let x3_3_2 = vx3_3.extract(2usize); let x3_3_3 = vx3_3.extract(3usize);
                a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0; a0_3 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s0_1; a0_3 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s0_2; a0_3 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s0_3;
                a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0; a1_3 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s1_1; a1_3 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s1_2; a1_3 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s1_3;
                a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0; a2_3 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s2_1; a2_3 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s2_2; a2_3 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s2_3;
                a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0; a3_3 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s3_1; a3_3 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s3_2; a3_3 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s3_3;
            }
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let vx0_0 = x[((xb0 + k + 0u32) / 4u32) as usize]; let x0_0_0 = vx0_0.extract(0usize); let x0_0_1 = vx0_0.extract(1usize); let x0_0_2 = vx0_0.extract(2usize); let x0_0_3 = vx0_0.extract(3usize);
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0;
            if c1 {
                let vx1_0 = x[((xb1 + k + 0u32) / 4u32) as usize]; let x1_0_0 = vx1_0.extract(0usize); let x1_0_1 = vx1_0.extract(1usize); let x1_0_2 = vx1_0.extract(2usize); let x1_0_3 = vx1_0.extract(3usize);
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0;
            }
            if c2 {
                let vx2_0 = x[((xb2 + k + 0u32) / 4u32) as usize]; let x2_0_0 = vx2_0.extract(0usize); let x2_0_1 = vx2_0.extract(1usize); let x2_0_2 = vx2_0.extract(2usize); let x2_0_3 = vx2_0.extract(3usize);
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0;
            }
            if c3 {
                let vx3_0 = x[((xb3 + k + 0u32) / 4u32) as usize]; let x3_0_0 = vx3_0.extract(0usize); let x3_0_1 = vx3_0.extract(1usize); let x3_0_2 = vx3_0.extract(2usize); let x3_0_3 = vx3_0.extract(3usize);
                a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0;
                a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0;
                a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0;
                a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0;
            }
            wi += 32;
        }
        let t0_0 = plane_sum(a0_0); let t0_1 = plane_sum(a0_1); let t0_2 = plane_sum(a0_2); let t0_3 = plane_sum(a0_3);
        let t1_0 = plane_sum(a1_0); let t1_1 = plane_sum(a1_1); let t1_2 = plane_sum(a1_2); let t1_3 = plane_sum(a1_3);
        let t2_0 = plane_sum(a2_0); let t2_1 = plane_sum(a2_1); let t2_2 = plane_sum(a2_2); let t2_3 = plane_sum(a2_3);
        let t3_0 = plane_sum(a3_0); let t3_1 = plane_sum(a3_1); let t3_2 = plane_sum(a3_2); let t3_3 = plane_sum(a3_3);
        if UNIT_POS_X == 0 {
            if row0 + 0u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 0u32) as usize] = t0_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 0u32) as usize] = t0_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 0u32) as usize] = t0_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 0u32) as usize] = t0_3; }
            }
            if row0 + 1u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 1u32) as usize] = t1_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 1u32) as usize] = t1_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 1u32) as usize] = t1_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 1u32) as usize] = t1_3; }
            }
            if row0 + 2u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 2u32) as usize] = t2_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 2u32) as usize] = t2_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 2u32) as usize] = t2_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 2u32) as usize] = t2_3; }
            }
            if row0 + 3u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 3u32) as usize] = t3_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 3u32) as usize] = t3_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 3u32) as usize] = t3_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 3u32) as usize] = t3_3; }
            }
        }
    }
}

#[cube]
fn dotx(v0: f32, v1: f32, v2: f32, v3: f32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    let mut part = 0.0f32;
    part += v0 * x0;
    part += v1 * x1;
    part += v2 * x2;
    part += v3 * x3;
    part
}

/// k_gemv4r with 8 rows per warp (PYRE_QN_GEMV4R=8).
#[cube(launch_unchecked)]
fn k_gemv4r8(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 8u32;
    if row0 < rows {
        let words = cols / 4; let nb = cols / 32;
        let mut rr0 = row0 + 0u32; if rr0 >= rows { rr0 = rows - 1u32; } let wb0 = rr0 * words; let sb0 = rr0 * nb;
        let mut rr1 = row0 + 1u32; if rr1 >= rows { rr1 = rows - 1u32; } let wb1 = rr1 * words; let sb1 = rr1 * nb;
        let mut rr2 = row0 + 2u32; if rr2 >= rows { rr2 = rows - 1u32; } let wb2 = rr2 * words; let sb2 = rr2 * nb;
        let mut rr3 = row0 + 3u32; if rr3 >= rows { rr3 = rows - 1u32; } let wb3 = rr3 * words; let sb3 = rr3 * nb;
        let mut rr4 = row0 + 4u32; if rr4 >= rows { rr4 = rows - 1u32; } let wb4 = rr4 * words; let sb4 = rr4 * nb;
        let mut rr5 = row0 + 5u32; if rr5 >= rows { rr5 = rows - 1u32; } let wb5 = rr5 * words; let sb5 = rr5 * nb;
        let mut rr6 = row0 + 6u32; if rr6 >= rows { rr6 = rows - 1u32; } let wb6 = rr6 * words; let sb6 = rr6 * nb;
        let mut rr7 = row0 + 7u32; if rr7 >= rows { rr7 = rows - 1u32; } let wb7 = rr7 * words; let sb7 = rr7 * nb;
        let mut cc0 = 0u32; if cc0 >= nt { cc0 = nt - 1u32; } let xb0 = xoff + cc0 * xst;
        let mut cc1 = 1u32; if cc1 >= nt { cc1 = nt - 1u32; } let xb1 = xoff + cc1 * xst;
        let mut cc2 = 2u32; if cc2 >= nt { cc2 = nt - 1u32; } let xb2 = xoff + cc2 * xst;
        let mut cc3 = 3u32; if cc3 >= nt { cc3 = nt - 1u32; } let xb3 = xoff + cc3 * xst;
        let mut a0_0 = 0.0f32; let mut a0_1 = 0.0f32; let mut a0_2 = 0.0f32; let mut a0_3 = 0.0f32;
        let mut a1_0 = 0.0f32; let mut a1_1 = 0.0f32; let mut a1_2 = 0.0f32; let mut a1_3 = 0.0f32;
        let mut a2_0 = 0.0f32; let mut a2_1 = 0.0f32; let mut a2_2 = 0.0f32; let mut a2_3 = 0.0f32;
        let mut a3_0 = 0.0f32; let mut a3_1 = 0.0f32; let mut a3_2 = 0.0f32; let mut a3_3 = 0.0f32;
        let mut a4_0 = 0.0f32; let mut a4_1 = 0.0f32; let mut a4_2 = 0.0f32; let mut a4_3 = 0.0f32;
        let mut a5_0 = 0.0f32; let mut a5_1 = 0.0f32; let mut a5_2 = 0.0f32; let mut a5_3 = 0.0f32;
        let mut a6_0 = 0.0f32; let mut a6_1 = 0.0f32; let mut a6_2 = 0.0f32; let mut a6_3 = 0.0f32;
        let mut a7_0 = 0.0f32; let mut a7_1 = 0.0f32; let mut a7_2 = 0.0f32; let mut a7_3 = 0.0f32;
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let k = wi * 4;
            let x0_0_0 = x[(xb0 + k + 0u32) as usize]; let x0_0_1 = x[(xb0 + k + 1u32) as usize]; let x0_0_2 = x[(xb0 + k + 2u32) as usize]; let x0_0_3 = x[(xb0 + k + 3u32) as usize];
            let x1_0_0 = x[(xb1 + k + 0u32) as usize]; let x1_0_1 = x[(xb1 + k + 1u32) as usize]; let x1_0_2 = x[(xb1 + k + 2u32) as usize]; let x1_0_3 = x[(xb1 + k + 3u32) as usize];
            let x2_0_0 = x[(xb2 + k + 0u32) as usize]; let x2_0_1 = x[(xb2 + k + 1u32) as usize]; let x2_0_2 = x[(xb2 + k + 2u32) as usize]; let x2_0_3 = x[(xb2 + k + 3u32) as usize];
            let x3_0_0 = x[(xb3 + k + 0u32) as usize]; let x3_0_1 = x[(xb3 + k + 1u32) as usize]; let x3_0_2 = x[(xb3 + k + 2u32) as usize]; let x3_0_3 = x[(xb3 + k + 3u32) as usize];
            let x0_1_0 = x[(xb0 + k + 128u32) as usize]; let x0_1_1 = x[(xb0 + k + 129u32) as usize]; let x0_1_2 = x[(xb0 + k + 130u32) as usize]; let x0_1_3 = x[(xb0 + k + 131u32) as usize];
            let x1_1_0 = x[(xb1 + k + 128u32) as usize]; let x1_1_1 = x[(xb1 + k + 129u32) as usize]; let x1_1_2 = x[(xb1 + k + 130u32) as usize]; let x1_1_3 = x[(xb1 + k + 131u32) as usize];
            let x2_1_0 = x[(xb2 + k + 128u32) as usize]; let x2_1_1 = x[(xb2 + k + 129u32) as usize]; let x2_1_2 = x[(xb2 + k + 130u32) as usize]; let x2_1_3 = x[(xb2 + k + 131u32) as usize];
            let x3_1_0 = x[(xb3 + k + 128u32) as usize]; let x3_1_1 = x[(xb3 + k + 129u32) as usize]; let x3_1_2 = x[(xb3 + k + 130u32) as usize]; let x3_1_3 = x[(xb3 + k + 131u32) as usize];
            let x0_2_0 = x[(xb0 + k + 256u32) as usize]; let x0_2_1 = x[(xb0 + k + 257u32) as usize]; let x0_2_2 = x[(xb0 + k + 258u32) as usize]; let x0_2_3 = x[(xb0 + k + 259u32) as usize];
            let x1_2_0 = x[(xb1 + k + 256u32) as usize]; let x1_2_1 = x[(xb1 + k + 257u32) as usize]; let x1_2_2 = x[(xb1 + k + 258u32) as usize]; let x1_2_3 = x[(xb1 + k + 259u32) as usize];
            let x2_2_0 = x[(xb2 + k + 256u32) as usize]; let x2_2_1 = x[(xb2 + k + 257u32) as usize]; let x2_2_2 = x[(xb2 + k + 258u32) as usize]; let x2_2_3 = x[(xb2 + k + 259u32) as usize];
            let x3_2_0 = x[(xb3 + k + 256u32) as usize]; let x3_2_1 = x[(xb3 + k + 257u32) as usize]; let x3_2_2 = x[(xb3 + k + 258u32) as usize]; let x3_2_3 = x[(xb3 + k + 259u32) as usize];
            let x0_3_0 = x[(xb0 + k + 384u32) as usize]; let x0_3_1 = x[(xb0 + k + 385u32) as usize]; let x0_3_2 = x[(xb0 + k + 386u32) as usize]; let x0_3_3 = x[(xb0 + k + 387u32) as usize];
            let x1_3_0 = x[(xb1 + k + 384u32) as usize]; let x1_3_1 = x[(xb1 + k + 385u32) as usize]; let x1_3_2 = x[(xb1 + k + 386u32) as usize]; let x1_3_3 = x[(xb1 + k + 387u32) as usize];
            let x2_3_0 = x[(xb2 + k + 384u32) as usize]; let x2_3_1 = x[(xb2 + k + 385u32) as usize]; let x2_3_2 = x[(xb2 + k + 386u32) as usize]; let x2_3_3 = x[(xb2 + k + 387u32) as usize];
            let x3_3_0 = x[(xb3 + k + 384u32) as usize]; let x3_3_1 = x[(xb3 + k + 385u32) as usize]; let x3_3_2 = x[(xb3 + k + 386u32) as usize]; let x3_3_3 = x[(xb3 + k + 387u32) as usize];
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            let w0_1 = w[(wb0 + wi + 32u32) as usize]; let s0_1 = s[(sb0 + k / 32 + 4u32) as usize];
            let v0_1_0 = dec(w0_1, 0u32); let v0_1_1 = dec(w0_1, 1u32); let v0_1_2 = dec(w0_1, 2u32); let v0_1_3 = dec(w0_1, 3u32);
            let w0_2 = w[(wb0 + wi + 64u32) as usize]; let s0_2 = s[(sb0 + k / 32 + 8u32) as usize];
            let v0_2_0 = dec(w0_2, 0u32); let v0_2_1 = dec(w0_2, 1u32); let v0_2_2 = dec(w0_2, 2u32); let v0_2_3 = dec(w0_2, 3u32);
            let w0_3 = w[(wb0 + wi + 96u32) as usize]; let s0_3 = s[(sb0 + k / 32 + 12u32) as usize];
            let v0_3_0 = dec(w0_3, 0u32); let v0_3_1 = dec(w0_3, 1u32); let v0_3_2 = dec(w0_3, 2u32); let v0_3_3 = dec(w0_3, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0; a0_0 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s0_1; a0_0 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s0_2; a0_0 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s0_3;
            a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0; a0_1 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s0_1; a0_1 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s0_2; a0_1 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s0_3;
            a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0; a0_2 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s0_1; a0_2 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s0_2; a0_2 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s0_3;
            a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0; a0_3 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s0_1; a0_3 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s0_2; a0_3 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s0_3;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            let w1_1 = w[(wb1 + wi + 32u32) as usize]; let s1_1 = s[(sb1 + k / 32 + 4u32) as usize];
            let v1_1_0 = dec(w1_1, 0u32); let v1_1_1 = dec(w1_1, 1u32); let v1_1_2 = dec(w1_1, 2u32); let v1_1_3 = dec(w1_1, 3u32);
            let w1_2 = w[(wb1 + wi + 64u32) as usize]; let s1_2 = s[(sb1 + k / 32 + 8u32) as usize];
            let v1_2_0 = dec(w1_2, 0u32); let v1_2_1 = dec(w1_2, 1u32); let v1_2_2 = dec(w1_2, 2u32); let v1_2_3 = dec(w1_2, 3u32);
            let w1_3 = w[(wb1 + wi + 96u32) as usize]; let s1_3 = s[(sb1 + k / 32 + 12u32) as usize];
            let v1_3_0 = dec(w1_3, 0u32); let v1_3_1 = dec(w1_3, 1u32); let v1_3_2 = dec(w1_3, 2u32); let v1_3_3 = dec(w1_3, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0; a1_0 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s1_1; a1_0 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s1_2; a1_0 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s1_3;
            a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0; a1_1 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s1_1; a1_1 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s1_2; a1_1 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s1_3;
            a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0; a1_2 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s1_1; a1_2 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s1_2; a1_2 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s1_3;
            a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0; a1_3 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s1_1; a1_3 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s1_2; a1_3 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s1_3;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            let w2_1 = w[(wb2 + wi + 32u32) as usize]; let s2_1 = s[(sb2 + k / 32 + 4u32) as usize];
            let v2_1_0 = dec(w2_1, 0u32); let v2_1_1 = dec(w2_1, 1u32); let v2_1_2 = dec(w2_1, 2u32); let v2_1_3 = dec(w2_1, 3u32);
            let w2_2 = w[(wb2 + wi + 64u32) as usize]; let s2_2 = s[(sb2 + k / 32 + 8u32) as usize];
            let v2_2_0 = dec(w2_2, 0u32); let v2_2_1 = dec(w2_2, 1u32); let v2_2_2 = dec(w2_2, 2u32); let v2_2_3 = dec(w2_2, 3u32);
            let w2_3 = w[(wb2 + wi + 96u32) as usize]; let s2_3 = s[(sb2 + k / 32 + 12u32) as usize];
            let v2_3_0 = dec(w2_3, 0u32); let v2_3_1 = dec(w2_3, 1u32); let v2_3_2 = dec(w2_3, 2u32); let v2_3_3 = dec(w2_3, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0; a2_0 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s2_1; a2_0 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s2_2; a2_0 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s2_3;
            a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0; a2_1 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s2_1; a2_1 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s2_2; a2_1 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s2_3;
            a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0; a2_2 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s2_1; a2_2 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s2_2; a2_2 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s2_3;
            a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0; a2_3 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s2_1; a2_3 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s2_2; a2_3 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s2_3;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            let w3_1 = w[(wb3 + wi + 32u32) as usize]; let s3_1 = s[(sb3 + k / 32 + 4u32) as usize];
            let v3_1_0 = dec(w3_1, 0u32); let v3_1_1 = dec(w3_1, 1u32); let v3_1_2 = dec(w3_1, 2u32); let v3_1_3 = dec(w3_1, 3u32);
            let w3_2 = w[(wb3 + wi + 64u32) as usize]; let s3_2 = s[(sb3 + k / 32 + 8u32) as usize];
            let v3_2_0 = dec(w3_2, 0u32); let v3_2_1 = dec(w3_2, 1u32); let v3_2_2 = dec(w3_2, 2u32); let v3_2_3 = dec(w3_2, 3u32);
            let w3_3 = w[(wb3 + wi + 96u32) as usize]; let s3_3 = s[(sb3 + k / 32 + 12u32) as usize];
            let v3_3_0 = dec(w3_3, 0u32); let v3_3_1 = dec(w3_3, 1u32); let v3_3_2 = dec(w3_3, 2u32); let v3_3_3 = dec(w3_3, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0; a3_0 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s3_1; a3_0 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s3_2; a3_0 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s3_3;
            a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0; a3_1 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s3_1; a3_1 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s3_2; a3_1 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s3_3;
            a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0; a3_2 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s3_1; a3_2 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s3_2; a3_2 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s3_3;
            a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0; a3_3 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s3_1; a3_3 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s3_2; a3_3 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s3_3;
            let w4_0 = w[(wb4 + wi + 0u32) as usize]; let s4_0 = s[(sb4 + k / 32 + 0u32) as usize];
            let v4_0_0 = dec(w4_0, 0u32); let v4_0_1 = dec(w4_0, 1u32); let v4_0_2 = dec(w4_0, 2u32); let v4_0_3 = dec(w4_0, 3u32);
            let w4_1 = w[(wb4 + wi + 32u32) as usize]; let s4_1 = s[(sb4 + k / 32 + 4u32) as usize];
            let v4_1_0 = dec(w4_1, 0u32); let v4_1_1 = dec(w4_1, 1u32); let v4_1_2 = dec(w4_1, 2u32); let v4_1_3 = dec(w4_1, 3u32);
            let w4_2 = w[(wb4 + wi + 64u32) as usize]; let s4_2 = s[(sb4 + k / 32 + 8u32) as usize];
            let v4_2_0 = dec(w4_2, 0u32); let v4_2_1 = dec(w4_2, 1u32); let v4_2_2 = dec(w4_2, 2u32); let v4_2_3 = dec(w4_2, 3u32);
            let w4_3 = w[(wb4 + wi + 96u32) as usize]; let s4_3 = s[(sb4 + k / 32 + 12u32) as usize];
            let v4_3_0 = dec(w4_3, 0u32); let v4_3_1 = dec(w4_3, 1u32); let v4_3_2 = dec(w4_3, 2u32); let v4_3_3 = dec(w4_3, 3u32);
            a4_0 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s4_0; a4_0 += dotx(v4_1_0, v4_1_1, v4_1_2, v4_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s4_1; a4_0 += dotx(v4_2_0, v4_2_1, v4_2_2, v4_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s4_2; a4_0 += dotx(v4_3_0, v4_3_1, v4_3_2, v4_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s4_3;
            a4_1 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s4_0; a4_1 += dotx(v4_1_0, v4_1_1, v4_1_2, v4_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s4_1; a4_1 += dotx(v4_2_0, v4_2_1, v4_2_2, v4_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s4_2; a4_1 += dotx(v4_3_0, v4_3_1, v4_3_2, v4_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s4_3;
            a4_2 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s4_0; a4_2 += dotx(v4_1_0, v4_1_1, v4_1_2, v4_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s4_1; a4_2 += dotx(v4_2_0, v4_2_1, v4_2_2, v4_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s4_2; a4_2 += dotx(v4_3_0, v4_3_1, v4_3_2, v4_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s4_3;
            a4_3 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s4_0; a4_3 += dotx(v4_1_0, v4_1_1, v4_1_2, v4_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s4_1; a4_3 += dotx(v4_2_0, v4_2_1, v4_2_2, v4_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s4_2; a4_3 += dotx(v4_3_0, v4_3_1, v4_3_2, v4_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s4_3;
            let w5_0 = w[(wb5 + wi + 0u32) as usize]; let s5_0 = s[(sb5 + k / 32 + 0u32) as usize];
            let v5_0_0 = dec(w5_0, 0u32); let v5_0_1 = dec(w5_0, 1u32); let v5_0_2 = dec(w5_0, 2u32); let v5_0_3 = dec(w5_0, 3u32);
            let w5_1 = w[(wb5 + wi + 32u32) as usize]; let s5_1 = s[(sb5 + k / 32 + 4u32) as usize];
            let v5_1_0 = dec(w5_1, 0u32); let v5_1_1 = dec(w5_1, 1u32); let v5_1_2 = dec(w5_1, 2u32); let v5_1_3 = dec(w5_1, 3u32);
            let w5_2 = w[(wb5 + wi + 64u32) as usize]; let s5_2 = s[(sb5 + k / 32 + 8u32) as usize];
            let v5_2_0 = dec(w5_2, 0u32); let v5_2_1 = dec(w5_2, 1u32); let v5_2_2 = dec(w5_2, 2u32); let v5_2_3 = dec(w5_2, 3u32);
            let w5_3 = w[(wb5 + wi + 96u32) as usize]; let s5_3 = s[(sb5 + k / 32 + 12u32) as usize];
            let v5_3_0 = dec(w5_3, 0u32); let v5_3_1 = dec(w5_3, 1u32); let v5_3_2 = dec(w5_3, 2u32); let v5_3_3 = dec(w5_3, 3u32);
            a5_0 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s5_0; a5_0 += dotx(v5_1_0, v5_1_1, v5_1_2, v5_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s5_1; a5_0 += dotx(v5_2_0, v5_2_1, v5_2_2, v5_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s5_2; a5_0 += dotx(v5_3_0, v5_3_1, v5_3_2, v5_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s5_3;
            a5_1 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s5_0; a5_1 += dotx(v5_1_0, v5_1_1, v5_1_2, v5_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s5_1; a5_1 += dotx(v5_2_0, v5_2_1, v5_2_2, v5_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s5_2; a5_1 += dotx(v5_3_0, v5_3_1, v5_3_2, v5_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s5_3;
            a5_2 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s5_0; a5_2 += dotx(v5_1_0, v5_1_1, v5_1_2, v5_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s5_1; a5_2 += dotx(v5_2_0, v5_2_1, v5_2_2, v5_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s5_2; a5_2 += dotx(v5_3_0, v5_3_1, v5_3_2, v5_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s5_3;
            a5_3 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s5_0; a5_3 += dotx(v5_1_0, v5_1_1, v5_1_2, v5_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s5_1; a5_3 += dotx(v5_2_0, v5_2_1, v5_2_2, v5_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s5_2; a5_3 += dotx(v5_3_0, v5_3_1, v5_3_2, v5_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s5_3;
            let w6_0 = w[(wb6 + wi + 0u32) as usize]; let s6_0 = s[(sb6 + k / 32 + 0u32) as usize];
            let v6_0_0 = dec(w6_0, 0u32); let v6_0_1 = dec(w6_0, 1u32); let v6_0_2 = dec(w6_0, 2u32); let v6_0_3 = dec(w6_0, 3u32);
            let w6_1 = w[(wb6 + wi + 32u32) as usize]; let s6_1 = s[(sb6 + k / 32 + 4u32) as usize];
            let v6_1_0 = dec(w6_1, 0u32); let v6_1_1 = dec(w6_1, 1u32); let v6_1_2 = dec(w6_1, 2u32); let v6_1_3 = dec(w6_1, 3u32);
            let w6_2 = w[(wb6 + wi + 64u32) as usize]; let s6_2 = s[(sb6 + k / 32 + 8u32) as usize];
            let v6_2_0 = dec(w6_2, 0u32); let v6_2_1 = dec(w6_2, 1u32); let v6_2_2 = dec(w6_2, 2u32); let v6_2_3 = dec(w6_2, 3u32);
            let w6_3 = w[(wb6 + wi + 96u32) as usize]; let s6_3 = s[(sb6 + k / 32 + 12u32) as usize];
            let v6_3_0 = dec(w6_3, 0u32); let v6_3_1 = dec(w6_3, 1u32); let v6_3_2 = dec(w6_3, 2u32); let v6_3_3 = dec(w6_3, 3u32);
            a6_0 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s6_0; a6_0 += dotx(v6_1_0, v6_1_1, v6_1_2, v6_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s6_1; a6_0 += dotx(v6_2_0, v6_2_1, v6_2_2, v6_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s6_2; a6_0 += dotx(v6_3_0, v6_3_1, v6_3_2, v6_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s6_3;
            a6_1 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s6_0; a6_1 += dotx(v6_1_0, v6_1_1, v6_1_2, v6_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s6_1; a6_1 += dotx(v6_2_0, v6_2_1, v6_2_2, v6_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s6_2; a6_1 += dotx(v6_3_0, v6_3_1, v6_3_2, v6_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s6_3;
            a6_2 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s6_0; a6_2 += dotx(v6_1_0, v6_1_1, v6_1_2, v6_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s6_1; a6_2 += dotx(v6_2_0, v6_2_1, v6_2_2, v6_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s6_2; a6_2 += dotx(v6_3_0, v6_3_1, v6_3_2, v6_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s6_3;
            a6_3 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s6_0; a6_3 += dotx(v6_1_0, v6_1_1, v6_1_2, v6_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s6_1; a6_3 += dotx(v6_2_0, v6_2_1, v6_2_2, v6_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s6_2; a6_3 += dotx(v6_3_0, v6_3_1, v6_3_2, v6_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s6_3;
            let w7_0 = w[(wb7 + wi + 0u32) as usize]; let s7_0 = s[(sb7 + k / 32 + 0u32) as usize];
            let v7_0_0 = dec(w7_0, 0u32); let v7_0_1 = dec(w7_0, 1u32); let v7_0_2 = dec(w7_0, 2u32); let v7_0_3 = dec(w7_0, 3u32);
            let w7_1 = w[(wb7 + wi + 32u32) as usize]; let s7_1 = s[(sb7 + k / 32 + 4u32) as usize];
            let v7_1_0 = dec(w7_1, 0u32); let v7_1_1 = dec(w7_1, 1u32); let v7_1_2 = dec(w7_1, 2u32); let v7_1_3 = dec(w7_1, 3u32);
            let w7_2 = w[(wb7 + wi + 64u32) as usize]; let s7_2 = s[(sb7 + k / 32 + 8u32) as usize];
            let v7_2_0 = dec(w7_2, 0u32); let v7_2_1 = dec(w7_2, 1u32); let v7_2_2 = dec(w7_2, 2u32); let v7_2_3 = dec(w7_2, 3u32);
            let w7_3 = w[(wb7 + wi + 96u32) as usize]; let s7_3 = s[(sb7 + k / 32 + 12u32) as usize];
            let v7_3_0 = dec(w7_3, 0u32); let v7_3_1 = dec(w7_3, 1u32); let v7_3_2 = dec(w7_3, 2u32); let v7_3_3 = dec(w7_3, 3u32);
            a7_0 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s7_0; a7_0 += dotx(v7_1_0, v7_1_1, v7_1_2, v7_1_3, x0_1_0, x0_1_1, x0_1_2, x0_1_3) * s7_1; a7_0 += dotx(v7_2_0, v7_2_1, v7_2_2, v7_2_3, x0_2_0, x0_2_1, x0_2_2, x0_2_3) * s7_2; a7_0 += dotx(v7_3_0, v7_3_1, v7_3_2, v7_3_3, x0_3_0, x0_3_1, x0_3_2, x0_3_3) * s7_3;
            a7_1 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s7_0; a7_1 += dotx(v7_1_0, v7_1_1, v7_1_2, v7_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s7_1; a7_1 += dotx(v7_2_0, v7_2_1, v7_2_2, v7_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s7_2; a7_1 += dotx(v7_3_0, v7_3_1, v7_3_2, v7_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s7_3;
            a7_2 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s7_0; a7_2 += dotx(v7_1_0, v7_1_1, v7_1_2, v7_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s7_1; a7_2 += dotx(v7_2_0, v7_2_1, v7_2_2, v7_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s7_2; a7_2 += dotx(v7_3_0, v7_3_1, v7_3_2, v7_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s7_3;
            a7_3 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s7_0; a7_3 += dotx(v7_1_0, v7_1_1, v7_1_2, v7_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s7_1; a7_3 += dotx(v7_2_0, v7_2_1, v7_2_2, v7_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s7_2; a7_3 += dotx(v7_3_0, v7_3_1, v7_3_2, v7_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s7_3;
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let x0_0_0 = x[(xb0 + k + 0u32) as usize]; let x0_0_1 = x[(xb0 + k + 1u32) as usize]; let x0_0_2 = x[(xb0 + k + 2u32) as usize]; let x0_0_3 = x[(xb0 + k + 3u32) as usize];
            let x1_0_0 = x[(xb1 + k + 0u32) as usize]; let x1_0_1 = x[(xb1 + k + 1u32) as usize]; let x1_0_2 = x[(xb1 + k + 2u32) as usize]; let x1_0_3 = x[(xb1 + k + 3u32) as usize];
            let x2_0_0 = x[(xb2 + k + 0u32) as usize]; let x2_0_1 = x[(xb2 + k + 1u32) as usize]; let x2_0_2 = x[(xb2 + k + 2u32) as usize]; let x2_0_3 = x[(xb2 + k + 3u32) as usize];
            let x3_0_0 = x[(xb3 + k + 0u32) as usize]; let x3_0_1 = x[(xb3 + k + 1u32) as usize]; let x3_0_2 = x[(xb3 + k + 2u32) as usize]; let x3_0_3 = x[(xb3 + k + 3u32) as usize];
            let w0_0 = w[(wb0 + wi + 0u32) as usize]; let s0_0 = s[(sb0 + k / 32 + 0u32) as usize];
            let v0_0_0 = dec(w0_0, 0u32); let v0_0_1 = dec(w0_0, 1u32); let v0_0_2 = dec(w0_0, 2u32); let v0_0_3 = dec(w0_0, 3u32);
            a0_0 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s0_0;
            a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0;
            a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0;
            a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0;
            let w1_0 = w[(wb1 + wi + 0u32) as usize]; let s1_0 = s[(sb1 + k / 32 + 0u32) as usize];
            let v1_0_0 = dec(w1_0, 0u32); let v1_0_1 = dec(w1_0, 1u32); let v1_0_2 = dec(w1_0, 2u32); let v1_0_3 = dec(w1_0, 3u32);
            a1_0 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s1_0;
            a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0;
            a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0;
            a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0;
            let w2_0 = w[(wb2 + wi + 0u32) as usize]; let s2_0 = s[(sb2 + k / 32 + 0u32) as usize];
            let v2_0_0 = dec(w2_0, 0u32); let v2_0_1 = dec(w2_0, 1u32); let v2_0_2 = dec(w2_0, 2u32); let v2_0_3 = dec(w2_0, 3u32);
            a2_0 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s2_0;
            a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0;
            a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0;
            a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0;
            let w3_0 = w[(wb3 + wi + 0u32) as usize]; let s3_0 = s[(sb3 + k / 32 + 0u32) as usize];
            let v3_0_0 = dec(w3_0, 0u32); let v3_0_1 = dec(w3_0, 1u32); let v3_0_2 = dec(w3_0, 2u32); let v3_0_3 = dec(w3_0, 3u32);
            a3_0 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s3_0;
            a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0;
            a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0;
            a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0;
            let w4_0 = w[(wb4 + wi + 0u32) as usize]; let s4_0 = s[(sb4 + k / 32 + 0u32) as usize];
            let v4_0_0 = dec(w4_0, 0u32); let v4_0_1 = dec(w4_0, 1u32); let v4_0_2 = dec(w4_0, 2u32); let v4_0_3 = dec(w4_0, 3u32);
            a4_0 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s4_0;
            a4_1 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s4_0;
            a4_2 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s4_0;
            a4_3 += dotx(v4_0_0, v4_0_1, v4_0_2, v4_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s4_0;
            let w5_0 = w[(wb5 + wi + 0u32) as usize]; let s5_0 = s[(sb5 + k / 32 + 0u32) as usize];
            let v5_0_0 = dec(w5_0, 0u32); let v5_0_1 = dec(w5_0, 1u32); let v5_0_2 = dec(w5_0, 2u32); let v5_0_3 = dec(w5_0, 3u32);
            a5_0 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s5_0;
            a5_1 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s5_0;
            a5_2 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s5_0;
            a5_3 += dotx(v5_0_0, v5_0_1, v5_0_2, v5_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s5_0;
            let w6_0 = w[(wb6 + wi + 0u32) as usize]; let s6_0 = s[(sb6 + k / 32 + 0u32) as usize];
            let v6_0_0 = dec(w6_0, 0u32); let v6_0_1 = dec(w6_0, 1u32); let v6_0_2 = dec(w6_0, 2u32); let v6_0_3 = dec(w6_0, 3u32);
            a6_0 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s6_0;
            a6_1 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s6_0;
            a6_2 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s6_0;
            a6_3 += dotx(v6_0_0, v6_0_1, v6_0_2, v6_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s6_0;
            let w7_0 = w[(wb7 + wi + 0u32) as usize]; let s7_0 = s[(sb7 + k / 32 + 0u32) as usize];
            let v7_0_0 = dec(w7_0, 0u32); let v7_0_1 = dec(w7_0, 1u32); let v7_0_2 = dec(w7_0, 2u32); let v7_0_3 = dec(w7_0, 3u32);
            a7_0 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x0_0_0, x0_0_1, x0_0_2, x0_0_3) * s7_0;
            a7_1 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s7_0;
            a7_2 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s7_0;
            a7_3 += dotx(v7_0_0, v7_0_1, v7_0_2, v7_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s7_0;
            wi += 32;
        }
        let t0_0 = plane_sum(a0_0); let t0_1 = plane_sum(a0_1); let t0_2 = plane_sum(a0_2); let t0_3 = plane_sum(a0_3);
        let t1_0 = plane_sum(a1_0); let t1_1 = plane_sum(a1_1); let t1_2 = plane_sum(a1_2); let t1_3 = plane_sum(a1_3);
        let t2_0 = plane_sum(a2_0); let t2_1 = plane_sum(a2_1); let t2_2 = plane_sum(a2_2); let t2_3 = plane_sum(a2_3);
        let t3_0 = plane_sum(a3_0); let t3_1 = plane_sum(a3_1); let t3_2 = plane_sum(a3_2); let t3_3 = plane_sum(a3_3);
        let t4_0 = plane_sum(a4_0); let t4_1 = plane_sum(a4_1); let t4_2 = plane_sum(a4_2); let t4_3 = plane_sum(a4_3);
        let t5_0 = plane_sum(a5_0); let t5_1 = plane_sum(a5_1); let t5_2 = plane_sum(a5_2); let t5_3 = plane_sum(a5_3);
        let t6_0 = plane_sum(a6_0); let t6_1 = plane_sum(a6_1); let t6_2 = plane_sum(a6_2); let t6_3 = plane_sum(a6_3);
        let t7_0 = plane_sum(a7_0); let t7_1 = plane_sum(a7_1); let t7_2 = plane_sum(a7_2); let t7_3 = plane_sum(a7_3);
        if UNIT_POS_X == 0 {
            if row0 + 0u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 0u32) as usize] = t0_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 0u32) as usize] = t0_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 0u32) as usize] = t0_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 0u32) as usize] = t0_3; }
            }
            if row0 + 1u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 1u32) as usize] = t1_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 1u32) as usize] = t1_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 1u32) as usize] = t1_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 1u32) as usize] = t1_3; }
            }
            if row0 + 2u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 2u32) as usize] = t2_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 2u32) as usize] = t2_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 2u32) as usize] = t2_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 2u32) as usize] = t2_3; }
            }
            if row0 + 3u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 3u32) as usize] = t3_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 3u32) as usize] = t3_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 3u32) as usize] = t3_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 3u32) as usize] = t3_3; }
            }
            if row0 + 4u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 4u32) as usize] = t4_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 4u32) as usize] = t4_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 4u32) as usize] = t4_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 4u32) as usize] = t4_3; }
            }
            if row0 + 5u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 5u32) as usize] = t5_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 5u32) as usize] = t5_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 5u32) as usize] = t5_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 5u32) as usize] = t5_3; }
            }
            if row0 + 6u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 6u32) as usize] = t6_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 6u32) as usize] = t6_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 6u32) as usize] = t6_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 6u32) as usize] = t6_3; }
            }
            if row0 + 7u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 7u32) as usize] = t7_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 7u32) as usize] = t7_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 7u32) as usize] = t7_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 7u32) as usize] = t7_3; }
            }
        }
    }
}

/// k_gemv_iq4d with the activation quantised inside each cube (no separate k_q8pack launch or scratch):
/// pass 1 one 32-block scale per thread, pass 2 int8 words, both in shared memory. cols <= 6144.
#[cube(launch_unchecked)]
fn k_gemv_iq4x(x: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    let mut xq = Shared::<[u32]>::new_slice(1536usize);
    let mut xs = Shared::<[f32]>::new_slice(192usize);
    let mut xi = Shared::<[f32]>::new_slice(192usize);
    let nb32 = cols / 32;
    let mut b = UNIT_POS;
    while b < nb32 {
        let mut amax = 0.0f32;
        for i in 0..32u32 { amax = f32::max(amax, f32::abs(x[(xoff + b * 32 + i) as usize])); }
        let dd = amax / 127.0f32;
        let mut id = 0.0f32;
        if dd > 0.0f32 { id = 1.0f32 / dd; }
        xs[b as usize] = dd;
        xi[b as usize] = id;
        b += CUBE_DIM;
    }
    sync_cube();
    let nw = cols / 4;
    let mut wq = UNIT_POS;
    while wq < nw {
        let id = xi[(wq / 8) as usize];
        let mut word = 0u32;
        #[unroll]
        for j in 0..4u32 {
            let v = i32::cast_from(f32::round(x[(xoff + wq * 4 + j) as usize] * id));
            word |= (u32::cast_from(v) & 255u32) << (8 * j);
        }
        xq[wq as usize] = word;
        wq += CUBE_DIM;
    }
    sync_cube();
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let mut acc = 0.0f32;
        let mut blk = 0u32;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let d = f32::reinterpret(w[b0 as usize]);
            let sh = w[(b0 + 1) as usize];
            let sl = w[(b0 + 2) as usize];
            let q = w[(b0 + 3 + UNIT_POS_X) as usize];
            let xw = blk * 64 + ib * 8 + k;
            let xa = i32::cast_from(xq[xw as usize]);
            let xb = i32::cast_from(xq[(xw + 4) as usize]);
            let xsc = xs[(blk * 8 + ib) as usize];
            let ls = i32::cast_from(((sl >> (4 * ib)) & 15u32) | (((sh >> (2 * ib)) & 3u32) << 4)) - 32i32;
            let u0 = t2[(q & 255u32) as usize];
            let u1 = t2[((q >> 8) & 255u32) as usize];
            let u2 = t2[((q >> 16) & 255u32) as usize];
            let u3 = t2[(q >> 24) as usize];
            let lo = (u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24);
            let hi = ((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24);
            let p = dp4a(i32::cast_from(hi), xb, dp4a(i32::cast_from(lo), xa, 0i32));
            acc += d * f32::cast_from(ls) * xsc * f32::cast_from(p);
            blk += 1u32;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}

/// k_q8pack for `nt` columns at once (column c read at soff + c*sst, written at c*nb blocks): the same per-block
/// arithmetic as k_q8pack, one launch instead of nt.
#[cube(launch_unchecked)]
fn k_q8pack_m(src: &[f32], q: &mut [u32], sc: &mut [f32], soff: u32, sst: u32, nb: u32) {
    let b = ABSOLUTE_POS_X;
    let c = CUBE_POS_Y;
    if b < nb {
        let base = soff + c * sst + b * 32;
        let mut amax = 0.0f32;
        for i in 0..32u32 { amax = f32::max(amax, f32::abs(src[(base + i) as usize])); }
        let d = amax / 127.0f32;
        let mut id = 0.0f32;
        if d > 0.0f32 { id = 1.0f32 / d; }
        sc[(c * nb + b) as usize] = d;
        for w in 0..8u32 {
            let mut word = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let v = i32::cast_from(f32::round(src[(base + w * 4 + j) as usize] * id));
                word |= (u32::cast_from(v) & 255u32) << (8 * j);
            }
            q[(c * nb * 8 + b * 8 + w) as usize] = word;
        }
    }
}

/// k_gemv_iq4d for up to 4 activation columns (packed by k_q8pack_m): each weight word is decoded once (levels
/// table in shared memory) and used for every column; per column the arithmetic and its order are k_gemv_iq4d's.
#[cube(launch_unchecked)]
fn k_gemv_iq4d_m(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let mut tb = Shared::<[u32]>::new_slice(256usize);
    let mut ti = UNIT_POS_Y * 32 + UNIT_POS_X;
    while ti < 256u32 { tb[ti as usize] = t2[ti as usize]; ti += CUBE_DIM_X * CUBE_DIM_Y; }
    sync_cube();
    if row < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let cq = cols / 4;
        let cs = cols / 32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut blk = 0u32;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let d = f32::reinterpret(w[b0 as usize]);
            let sh = w[(b0 + 1) as usize];
            let sl = w[(b0 + 2) as usize];
            let q = w[(b0 + 3 + UNIT_POS_X) as usize];
            let xw = blk * 64 + ib * 8 + k;
            let ls = i32::cast_from(((sl >> (4 * ib)) & 15u32) | (((sh >> (2 * ib)) & 3u32) << 4)) - 32i32;
            let u0 = tb[(q & 255u32) as usize];
            let u1 = tb[((q >> 8) & 255u32) as usize];
            let u2 = tb[((q >> 16) & 255u32) as usize];
            let u3 = tb[(q >> 24) as usize];
            let lo = i32::cast_from((u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24));
            let hi = i32::cast_from(((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24));
            let dl = d * f32::cast_from(ls);
            let xi = blk * 8 + ib;
            let p0 = dp4a(hi, i32::cast_from(xq[(xw + 4) as usize]), dp4a(lo, i32::cast_from(xq[xw as usize]), 0i32));
            a0 += dl * xs[xi as usize] * f32::cast_from(p0);
            if nt > 1u32 {
                let p1 = dp4a(hi, i32::cast_from(xq[(cq + xw + 4) as usize]), dp4a(lo, i32::cast_from(xq[(cq + xw) as usize]), 0i32));
                a1 += dl * xs[(cs + xi) as usize] * f32::cast_from(p1);
            }
            if nt > 2u32 {
                let p2 = dp4a(hi, i32::cast_from(xq[(2 * cq + xw + 4) as usize]), dp4a(lo, i32::cast_from(xq[(2 * cq + xw) as usize]), 0i32));
                a2 += dl * xs[(2 * cs + xi) as usize] * f32::cast_from(p2);
            }
            if nt > 3u32 {
                let p3 = dp4a(hi, i32::cast_from(xq[(3 * cq + xw + 4) as usize]), dp4a(lo, i32::cast_from(xq[(3 * cq + xw) as usize]), 0i32));
                a3 += dl * xs[(3 * cs + xi) as usize] * f32::cast_from(p3);
            }
            blk += 1u32;
        }
        let t0 = plane_sum(a0);
        let t1 = plane_sum(a1);
        let t2s = plane_sum(a2);
        let t3 = plane_sum(a3);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t0;
            if nt > 1u32 { out[(ooff + ost + row) as usize] = t1; }
            if nt > 2u32 { out[(ooff + 2 * ost + row) as usize] = t2s; }
            if nt > 3u32 { out[(ooff + 3 * ost + row) as usize] = t3; }
        }
    }
}

/// k_gemv_iq4d_m with 4 rows per warp: each lane loads its activation words once per block and reuses them for
/// 4 weight rows (the per-row x loads dominated). Per row and column the arithmetic and its order are k_gemv_iq4d_m's.
#[cube(launch_unchecked)]
fn k_gemv_iq4d_r(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 4u32;
    let mut tb = Shared::<[u32]>::new_slice(256usize);
    let mut ti = UNIT_POS_Y * 32 + UNIT_POS_X;
    while ti < 256u32 { tb[ti as usize] = t2[ti as usize]; ti += CUBE_DIM_X * CUBE_DIM_Y; }
    sync_cube();
    if row0 < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let cq = cols / 4;
        let cs = cols / 32;
        let mut acc = Array::<f32>::new(16usize);
        #[unroll]
        for i in 0..16u32 { acc[i as usize] = 0.0f32; }
        let mut xl = Array::<i32>::new(4usize);
        let mut xh = Array::<i32>::new(4usize);
        let mut xv = Array::<f32>::new(4usize);
        let mut blk = 0u32;
        while blk < nb {
            let xw = blk * 64 + ib * 8 + k;
            let xi = blk * 8 + ib;
            #[unroll]
            for c in 0..4u32 {
                if c < nt {
                    xl[c as usize] = i32::cast_from(xq[(c * cq + xw) as usize]);
                    xh[c as usize] = i32::cast_from(xq[(c * cq + xw + 4) as usize]);
                    xv[c as usize] = xs[(c * cs + xi) as usize];
                }
            }
            #[unroll]
            for r in 0..4u32 {
                let rr = row0 + r;
                if rr < rows {
                    let b0 = rr * rw + blk * 35;
                    let d = f32::reinterpret(w[b0 as usize]);
                    let sh = w[(b0 + 1) as usize];
                    let sl = w[(b0 + 2) as usize];
                    let q = w[(b0 + 3 + UNIT_POS_X) as usize];
                    let ls = i32::cast_from(((sl >> (4 * ib)) & 15u32) | (((sh >> (2 * ib)) & 3u32) << 4)) - 32i32;
                    let u0 = tb[(q & 255u32) as usize];
                    let u1 = tb[((q >> 8) & 255u32) as usize];
                    let u2 = tb[((q >> 16) & 255u32) as usize];
                    let u3 = tb[(q >> 24) as usize];
                    let lo = i32::cast_from((u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24));
                    let hi = i32::cast_from(((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24));
                    let dl = d * f32::cast_from(ls);
                    #[unroll]
                    for c in 0..4u32 {
                        if c < nt {
                            let p = dp4a(hi, xh[c as usize], dp4a(lo, xl[c as usize], 0i32));
                            acc[(r * 4 + c) as usize] += dl * xv[c as usize] * f32::cast_from(p);
                        }
                    }
                }
            }
            blk += 1u32;
        }
        #[unroll]
        for r in 0..4u32 {
            #[unroll]
            for c in 0..4u32 {
                let t = plane_sum(acc[(r * 4 + c) as usize]);
                if UNIT_POS_X == 0 && c < nt && row0 + r < rows { out[(ooff + c * ost + row0 + r) as usize] = t; }
            }
        }
    }
}

/// PTX prmt.b32 (CUDA __byte_perm): byte n of the result = byte sel[n] of {b:a}. The selector nibbles are masked
/// to 3 bits, as __byte_perm does (prmt's bit 3 would sign-replicate).
#[cube]
fn prmt(a: u32, b: u32, sel: u32) -> u32 {
    let s7 = sel & 0x7777u32;
    // copies: a bare parameter reached the asm as the address of its slot (LLVM backend)
    let a1 = a | 0u32;
    let b1 = b | 0u32;
    let d: u32;
    gpu_asm!("prmt.b32 {d}, {a}, {b}, {s};", a = in(_) a1, b = in(_) b1, s = in(_) s7, d = out(_) d, options(pure, nomem));
    d
}

/// The IQ4_NL levels of the 8 nibbles of q4 as (low nibbles, high nibbles) int8x4, register-only: Strata's
/// iq4_table_lookup (MIT, after llama.cpp's get_int_from_table_16), 6 byte permutes per word.
#[cube]
fn iq4_lookup(q4: u32) -> (u32, u32) {
    let t0 = 0xbfad9881u32; let t1 = 0xf6eaddcfu32; let t2 = 0x26190d01u32; let t3 = 0x71594535u32;
    let sel = 0x32103210u32 | ((q4 & 0x88888888u32) >> 1);
    let l0 = prmt(t0, t1, q4);
    let h0 = prmt(t2, t3, q4);
    let m0 = prmt(l0, h0, sel);
    let l1 = prmt(t0, t1, q4 >> 16);
    let h1 = prmt(t2, t3, q4 >> 16);
    let m1 = prmt(l1, h1, sel >> 16);
    (prmt(m0, m1, 0x6420u32), prmt(m0, m1, 0x7531u32))
}

/// IQ4_XS x int8 GEMV for up to 4 columns, after Strata's native_iq4_xs_mmvq / iq4_xs_q8_dot (MIT, from
/// llama.cpp's vec_dot_iq4_xs_q8_1): a lane takes one 32-value sub-block (4 qs words), sums it in the integer
/// domain with dp4a, applies the signed 6-bit scale as an integer, then one float multiply (d_w * d_x) * sumi.
/// 8 lanes per 256-block, a warp per row, 4 blocks per warp step. Levels via a 256-entry shared byte table.
#[cube(launch_unchecked)]
fn k_gemv_iq4s_m(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X % 8;
        let cq = cols / 4;
        let cs = cols / 32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut blk = UNIT_POS_X / 8;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let d = f32::reinterpret(w[b0 as usize]);
            let sh = w[(b0 + 1) as usize];
            let sl = w[(b0 + 2) as usize];
            let ls = i32::cast_from(((sl >> (4 * ib)) & 15u32) | (((sh >> (2 * ib)) & 3u32) << 4)) - 32i32;
            let xw = blk * 64 + ib * 8;
            let mut s0 = 0i32;
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            let mut s3 = 0i32;
            #[unroll]
            for j in 0..4u32 {
                let q = w[(b0 + 3 + ib * 4 + j) as usize];
                let (lu, hu) = iq4_lookup(q);
                let lo = i32::cast_from(lu);
                let hi = i32::cast_from(hu);
                s0 = dp4a(hi, i32::cast_from(xq[(xw + 4 + j) as usize]), dp4a(lo, i32::cast_from(xq[(xw + j) as usize]), s0));
                if nt > 1u32 { s1 = dp4a(hi, i32::cast_from(xq[(cq + xw + 4 + j) as usize]), dp4a(lo, i32::cast_from(xq[(cq + xw + j) as usize]), s1)); }
                if nt > 2u32 { s2 = dp4a(hi, i32::cast_from(xq[(2 * cq + xw + 4 + j) as usize]), dp4a(lo, i32::cast_from(xq[(2 * cq + xw + j) as usize]), s2)); }
                if nt > 3u32 { s3 = dp4a(hi, i32::cast_from(xq[(3 * cq + xw + 4 + j) as usize]), dp4a(lo, i32::cast_from(xq[(3 * cq + xw + j) as usize]), s3)); }
            }
            let xi = blk * 8 + ib;
            a0 += (d * xs[xi as usize]) * f32::cast_from(s0 * ls);
            if nt > 1u32 { a1 += (d * xs[(cs + xi) as usize]) * f32::cast_from(s1 * ls); }
            if nt > 2u32 { a2 += (d * xs[(2 * cs + xi) as usize]) * f32::cast_from(s2 * ls); }
            if nt > 3u32 { a3 += (d * xs[(3 * cs + xi) as usize]) * f32::cast_from(s3 * ls); }
            blk += 4u32;
        }
        let t0 = plane_sum(a0);
        let t1 = plane_sum(a1);
        let t2s = plane_sum(a2);
        let t3 = plane_sum(a3);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t0;
            if nt > 1u32 { out[(ooff + ost + row) as usize] = t1; }
            if nt > 2u32 { out[(ooff + 2 * ost + row) as usize] = t2s; }
            if nt > 3u32 { out[(ooff + 3 * ost + row) as usize] = t3; }
        }
    }
}

/// IQ4_XS GEMV with int8 activations (x packed by k_q8pack) and dp4a. t2[b] = IQ4_NL level of the low
/// nibble of byte b in bits 0..8 and of the high nibble in bits 8..16, so one lookup per weight byte.
#[cube(launch_unchecked)]
fn k_gemv_iq4d(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let mut acc = 0.0f32;
        let mut blk = 0u32;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let d = f32::reinterpret(w[b0 as usize]);
            let sh = w[(b0 + 1) as usize];
            let sl = w[(b0 + 2) as usize];
            let q = w[(b0 + 3 + UNIT_POS_X) as usize];
            let xw = blk * 64 + ib * 8 + k;
            let xa = i32::cast_from(xq[xw as usize]);
            let xb = i32::cast_from(xq[(xw + 4) as usize]);
            let xsc = xs[(blk * 8 + ib) as usize];
            let ls = i32::cast_from(((sl >> (4 * ib)) & 15u32) | (((sh >> (2 * ib)) & 3u32) << 4)) - 32i32;
            let u0 = t2[(q & 255u32) as usize];
            let u1 = t2[((q >> 8) & 255u32) as usize];
            let u2 = t2[((q >> 16) & 255u32) as usize];
            let u3 = t2[(q >> 24) as usize];
            let lo = (u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24);
            let hi = ((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24);
            let p = dp4a(i32::cast_from(hi), xb, dp4a(i32::cast_from(lo), xa, 0i32));
            acc += d * f32::cast_from(ls) * xsc * f32::cast_from(p);
            blk += 1u32;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}

/// GEMV straight from IQ4_XS blocks (repacked: f32 d, scales_h, scales_l, 32 words of nibbles =
/// 35 words per 256). Each lane takes one qs word: 4 low nibbles (x[32ib+4k..]) and 4 high
/// nibbles (x[32ib+16+4k..]), so adjacent lanes read adjacent activations. kv = the 16 IQ4_NL levels.
#[cube(launch_unchecked)]
fn k_gemv_iq4(x: &[f32], w: &[u32], kv: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    // x staged in shared memory, padded by one float per 32 so the (sub-block, word) lane layout
    // below is (nearly) bank-conflict free; the 8 rows of a cube share it.
    let mut xs = Shared::<[f32]>::new_slice(6336usize);
    let mut tab = Shared::<[f32]>::new_slice(16usize);
    let mut i = UNIT_POS;
    while i < cols {
        xs[(i + i / 32) as usize] = x[(xoff + i) as usize];
        i += CUBE_DIM;
    }
    if UNIT_POS < 16 {
        tab[UNIT_POS as usize] = kv[UNIT_POS as usize];
    }
    sync_cube();
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let rw = (cols / 256) * 35;
        let nb = cols / 256;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let mut acc = 0.0f32;
        // two blocks per step: all eight global loads issued before any use
        let mut blk = 0u32;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let b1 = b0 + 35;
            let d0 = f32::reinterpret(w[b0 as usize]);
            let sh0 = w[(b0 + 1) as usize];
            let sl0 = w[(b0 + 2) as usize];
            let q0 = w[(b0 + 3 + UNIT_POS_X) as usize];
            let mut d1 = 0.0f32;
            let mut sh1 = 0u32;
            let mut sl1 = 0u32;
            let mut q1 = 0u32;
            if blk + 1 < nb {
                d1 = f32::reinterpret(w[b1 as usize]);
                sh1 = w[(b1 + 1) as usize];
                sl1 = w[(b1 + 2) as usize];
                q1 = w[(b1 + 3 + UNIT_POS_X) as usize];
            }
            let ls0 = ((sl0 >> (4 * ib)) & 15u32) | (((sh0 >> (2 * ib)) & 3u32) << 4);
            let ls1 = ((sl1 >> (4 * ib)) & 15u32) | (((sh1 >> (2 * ib)) & 3u32) << 4);
            let p0 = blk * 264 + ib * 33 + 4 * k;
            let p1 = p0 + 264;
            let mut part0 = 0.0f32;
            let mut part1 = 0.0f32;
            #[unroll]
            for j in 0..4u32 {
                part0 += tab[((q0 >> (8 * j)) & 15u32) as usize] * xs[(p0 + j) as usize];
                part0 += tab[((q0 >> (8 * j + 4)) & 15u32) as usize] * xs[(p0 + 16 + j) as usize];
            }
            acc += part0 * (d0 * (f32::cast_from(ls0) - 32.0f32));
            if blk + 1 < nb {
                #[unroll]
                for j in 0..4u32 {
                    part1 += tab[((q1 >> (8 * j)) & 15u32) as usize] * xs[(p1 + j) as usize];
                    part1 += tab[((q1 >> (8 * j + 4)) & 15u32) as usize] * xs[(p1 + 16 + j) as usize];
                }
                acc += part1 * (d1 * (f32::cast_from(ls1) - 32.0f32));
            }
            blk += 2u32;
        }        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}
/// One cube (256 units) per stream: xn = x * rsqrt(mean(x^2) + eps) * norm.
/// k_rms for up to 4 tokens in one launch (cube (stream, token)); each cube does exactly k_rms's operations in its
/// order, so the bits are k_rms's (Strata's split norm, src/kernels/cuda/fused_gr.cu: one block per token AND stream).
#[cube(launch_unchecked)]
fn k_rms4(x0: &[f32], x1: &[f32], x2: &[f32], x3: &[f32], norm: &[f32], xn: &mut [f32], d: u32, w: u32, eps: f32) {
    let base = CUBE_POS_X * d; let tk = CUBE_POS_Y; let ob = tk * w;
    let mut acc = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d {
        let j = (base + i) as usize;
        let mut v = x0[j]; if tk == 1u32 { v = x1[j]; } if tk == 2u32 { v = x2[j]; } if tk == 3u32 { v = x3[j]; }
        acc += v * v;
        i += CUBE_DIM_X;
    }
    let p = plane_sum(acc);
    let mut sh = Shared::<[f32]>::new_slice(8usize);
    if UNIT_POS_X % 32 == 0 {
        sh[(UNIT_POS_X / 32) as usize] = p;
    }
    sync_cube();
    let mut tot = 0.0f32;
    for j in 0..8u32 {
        tot += sh[j as usize];
    }
    let r = 1.0f32 / f32::sqrt(tot / f32::cast_from(d) + eps);
    let mut i2 = UNIT_POS_X;
    while i2 < d {
        let k = (base + i2) as usize;
        let mut v = x0[k]; if tk == 1u32 { v = x1[k]; } if tk == 2u32 { v = x2[k]; } if tk == 3u32 { v = x3[k]; }
        xn[(ob + base + i2) as usize] = v * r * norm[k];
        i2 += CUBE_DIM_X;
    }
}#[cube(launch_unchecked)]
fn k_rms(x: &[f32], norm: &[f32], xn: &mut [f32], d: u32, eps: f32) {
    let base = CUBE_POS_X * d;
    let mut acc = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d {
        let v = x[(base + i) as usize];
        acc += v * v;
        i += CUBE_DIM_X;
    }
    let p = plane_sum(acc);
    let mut sh = Shared::<[f32]>::new_slice(8usize);
    if UNIT_POS_X % 32 == 0 {
        sh[(UNIT_POS_X / 32) as usize] = p;
    }
    sync_cube();
    let mut tot = 0.0f32;
    for j in 0..8u32 {
        tot += sh[j as usize];
    }
    let r = 1.0f32 / f32::sqrt(tot / f32::cast_from(d) + eps);
    let mut i2 = UNIT_POS_X;
    while i2 < d {
        let k = (base + i2) as usize;
        xn[k] = x[k] * r * norm[k];
        i2 += CUBE_DIM_X;
    }
}

/// k_gate for t tokens of a verify window (comptime t <= 4): each weight word is loaded once and used for
/// every token; each token's arithmetic is the same, in the same order, as k_gate (identical results).
/// Token j reads xn[j*xs..], dn[j*ds..] and writes out[j*os + ooff + i].
#[cube(launch_unchecked)]
fn k_gate_m(xn: &[f32], dn: &[f32], w: &[u32], s: &[f32], out: &mut [f32], d: u32, ooff: u32, xs: u32, ds: u32, os: u32, #[comptime] lr: usize, #[comptime] hc: u32, #[comptime] t: u32) {
    let mut lo = Shared::<[f32]>::new_slice(comptime!(lr * t as usize));
    let hcf = f32::cast_from(hc);
    let tid = UNIT_POS_Y * 32 + UNIT_POS_X;
    #[unroll]
    for tk in 0..t {
        let mut j = tid;
        while (j as usize) < lr {
            let v = dn[(tk * ds + j) as usize] / hcf;
            lo[(tk * comptime!(lr as u32) + j) as usize] = v / (1.0f32 + f32::exp(-v));
            j += 256u32;
        }
    }
    sync_cube();
    let i = CUBE_POS_X * 8 + UNIT_POS_Y;
    if i < d {
        let words = comptime!(lr as u32 / 4);
        let nb = comptime!(lr as u32 / 32);
        let mut acc = Array::<f32>::new(comptime!(t as usize));
        #[unroll]
        for tk in 0..t { acc[tk as usize] = 0.0f32; }
        #[unroll]
        for st in 0..hc {
            let row = st * d + i;
            let mut g = Array::<f32>::new(comptime!(t as usize));
            #[unroll]
            for tk in 0..t { g[tk as usize] = 0.0f32; }
            let mut wi = UNIT_POS_X;
            while wi < words {
                let word = w[(row * words + wi) as usize];
                let sc = s[(row * nb + wi / 8) as usize];
                #[unroll]
                for tk in 0..t {
                    let mut part = 0.0f32;
                    #[unroll]
                    for jj in 0..4u32 {
                        let v = dec(word, jj);
                        part += v * lo[(tk * comptime!(lr as u32) + wi * 4 + jj) as usize];
                    }
                    g[tk as usize] += part * sc;
                }
                wi += 32u32;
            }
            #[unroll]
            for tk in 0..t {
                let gt = plane_sum(g[tk as usize]);
                acc[tk as usize] += xn[(tk * xs + row) as usize] / (1.0f32 + f32::exp(-gt));
            }
        }
        if UNIT_POS_X == 0 {
            #[unroll]
            for tk in 0..t { out[(tk * os + ooff + i) as usize] = acc[tk as usize] / hcf; }
        }
    }
}

/// mixed[i] = mean_s xn[s,i] * sigmoid(up[s*d+i] . silu(dn[0..lr]/hc)); one plane per output i.
#[cube(launch_unchecked)]
fn k_gate(xn: &[f32], dn: &[f32], w: &[u32], s: &[f32], out: &mut [f32], d: u32, ooff: u32, #[comptime] lr: usize, #[comptime] hc: u32) {
    let mut lo = Shared::<[f32]>::new_slice(lr);
    let hcf = f32::cast_from(hc);
    let tid = UNIT_POS_Y * 32 + UNIT_POS_X;
    let mut j = tid;
    while (j as usize) < lr {
        let v = dn[j as usize] / hcf;
        lo[j as usize] = v / (1.0f32 + f32::exp(-v));
        j += 256u32;
    }
    sync_cube();
    let i = CUBE_POS_X * 8 + UNIT_POS_Y;
    if i < d {
        let words = comptime!(lr as u32 / 4);
        let nb = comptime!(lr as u32 / 32);
        let mut acc = 0.0f32;
        #[unroll]
        for st in 0..hc {
            let row = st * d + i;
            let mut g = 0.0f32;
            let mut wi = UNIT_POS_X;
            while wi < words {
                let word = w[(row * words + wi) as usize];
                let mut part = 0.0f32;
                #[unroll]
                for jj in 0..4u32 {
                    let v = dec(word, jj);
                    part += v * lo[(wi * 4 + jj) as usize];
                }
                g += part * s[(row * nb + wi / 8) as usize];
                wi += 32u32;
            }
            let gt = plane_sum(g);
            acc += xn[row as usize] / (1.0f32 + f32::exp(-gt));
        }
        if UNIT_POS_X == 0 {
            out[(ooff + i) as usize] = acc / hcf;
        }
    }
}

/// x[s,i] += (a[i] * (gate? sigmoid(gb[goff]) : 1) + (b? b[i] : 0)) * 2 sigmoid(dn[lr+s]/hc)
#[cube(launch_unchecked)]
fn k_comb(x: &mut [f32], a: &[f32], b: &[f32], c: &[f32], gb: &[f32], dn: &[f32], d: u32, goff: u32, lr: u32, #[comptime] hc: u32, #[comptime] gate: bool, #[comptime] addb: bool, #[comptime] addc: bool) {
    let k = ABSOLUTE_POS as u32;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let mut o = a[i as usize];
        if comptime!(gate) {
            o = o / (1.0f32 + f32::exp(-gb[goff as usize]));
        }
        if comptime!(addb) {
            o += b[i as usize];
        }
        if comptime!(addc) {
            o += c[i as usize];
        }
        let inj = dn[(lr + st) as usize] / f32::cast_from(hc);
        x[k as usize] += o * 2.0f32 / (1.0f32 + f32::exp(-inj));
    }
}

#[cube(launch_unchecked)]
fn k_copy(src: &[f32], dst: &mut [f32], so: u32, dof: u32, n: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        dst[(dof + j) as usize] = src[(so + j) as usize];
    }
}

#[cube(launch_unchecked)]
fn k_silu_mul(f: &[f32], h: &mut [f32], og: u32, ou: u32, n: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        let g = f[(og + j) as usize];
        h[j as usize] = g / (1.0f32 + f32::exp(-g)) * f[(ou + j) as usize];
    }
}

/// Gated DeltaNet decode, step 1: causal depthwise conv + SiLU per channel, shifting the conv state.
#[cube(launch_unchecked)]
fn k_conv(proj: &[f32], w: &[f32], cs: &mut [f32], co: &mut [f32], ch: u32, #[comptime] kk: u32) {
    let c = ABSOLUTE_POS as u32;
    if c < ch {
        let cur = proj[c as usize];
        let mut o = w[(c * kk + kk - 1) as usize] * cur;
        #[unroll]
        for k in 0..kk - 1 {
            o += w[(c * kk + k) as usize] * cs[(k * ch + c) as usize];
        }
        co[c as usize] = o / (1.0f32 + f32::exp(-o));
        #[unroll]
        for k in 0..kk - 2 {
            cs[(k * ch + c) as usize] = cs[((k + 1) * ch + c) as usize];
        }
        cs[((kk - 2) * ch + c) as usize] = cur;
    }
}

#[cube]
fn cube_sum(v: f32, red: &mut Shared<[f32]>, #[comptime] planes: u32) -> f32 {
    let p = plane_sum(v);
    if UNIT_POS_X % 32 == 0 {
        red[(UNIT_POS_X / 32) as usize] = p;
    }
    sync_cube();
    let mut t = 0.0f32;
    #[unroll]
    for i in 0..planes {
        t += red[i as usize];
    }
    sync_cube();
    t
}

/// Gated DeltaNet decode, step 2: one cube per value head, one unit per value column j (dk = dv = CUBE_DIM).
/// q/k L2 norms, decay g = exp(softplus(alpha + dt) * a), delta rule, RMSNorm * sigmoid(z).
#[cube(launch_unchecked)]
fn k_delta(co: &[f32], proj: &[f32], a: &[f32], dt: &[f32], norm: &[f32], s: &mut [f32], out: &mut [f32],
    nk: u32, nv: u32, zoff: u32, boff: u32, aoff: u32, eps: f32, #[comptime] dk: u32) {
    let h = CUBE_POS_X;
    let j = UNIT_POS_X;
    let kh = h % nk;
    let mut qs = Shared::<[f32]>::new_slice(dk as usize);
    let mut ks = Shared::<[f32]>::new_slice(dk as usize);
    let mut red = Shared::<[f32]>::new_slice(comptime!((dk / 32) as usize));
    let qj = co[(kh * dk + j) as usize];
    let kj = co[(nk * dk + kh * dk + j) as usize];
    let sq = cube_sum(qj * qj, &mut red, comptime!(dk / 32));
    let sk = cube_sum(kj * kj, &mut red, comptime!(dk / 32));
    qs[j as usize] = qj / f32::sqrt(sq + 1e-6f32) / f32::sqrt(f32::cast_from(dk));
    ks[j as usize] = kj / f32::sqrt(sk + 1e-6f32);
    sync_cube();
    let vj = co[(2 * nk * dk + h * dk + j) as usize];
    let x = proj[(aoff + h) as usize] + dt[h as usize];
    let mut sp = x;
    if x <= 20.0f32 {
        sp = f32::ln(1.0f32 + f32::exp(x));
    }
    let g = f32::exp(sp * a[h as usize]);
    let b = 1.0f32 / (1.0f32 + f32::exp(-proj[(boff + h) as usize]));
    let base = h * dk * dk;
    let mut mem = 0.0f32;
    for i in 0..dk {
        mem += s[(base + i * dk + j) as usize] * ks[i as usize];
    }
    mem *= g;
    let delta = (vj - mem) * b;
    let mut o = 0.0f32;
    for i in 0..dk {
        let idx = (base + i * dk + j) as usize;
        let v = s[idx] * g + ks[i as usize] * delta;
        s[idx] = v;
        o += v * qs[i as usize];
    }
    let so = cube_sum(o * o, &mut red, comptime!(dk / 32));
    let r = 1.0f32 / f32::sqrt(so / f32::cast_from(dk) + eps);
    let z = proj[(zoff + h * dk + j) as usize];
    out[(h * dk + j) as usize] = o * r * norm[j as usize] / (1.0f32 + f32::exp(-z));
    let _ = nv;
}

/// k_delta for `nt` consecutive tokens of a verify window in one launch: each thread keeps its state column in
/// registers across the tokens (one state read and one write instead of two reads and a write per token, and the
/// rollback checkpoints written from registers instead of a separate copy). Per token the arithmetic and its order
/// are k_delta's, so the results are the same.
#[cube(launch_unchecked)]
fn k_delta_m(co: &[f32], proj: &[f32], a: &[f32], dt: &[f32], norm: &[f32], s: &mut [f32], out: &mut [f32], c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], c3: &mut [f32],
    nk: u32, ch: u32, pn: u32, ostr: u32, ckm: u32, zoff: u32, boff: u32, aoff: u32, eps: f32, #[comptime] dk: u32, #[comptime] nt: u32) {
    let h = CUBE_POS_X;
    let j = UNIT_POS_X;
    let kh = h % nk;
    let mut qs = Shared::<[f32]>::new_slice(dk as usize);
    let mut ks = Shared::<[f32]>::new_slice(dk as usize);
    let mut red = Shared::<[f32]>::new_slice(comptime!((dk / 32) as usize));
    let base = h * dk * dk;
    let mut st = Array::<f32>::new(comptime!(dk as usize));
    #[unroll]
    for i in 0..dk {
        st[i as usize] = s[(base + i * dk + j) as usize];
    }
    #[unroll]
    for t in 0..nt {
        let cbs = t * ch;
        let pb = t * pn;
        let qj = co[(cbs + kh * dk + j) as usize];
        let kj = co[(cbs + nk * dk + kh * dk + j) as usize];
        let sq = cube_sum(qj * qj, &mut red, comptime!(dk / 32));
        let sk = cube_sum(kj * kj, &mut red, comptime!(dk / 32));
        qs[j as usize] = qj / f32::sqrt(sq + 1e-6f32) / f32::sqrt(f32::cast_from(dk));
        ks[j as usize] = kj / f32::sqrt(sk + 1e-6f32);
        sync_cube();
        let vj = co[(cbs + 2 * nk * dk + h * dk + j) as usize];
        let x = proj[(pb + aoff + h) as usize] + dt[h as usize];
        let mut sp = x;
        if x <= 20.0f32 {
            sp = f32::ln(1.0f32 + f32::exp(x));
        }
        let g = f32::exp(sp * a[h as usize]);
        let b = 1.0f32 / (1.0f32 + f32::exp(-proj[(pb + boff + h) as usize]));
        let mut mem = 0.0f32;
        #[unroll]
        for i in 0..dk {
            mem += st[i as usize] * ks[i as usize];
        }
        mem *= g;
        let delta = (vj - mem) * b;
        let mut o = 0.0f32;
        #[unroll]
        for i in 0..dk {
            let v = st[i as usize] * g + ks[i as usize] * delta;
            st[i as usize] = v;
            o += v * qs[i as usize];
        }
        if ((ckm >> t) & 1u32) == 1u32 {
            #[unroll]
            for i in 0..dk {
                let idx = (base + i * dk + j) as usize;
                if t == 0u32 { c0[idx] = st[i as usize]; } else if t == 1u32 { c1[idx] = st[i as usize]; } else if t == 2u32 { c2[idx] = st[i as usize]; } else { c3[idx] = st[i as usize]; }
            }
        }
        let so = cube_sum(o * o, &mut red, comptime!(dk / 32));
        let r = 1.0f32 / f32::sqrt(so / f32::cast_from(dk) + eps);
        let z = proj[(pb + zoff + h * dk + j) as usize];
        out[(t * ostr + h * dk + j) as usize] = o * r * norm[j as usize] / (1.0f32 + f32::exp(-z));
        sync_cube();
    }
    #[unroll]
    for i in 0..dk {
        s[(base + i * dk + j) as usize] = st[i as usize];
    }
}

// ---- routed experts cached on the GPU in their native GGUF formats (Strata's expert cache idea) ----
// type codes: 0 = IQ2_S, 1 = IQ2_XXS, 2 = IQ1_M, 3 = Q2_0

#[cube]
fn byte(w: &[u32], off: u32) -> u32 {
    (w[(off / 4) as usize] >> ((off % 4) * 8)) & 255u32
}

#[cube]
fn hf(h: u32) -> f32 {
    let e = (h >> 10) & 31u32;
    let m = f32::cast_from(h & 1023u32);
    let mut v = m * 0.000000059604645f32;
    if e != 0 {
        v = (1.0f32 + m / 1024.0f32) * f32::powf(2.0f32, f32::cast_from(e) - 15.0f32);
    }
    if (h >> 15) & 1u32 == 1u32 {
        v = -v;
    }
    v
}

/// Dot of the 32 weights of sub-block `sb` of a row (starting at word `rb`, repacked by `repack`)
/// with x[xo + sb*32 ..].
#[cube]
fn subdot(w: &[u32], rb: u32, sb: u32, x: &[f32], xo: u32, g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], #[comptime] ty: u32) -> f32 {
    let mut acc = 0.0f32;
    let xb = xo + sb * 32;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        #[unroll]
        for l in 0..4u32 {
            let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
            let s8 = (sg >> (8 * l)) & 255u32;
            let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
            if l >= 2 {
                scl = 0.5f32 + f32::cast_from(sc >> 4);
            }
            let ga = g2s[(idx * 2) as usize];
            let gb = g2s[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb;
                }
                let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + 8 * l + j) as usize];
                if (s8 >> j) & 1u32 == 1u32 {
                    v = -v;
                }
                part += v;
            }
            acc += part * scl;
        }
        acc *= d;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let q = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        #[unroll]
        for l in 0..4u32 {
            let idx = (q >> (8 * l)) & 255u32;
            let sg = ks[((a1 >> (7 * l)) & 127u32) as usize];
            let ga = g2x[(idx * 2) as usize];
            let gb = g2x[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb;
                }
                let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + 8 * l + j) as usize];
                if (sg >> j) & 1u32 == 1u32 {
                    v = -v;
                }
                part += v;
            }
            acc += part;
        }
        acc *= db;
    }
    if comptime!(ty == 2) {
        let blk = rb + (sb / 8) * 15;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let hw = (w[(blk + 9 + ib / 2) as usize] >> ((ib % 2) * 16)) & 65535u32;
        let scw = (w[(blk + 13 + ib / 4) as usize] >> (((ib / 2) % 2) * 16)) & 65535u32;
        let sh = 6 * (ib % 2);
        let dl1 = d * f32::cast_from(2 * ((scw >> sh) & 7u32) + 1);
        let dl2 = d * f32::cast_from(2 * ((scw >> (sh + 3)) & 7u32) + 1);
        #[unroll]
        for l in 0..4u32 {
            let qb = (qs >> (8 * l)) & 255u32;
            let hh = (hw >> (8 * (l / 2))) & 255u32;
            let mut idx = qb | ((hh << 8) & 1792u32);
            let mut dbit = hh & 8u32;
            if l % 2 == 1 {
                idx = qb | ((hh << 4) & 1792u32);
                dbit = hh & 128u32;
            }
            let mut delta = 0.125f32;
            if dbit != 0 {
                delta = -0.125f32;
            }
            let mut sl = dl1;
            if l >= 2 {
                sl = dl2;
            }
            let ga = g1[(idx * 2) as usize];
            let gb2 = g1[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb2;
                }
                let gb = (gw >> ((j % 4) * 8)) & 255u32;
                let mut gv = f32::cast_from(gb);
                if gb >= 128 {
                    gv = gv - 256.0f32;
                }
                part += (gv + delta) * x[(xb + 8 * l + j) as usize];
            }
            acc += part * sl;
        }
    }
    if comptime!(ty == 3) {
        let blk = rb + (sb / 2) * 5;
        let d = f32::reinterpret(w[blk as usize]);
        #[unroll]
        for k2 in 0..2u32 {
            let qw = w[(blk + 1 + 2 * (sb % 2) + k2) as usize];
            #[unroll]
            for k in 0..4u32 {
                let b = (qw >> (8 * k)) & 255u32;
                #[unroll]
                for j in 0..4u32 {
                    acc += (f32::cast_from((b >> (2 * j)) & 3u32) - 1.0f32) * x[(xb + 16 * k2 + 4 * k + j) as usize];
                }
            }
        }
        acc *= d;
    }
    acc
}


/// Like `subdot` but for 8 values only (`q` = index of the 8-group: sub-block q/4, part q%4), so
/// neighbouring lanes read neighbouring activations (the 32-per-lane layout made every x load a
/// 32-way bank conflict / uncoalesced access).
#[cube]
fn subdot8(w: &[u32], rb: u32, q: u32, x: &[f32], xo: u32, g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], #[comptime] ty: u32) -> f32 {
    let mut acc = 0.0f32;
    let sb = q / 4;
    let l = q % 4;
    let xb = xo + q * 8;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
        let s8 = (sg >> (8 * l)) & 255u32;
        let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
        if l >= 2 {
            scl = 0.5f32 + f32::cast_from(sc >> 4);
        }
        let ga = g2s[(idx * 2) as usize];
        let gb = g2s[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb;
            }
            let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + j) as usize];
            if (s8 >> j) & 1u32 == 1u32 {
                v = -v;
            }
            acc += v;
        }
        acc *= d * scl;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        let idx = (qw >> (8 * l)) & 255u32;
        let sg = ks[((a1 >> (7 * l)) & 127u32) as usize];
        let ga = g2x[(idx * 2) as usize];
        let gb = g2x[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb;
            }
            let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + j) as usize];
            if (sg >> j) & 1u32 == 1u32 {
                v = -v;
            }
            acc += v;
        }
        acc *= db;
    }
    if comptime!(ty == 2) {
        let blk = rb + (sb / 8) * 15;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let hw = (w[(blk + 9 + ib / 2) as usize] >> ((ib % 2) * 16)) & 65535u32;
        let scw = (w[(blk + 13 + ib / 4) as usize] >> (((ib / 2) % 2) * 16)) & 65535u32;
        let mut sh = 6 * (ib % 2);
        if l >= 2 {
            sh += 3;
        }
        let sl = d * f32::cast_from(2 * ((scw >> sh) & 7u32) + 1);
        let qb = (qs >> (8 * l)) & 255u32;
        let hh = (hw >> (8 * (l / 2))) & 255u32;
        let mut idx = qb | ((hh << 8) & 1792u32);
        let mut dbit = hh & 8u32;
        if l % 2 == 1 {
            idx = qb | ((hh << 4) & 1792u32);
            dbit = hh & 128u32;
        }
        let mut delta = 0.125f32;
        if dbit != 0 {
            delta = -0.125f32;
        }
        let ga = g1[(idx * 2) as usize];
        let gb2 = g1[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb2;
            }
            let gb = (gw >> ((j % 4) * 8)) & 255u32;
            let mut gv = f32::cast_from(gb);
            if gb >= 128 {
                gv = gv - 256.0f32;
            }
            acc += (gv + delta) * x[(xb + j) as usize];
        }
        acc *= sl;
    }
    if comptime!(ty == 3) {
        let blk = rb + (sb / 2) * 5;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * (sb % 2) + l / 2) as usize];
        let half = (qw >> (16 * (l % 2))) & 65535u32;
        #[unroll]
        for k in 0..2u32 {
            let b = (half >> (8 * k)) & 255u32;
            #[unroll]
            for j in 0..4u32 {
                acc += (f32::cast_from((b >> (2 * j)) & 3u32) - 1.0f32) * x[(xb + 4 * k + j) as usize];
            }
        }
        acc *= d;
    }
    acc
}
/// Repack GGUF blocks into word-aligned blocks with an f32 scale first (layouts in `subdot`).
pub fn hpin() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_HPIN").map(|v| v == "1").unwrap_or(false)) }
/// A repacked expert in host memory: (gate|up|down words as shared bytes, up offset, down offset) in u32 words.
pub type Pk = std::sync::Arc<(cubecl::bytes::Bytes, usize, usize)>;
pub fn gemv1r() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_GEMV1R").map(|v| v == "1").unwrap_or(false)) }
pub fn xs2() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XS2").map(|v| v == "1").unwrap_or(false)) }
pub fn xpin() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XPIN").map(|v| v == "1").unwrap_or(false)) }
pub fn repack(ty: u32, raw: &[u8]) -> Vec<u32> {
    let f = |b: &[u8], o: usize| half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
    let wd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut out = vec![];
    match ty {
        0 => for b in raw.chunks_exact(82) { out.push((f(b, 0) * 0.25).to_bits()); for o in (2..82).step_by(4) { out.push(wd(b, o)); } },
        1 => for b in raw.chunks_exact(66) { out.push((f(b, 0) * 0.25).to_bits()); for o in (2..66).step_by(4) { out.push(wd(b, o)); } },
        2 => for b in raw.chunks_exact(56) { let s = |k: usize| u16::from_le_bytes([b[48 + 2 * k], b[49 + 2 * k]]);
            let d = half::f16::from_bits((s(0) >> 12) | ((s(1) >> 8) & 0x00f0) | ((s(2) >> 4) & 0x0f00) | (s(3) & 0xf000)).to_f32();
            out.push(d.to_bits()); for o in (0..56).step_by(4) { out.push(wd(b, o)); } },
        _ => for b in raw.chunks_exact(18) { out.push(f(b, 0).to_bits()); for o in (2..18).step_by(4) { out.push(wd(b, o)); } },
    }
    out
}
pub fn block_words(ty: u32) -> (usize, usize) { match ty { 0 => (256, 21), 1 => (256, 17), 2 => (256, 15), _ => (64, 5) } }

/// h[row] = silu(gate_row . x) * (up_row . x) for one cached expert; one plane per row.
#[cube(launch_unchecked)]
fn k_xgu(x: &[f32], wg: &[u32], wu: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], h: &mut [f32], hoff: u32, rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            gs += subdot8(wg, row * rbg, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            us += subdot8(wu, row * rbu, sb, x, 0u32, g2s, g2x, ks, g1, tu);
            sb += 32u32;
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if UNIT_POS_X == 0 {
            h[(hoff + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// Round h to the CPU's Q8 activation grid (per 32: d = amax/127), so GPU and CPU experts agree.
#[cube(launch_unchecked)]
fn k_q8sim(h: &mut [f32], n: u32) {
    let b = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let i = b * 32 + UNIT_POS_X;
    if b * 32 < n {
        let v = h[i as usize];
        let amax = plane_max(f32::abs(v));
        let d = amax / 127.0f32;
        let mut q = 0.0f32;
        if d > 0.0f32 {
            q = f32::round(v * (1.0f32 / d));
        }
        h[i as usize] = q * d;
    }
}

/// dst[i] = src[soff + i] rounded to the Q8 grid (per 32); one plane per block. Also zeroes `z`.
#[cube(launch_unchecked)]
fn k_q8copy(src: &[f32], dst: &mut [f32], z: &mut [f32], soff: u32, n: u32) {
    let b = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let i = b * 32 + UNIT_POS_X;
    if b * 32 < n {
        let v = src[(soff + i) as usize];
        let amax = plane_max(f32::abs(v));
        let d = amax / 127.0f32;
        let mut q = 0.0f32;
        if d > 0.0f32 {
            q = f32::round(v * (1.0f32 / d));
        }
        dst[i as usize] = q * d;
        z[i as usize] = 0.0f32;
    }
}

/// y[row] += wgt * (down_row . h) for one cached expert.
#[cube(launch_unchecked)]
fn k_xdown(h: &[f32], wd: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], y: &mut [f32], hoff: u32, rows: u32, cols: u32, rbd: u32, wgt: f32, #[comptime] td: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut s = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            s += subdot8(wd, row * rbd, sb, h, hoff, g2s, g2x, ks, g1, td);
            sb += 32u32;
        }
        let t = plane_sum(s);
        if UNIT_POS_X == 0 {
            y[row as usize] += wgt * t;
        }
    }
}


/// Batched cached experts (all of one layer in one launch): expert = CUBE_POS_Y, its word offsets
/// in the arena are meta[4e], meta[4e+1] (gate, up); h is [expert][rows].
#[cube(launch_unchecked)]
fn k_xgu_b(x: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], h: &mut [f32], rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let go = meta[(e * 4) as usize] + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let nsb = cols / 8;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            gs += subdot8(ar, go, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            us += subdot8(ar, uo, sb, x, 0u32, g2s, g2x, ks, g1, tu);
            sb += 32u32;
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if UNIT_POS_X == 0 {
            h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// k_xgu_b with four independent sub-block chains in flight per lane (the one-chain loop was latency-bound).
#[cube(launch_unchecked)]
fn k_xgu_b4(x: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], h: &mut [f32], rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let go = meta[(e * 4) as usize] + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let nsb = cols / 8;
        let mut g0 = 0.0f32; let mut g1s = 0.0f32; let mut u0 = 0.0f32; let mut u1 = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb + 32 < nsb {
            g0 += subdot8(ar, go, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            u0 += subdot8(ar, uo, sb, x, 0u32, g2s, g2x, ks, g1, tu);
            g1s += subdot8(ar, go, sb + 32, x, 0u32, g2s, g2x, ks, g1, tg);
            u1 += subdot8(ar, uo, sb + 32, x, 0u32, g2s, g2x, ks, g1, tu);
            sb += 64u32;
        }
        if sb < nsb {
            g0 += subdot8(ar, go, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            u0 += subdot8(ar, uo, sb, x, 0u32, g2s, g2x, ks, g1, tu);
        }
        let g = plane_sum(g0 + g1s);
        let u = plane_sum(u0 + u1);
        if UNIT_POS_X == 0 {
            h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// k_xdown_b with the (expert, sub-block) loop flattened across the lanes: all experts' loads of a row are
/// independent, instead of one expert after another.
#[cube(launch_unchecked)]
fn k_xdown_b4(h: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], y: &mut [f32], ne: u32, rows: u32, cols: u32, rbd: u32, #[comptime] td: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let tot = ne * nsb;
        let mut s0 = 0.0f32; let mut s1 = 0.0f32;
        let mut j = UNIT_POS_X;
        while j + 32 < tot {
            let e0 = j / nsb; let e1 = (j + 32) / nsb;
            s0 += f32::reinterpret(meta[(e0 * 4 + 3) as usize]) * subdot8(ar, meta[(e0 * 4 + 2) as usize] + row * rbd, j - e0 * nsb, h, e0 * cols, g2s, g2x, ks, g1, td);
            s1 += f32::reinterpret(meta[(e1 * 4 + 3) as usize]) * subdot8(ar, meta[(e1 * 4 + 2) as usize] + row * rbd, j + 32 - e1 * nsb, h, e1 * cols, g2s, g2x, ks, g1, td);
            j += 64u32;
        }
        if j < tot {
            let e0 = j / nsb;
            s0 += f32::reinterpret(meta[(e0 * 4 + 3) as usize]) * subdot8(ar, meta[(e0 * 4 + 2) as usize] + row * rbd, j - e0 * nsb, h, e0 * cols, g2s, g2x, ks, g1, td);
        }
        let t = plane_sum(s0 + s1);
        if UNIT_POS_X == 0 {
            y[row as usize] += t;
        }
    }
}

/// Q8 pack: per 32 values d = amax/127, q = round(v/d) as int8 (4 per word) - the same grid as
/// k_q8copy / the CPU Q8 activations. One thread per 32-block.
#[cube(launch_unchecked)]
fn k_q8pack(src: &[f32], q: &mut [u32], sc: &mut [f32], soff: u32, nb: u32) {
    let b = ABSOLUTE_POS_X;
    if b < nb {
        let base = soff + b * 32;
        let mut amax = 0.0f32;
        for i in 0..32u32 { amax = f32::max(amax, f32::abs(src[(base + i) as usize])); }
        let d = amax / 127.0f32;
        let mut id = 0.0f32;
        if d > 0.0f32 { id = 1.0f32 / d; }
        sc[b as usize] = d;
        for w in 0..8u32 {
            let mut word = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let v = i32::cast_from(f32::round(src[(base + w * 4 + j) as usize] * id));
                word |= (u32::cast_from(v) & 255u32) << (8 * j);
            }
            q[(b * 8 + w) as usize] = word;
        }
    }
}

/// Signed grid word: bytes of g negated where the mask byte is 0xFF (grid bytes are 1..127, no carries).
#[cube]
fn sgn(g: u32, m: u32) -> i32 {
    i32::cast_from((g ^ m) + (m & 16843009u32))
}

/// IQ2_S (ty 0) / IQ2_XXS (ty 1) 8-group dot against int8 activations (xq words, xs per-32 scales).
#[cube]
fn gdot8(w: &[u32], rb: u32, q: u32, xq: &[u32], xs: &[f32], g2s: &[u32], g2x: &[u32], ks: &[u32], sm: &[u32], #[comptime] ty: u32) -> f32 {
    let sb = q / 4;
    let l = q % 4;
    let x0 = i32::cast_from(xq[(q * 2) as usize]);
    let x1 = i32::cast_from(xq[(q * 2 + 1) as usize]);
    let mut r = 0.0f32;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
        let s8 = (sg >> (8 * l)) & 255u32;
        let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
        if l >= 2 { scl = 0.5f32 + f32::cast_from(sc >> 4); }
        let a = sgn(g2s[(idx * 2) as usize], sm[(s8 * 2) as usize]);
        let b = sgn(g2s[(idx * 2 + 1) as usize], sm[(s8 * 2 + 1) as usize]);
        let dot = dp4a(b, x1, dp4a(a, x0, 0i32));
        r = f32::cast_from(dot) * d * scl * xs[sb as usize];
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        let idx = (qw >> (8 * l)) & 255u32;
        let s8 = ks[((a1 >> (7 * l)) & 127u32) as usize] & 255u32;
        let a = sgn(g2x[(idx * 2) as usize], sm[(s8 * 2) as usize]);
        let b = sgn(g2x[(idx * 2 + 1) as usize], sm[(s8 * 2 + 1) as usize]);
        let dot = dp4a(b, x1, dp4a(a, x0, 0i32));
        r = f32::cast_from(dot) * db * xs[sb as usize];
    }
    r
}

/// Batched cached experts, gate/up with int8 activations and dp4a (IQ2_S / IQ2_XXS).
#[cube(launch_unchecked)]
fn k_xgu_q(xq: &[u32], xs: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], sm: &[u32], h: &mut [f32], rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let go = meta[(e * 4) as usize] + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let nsb = cols / 8;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            gs += gdot8(ar, go, sb, xq, xs, g2s, g2x, ks, sm, tg);
            us += gdot8(ar, uo, sb, xq, xs, g2s, g2x, ks, sm, tu);
            sb += 32u32;
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if UNIT_POS_X == 0 {
            h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// gdot8 with the activation words / scale already in registers.
#[cube]
fn gdot8x(w: &[u32], rb: u32, q: u32, x0: i32, x1: i32, xsc: f32, g2s: &[u32], g2x: &[u32], ks: &[u32], sm: &[u32], #[comptime] ty: u32) -> f32 {
    let sb = q / 4;
    let l = q % 4;
    let mut r = 0.0f32;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
        let s8 = (sg >> (8 * l)) & 255u32;
        let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
        if l >= 2 { scl = 0.5f32 + f32::cast_from(sc >> 4); }
        let a = sgn(g2s[(idx * 2) as usize], sm[(s8 * 2) as usize]);
        let b = sgn(g2s[(idx * 2 + 1) as usize], sm[(s8 * 2 + 1) as usize]);
        r = f32::cast_from(dp4a(b, x1, dp4a(a, x0, 0i32))) * d * scl * xsc;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        let idx = (qw >> (8 * l)) & 255u32;
        let s8 = ks[((a1 >> (7 * l)) & 127u32) as usize] & 255u32;
        let a = sgn(g2x[(idx * 2) as usize], sm[(s8 * 2) as usize]);
        let b = sgn(g2x[(idx * 2 + 1) as usize], sm[(s8 * 2 + 1) as usize]);
        r = f32::cast_from(dp4a(b, x1, dp4a(a, x0, 0i32))) * db * xsc;
    }
    r
}

/// k_xgu_q with `rw` rows per warp and the lane's activation groups held in registers (ng = cols / 256).
#[cube(launch_unchecked)]
fn k_xgu_q4(xq: &[u32], xs: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], sm: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32,
    #[comptime] ng: u32, #[comptime] rw: u32, #[comptime] tg: u32, #[comptime] tu: u32) {
    let e = CUBE_POS_Y;
    let lane = UNIT_POS_X;
    let r0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * rw;
    let mut xa = Array::<i32>::new(comptime!((2 * ng) as usize));
    let mut xsa = Array::<f32>::new(comptime!(ng as usize));
    #[unroll]
    for k in 0..ng {
        let q = lane + 32 * k;
        xa[(2 * k) as usize] = i32::cast_from(xq[(q * 2) as usize]);
        xa[(2 * k + 1) as usize] = i32::cast_from(xq[(q * 2 + 1) as usize]);
        xsa[k as usize] = xs[(q / 4) as usize];
    }
    let gb = meta[(e * 4) as usize];
    let ub = meta[(e * 4 + 1) as usize];
    #[unroll]
    for r in 0..rw {
        let row = r0 + r;
        if row < rows {
            let mut gs = 0.0f32;
            let mut us = 0.0f32;
            #[unroll]
            for k in 0..ng {
                let q = lane + 32 * k;
                gs += gdot8x(ar, gb + row * rbg, q, xa[(2 * k) as usize], xa[(2 * k + 1) as usize], xsa[k as usize], g2s, g2x, ks, sm, tg);
                us += gdot8x(ar, ub + row * rbu, q, xa[(2 * k) as usize], xa[(2 * k + 1) as usize], xsa[k as usize], g2s, g2x, ks, sm, tu);
            }
            let g = plane_sum(gs);
            let u = plane_sum(us);
            if lane == 0 {
                h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
            }
        }
    }
}

/// Sign mask (0xFF bytes) for the low 4 bits of s, without a table.
#[cube]
fn smask4(s: u32) -> u32 {
    (((s & 15u32) * 2113665u32) & 16843009u32) * 255u32
}

/// gdot8x with the grid in shared memory and the sign masks / IQ2_XXS sign parity computed.
#[cube]
fn gdot8s(w: &[u32], rb: u32, q: u32, x0: i32, x1: i32, xsc: f32, grid: &Shared<[u32]>, #[comptime] ty: u32) -> f32 {
    let sb = q / 4;
    let l = q % 4;
    let mut r = 0.0f32;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
        let s8 = (sg >> (8 * l)) & 255u32;
        let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
        if l >= 2 { scl = 0.5f32 + f32::cast_from(sc >> 4); }
        let m0 = smask4(s8); let m1 = smask4(s8 >> 4);
        let a = i32::cast_from((grid[(idx * 2) as usize] ^ m0) + (m0 & 16843009u32));
        let b = i32::cast_from((grid[(idx * 2 + 1) as usize] ^ m1) + (m1 & 16843009u32));
        r = f32::cast_from(dp4a(b, x1, dp4a(a, x0, 0i32))) * d * scl * xsc;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        let idx = (qw >> (8 * l)) & 255u32;
        let k7 = (a1 >> (7 * l)) & 127u32;
        let s8 = k7 | ((u32::count_ones(k7) & 1u32) << 7);
        let m0 = smask4(s8); let m1 = smask4(s8 >> 4);
        let a = i32::cast_from((grid[(idx * 2) as usize] ^ m0) + (m0 & 16843009u32));
        let b = i32::cast_from((grid[(idx * 2 + 1) as usize] ^ m1) + (m1 & 16843009u32));
        r = f32::cast_from(dp4a(b, x1, dp4a(a, x0, 0i32))) * db * xsc;
    }
    r
}

/// Gate/up (both of type ty) with the grid table staged in shared memory; one row per warp.
#[cube(launch_unchecked)]
fn k_xgu_s(xq: &[u32], xs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32,
    #[comptime] ng: u32, #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let lane = UNIT_POS_X;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let m0 = meta[(e * 4) as usize];
        let sel = m0 >> 31;
        let go = (m0 & 2147483647u32) + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        #[unroll]
        for k in 0..ng {
            let q = lane + 32 * k;
            let x0 = i32::cast_from(xq[(q * 2) as usize]);
            let x1 = i32::cast_from(xq[(q * 2 + 1) as usize]);
            let xsc = xs[(q / 4) as usize];
            if sel == 0u32 {
                gs += gdot8s(ar, go, q, x0, x1, xsc, &grid, ty);
                us += gdot8s(ar, uo, q, x0, x1, xsc, &grid, ty);
            } else {
                gs += gdot8s(ar1, go, q, x0, x1, xsc, &grid, ty);
                us += gdot8s(ar1, uo, q, x0, x1, xsc, &grid, ty);
            }
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if lane == 0 {
            h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// Batched cached experts, Q2_0 down with int8 h and dp4a; (expert, group) flattened over the lanes.
/// q2l[b] = the four 2-bit fields of byte b as bytes.
#[cube(launch_unchecked)]
fn k_xdown_q(hq: &[u32], hs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], q2l: &[u32], y: &mut [f32], ne: u32, rows: u32, cols: u32, rbd: u32, acc: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let tot = ne * nsb;
        let mut s = 0.0f32;
        let mut j = UNIT_POS_X;
        while j < tot {
            let e = j / nsb;
            let q = j - e * nsb;
            let sb = q / 4;
            let l = q % 4;
            let m2 = meta[(e * 4 + 2) as usize];
            let blk = (m2 & 2147483647u32) + row * rbd + (sb / 2) * 5;
            let wi = blk + 1 + 2 * (sb % 2) + l / 2;
            let mut d = 0.0f32;
            let mut qw = 0u32;
            if (m2 >> 31) == 0u32 { d = f32::reinterpret(ar[blk as usize]); qw = ar[wi as usize]; } else { d = f32::reinterpret(ar1[blk as usize]); qw = ar1[wi as usize]; }
            let half = (qw >> (16 * (l % 2))) & 65535u32;
            let xw = e * cols / 4 + q * 2;
            let x0 = i32::cast_from(hq[xw as usize]);
            let x1 = i32::cast_from(hq[(xw + 1) as usize]);
            let v = dp4a(i32::cast_from(q2l[(half >> 8) as usize]), x1, dp4a(i32::cast_from(q2l[(half & 255u32) as usize]), x0, 0i32));
            let ones = dp4a(16843009i32, x1, dp4a(16843009i32, x0, 0i32));
            s += f32::reinterpret(meta[(e * 4 + 3) as usize]) * d * hs[((e * cols) / 32 + q / 4) as usize] * f32::cast_from(v - ones);
            j += 32u32;
        }
        let t = plane_sum(s);
        if UNIT_POS_X == 0 {
            if acc == 1u32 { y[row as usize] += t; } else { y[row as usize] = t; }
        }
    }
}

/// y[row] += sum_e wgt_e * (down_e row . h_e); down offset meta[4e+2], weight bits meta[4e+3].
#[cube(launch_unchecked)]
fn k_xdown_b(h: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], y: &mut [f32], ne: u32, rows: u32, cols: u32, rbd: u32, #[comptime] td: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut s = 0.0f32;
        for e in 0..ne {
            let dof = meta[(e * 4 + 2) as usize] + row * rbd;
            let w = f32::reinterpret(meta[(e * 4 + 3) as usize]);
            let mut p = 0.0f32;
            let mut sb = UNIT_POS_X;
            while sb < nsb {
                p += subdot8(ar, dof, sb, h, e * cols, g2s, g2x, ks, g1, td);
                sb += 32u32;
            }
            s += w * p;
        }
        let t = plane_sum(s);
        if UNIT_POS_X == 0 {
            y[row as usize] += t;
        }
    }
}
/// Full-attention prep for one token: q heads (cube < nh) get RMS norm + RoPE into q; k heads get
/// norm + RoPE and, with v, are written to the KV cache at `pos`. cs = [cos(half), sin(half)].
#[cube(launch_unchecked)]
fn k_attn_prep(p: &[f32], qn: &[f32], kn: &[f32], cs: &[f32], q: &mut [f32], kc: &mut [f32], vc: &mut [f32], ko: u32, vo: u32, nh: u32, nkv: u32, hd: u32, half: u32, pos: u32, eps: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let mut base = h * 2 * hd;
    let mut dst = h * hd;
    if h >= nh {
        base = ko + (h - nh) * hd;
        dst = (pos * nkv + h - nh) * hd;
    }
    let mut ss = 0.0f32;
    let mut i = lane;
    while i < hd {
        let v = p[(base + i) as usize];
        ss += v * v;
        i += 32u32;
    }
    let s = 1.0f32 / f32::sqrt(plane_sum(ss) / f32::cast_from(hd) + eps);
    i = lane;
    while i < hd {
        let mut wi = 0.0f32;
        let mut wj = 0.0f32;
        let mut j = i;
        if i < half {
            j = i + half;
        } else if i < 2 * half {
            j = i - half;
        }
        if h < nh {
            wi = qn[i as usize];
            wj = qn[j as usize];
        } else {
            wi = kn[i as usize];
            wj = kn[j as usize];
        }
        let ni = p[(base + i) as usize] * (s * wi);
        let nj = p[(base + j) as usize] * (s * wj);
        let mut o = ni;
        if i < half {
            o = ni * cs[i as usize] - nj * cs[(half + i) as usize];
        } else if i < 2 * half {
            o = ni * cs[j as usize] + nj * cs[(half + j) as usize];
        }
        if h < nh {
            q[(dst + i) as usize] = o;
        } else {
            kc[(dst + i) as usize] = o;
            vc[(dst + i) as usize] = p[(vo + (h - nh) * hd + i) as usize];
        }
        i += 32u32;
    }
}

/// One query head per cube (one plane): softmax(q.K^T * scale) V over n positions, times sigmoid(gate).
#[cube(launch_unchecked)]
fn k_attn(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n: u32, cap: u32, scale: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let kvh = h / (nh / nkv);
    let mut mx = scale * -3.0e30f32;
    let mut pp = lane;
    while pp < n {
        let kb = (pp * nkv + kvh) * hd;
        let mut dd = 0.0f32;
        for i in 0..hd {
            dd += q[(h * hd + i) as usize] * kc[(kb + i) as usize];
        }
        let v = dd * scale;
        sc[(h * cap + pp) as usize] = v;
        mx = f32::max(mx, v);
        pp += 32u32;
    }
    let m = plane_max(mx);
    let mut sum = 0.0f32;
    pp = lane;
    while pp < n {
        let e = f32::exp(sc[(h * cap + pp) as usize] - m);
        sc[(h * cap + pp) as usize] = e;
        sum += e;
        pp += 32u32;
    }
    let inv = 1.0f32 / plane_sum(sum);
    sync_cube();
    let mut i = lane;
    while i < hd {
        let mut acc = 0.0f32;
        for t in 0..n {
            acc += sc[(h * cap + t) as usize] * inv * vc[((t * nkv + kvh) * hd + i) as usize];
        }
        let g = p[(h * 2 * hd + hd + i) as usize];
        o[(h * hd + i) as usize] = acc * (1.0f32 / (1.0f32 + f32::exp(-g)));
        i += 32u32;
    }
}
/// Grid tables for the expert kernels.
/// New expert kernels (PYRE_QN_XK=1 selects the old ones).
fn xk4() -> bool { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *V.get_or_init(|| std::env::var("PYRE_QN_XK").map(|v| v != "1").unwrap_or(true)) }

/// dp4a expert kernels (PYRE_QN_XQ=0 turns them off).
pub fn xkq() -> bool { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *V.get_or_init(|| std::env::var("PYRE_QN_XQ").map(|v| v != "0").unwrap_or(true)) }

/// rows per warp for the register-x gate/up kernel (PYRE_QN_XRW, 0 = old one-row kernel).
pub fn xrw() -> u32 { std::env::var("PYRE_QN_XRW").ok().and_then(|v| v.parse().ok()).unwrap_or(1) }

/// shared-memory grid gate/up kernel (PYRE_QN_XSH=0 off) and its warps per cube (PYRE_QN_XSHR).
pub fn xsh() -> bool { std::env::var("PYRE_QN_XSH").map(|v| v != "0").unwrap_or(true) }
pub fn xshr() -> u32 { std::env::var("PYRE_QN_XSHR").ok().and_then(|v| v.parse().ok()).unwrap_or(8) }

/// Reusable buffers of the one-arena expert path (one layer at a time on the stream).
pub struct XScr { xq: Handle, xs: Handle, hq: Handle, hs: Handle, h: Handle, pub y: Handle, max_ne: usize }

pub struct XTabs { g2s: Handle, g2x: Handle, ks: Handle, g1: Handle, sm: Handle, q2l: Handle }
/// One cached expert: raw GGUF bytes of its gate, up and down rows.
pub struct XEnt { pub off: [u32; 3], pub arena: usize, pub slot: usize, pub bytes: usize }
/// Words of one repacked expert (gate, up, down).
pub fn xwords(tys: (u32, u32, u32), d: usize, ff: usize) -> (usize, usize, usize) { let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w }; (ff * rw(tys.0, d), ff * rw(tys.1, d), d * rw(tys.2, ff)) }
/// Words of one repacked expert (gate, up, down). fn xwords(tys: (u32, u32, u32), d: usize, ff: usize) -> (usize, usize, usize) { let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w }; (ff * rw(tys.0, d), ff * rw(tys.1, d), d * rw(tys.2, ff)) }
pub fn xtype(t: super::gguf_q::QType) -> Option<u32> {
    use super::gguf_q::QType::*;
    match t { IQ2S => Some(0), IQ2XXS => Some(1), IQ1M => Some(2), Q2_0 => Some(3), _ => None }
}

pub struct Cube { pub client: Client, pub budget: std::sync::atomic::AtomicI64 }

pub struct CMat { w: Handle, s: Handle, pub rows: usize, pub cols: usize, pub splits: Vec<usize>,
    /// native IQ4_XS row ranges (out row, rows, words) and Q8 row ranges (out row, rows, row in w/s); empty = all Q8
    nat: Vec<(usize, usize, Handle)>, qseg: Vec<(usize, usize, usize)>, kv: Option<Handle> }

impl Cube {
    pub fn new() -> anyhow::Result<Self> {
        let client = cubecl::Device::cuda(0).map_err(|e| anyhow::anyhow!("cuda: {e:?}"))?.client();
        let v = crate::limits::vram().ok_or_else(|| anyhow::anyhow!("no NVML VRAM reading"))?;
        let budget = v.free as i64 - crate::limits::reserve() as i64 - (1536i64 << 20);
        Ok(Self { client, budget: budget.into() })
    }
    pub fn take(&self, bytes: usize) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let n = bytes as i64;
        if self.budget.fetch_sub(n, Relaxed) < n { self.budget.fetch_add(n, Relaxed); false } else { true }
    }
    /// Upload the rows of `ms` (then the f32 `extra` rows) as one Q8 matrix.
    pub fn upload(&self, ms: &[&QMat], extra: &[&[f32]]) -> Option<CMat> {
        let cols = ms[0].cols;
        let rows: usize = ms.iter().map(|m| m.rows).sum::<usize>() + extra.len();
        if !self.take(rows * cols + rows * cols / 8) { return None; }
        let q = super::qn_gpu::to_q8_0(ms);
        let nb = cols / 32;
        let mut words: Vec<u32> = Vec::with_capacity(rows * cols / 4);
        let mut sc: Vec<f32> = Vec::with_capacity(rows * nb);
        for blk in q.chunks_exact(34) {
            sc.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32());
            for c in blk[2..].chunks_exact(4) { words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); }
        }
        for e in extra { for b in 0..nb { let x = &e[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs())); let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            sc.push(d); for c in x.chunks_exact(4) { let qb: Vec<u8> = c.iter().map(|v| ((v * id).round().clamp(-127.0, 127.0) as i8) as u8).collect(); words.push(u32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]])); } } }
        let mut splits = vec![0]; for m in ms { splits.push(splits.last().unwrap() + m.rows); } for _ in extra { splits.push(splits.last().unwrap() + 1); }
        Some(CMat { w: self.hup(u32::as_bytes(&words)), s: self.hup(f32::as_bytes(&sc)), rows, cols, splits, nat: vec![], qseg: vec![], kv: None })
    }
    /// Like `upload`, but IQ4_XS matrices stay in their own format (half the bytes of Q8_0);
    /// the rest are packed as Q8_0 as before. Opt-in (PYRE_QN_NATIVE=1): on the RTX 3060 this kernel is
    /// still ALU/LSU-bound (mout 2560x6144: 65 us vs 59 us for Q8_0), so it is not a win yet.
    /// A matrix already in Q8_0 blocks (34 bytes per 32 values, rows in order), e.g. the drafter's draft-vocabulary head.
    pub fn upload_q8(&self, rows: usize, cols: usize, q: &[u8]) -> Option<CMat> {
        if !self.take(rows * cols + rows * cols / 8) { return None; }
        let mut words: Vec<u32> = Vec::with_capacity(rows * cols / 4); let mut sc: Vec<f32> = Vec::with_capacity(rows * cols / 32);
        for blk in q.chunks_exact(34) { sc.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32()); for c in blk[2..].chunks_exact(4) { words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } }
        Some(CMat { w: self.hup(u32::as_bytes(&words)), s: self.hup(f32::as_bytes(&sc)), rows, cols, splits: vec![0, rows], nat: vec![], qseg: vec![], kv: None })
    }
    pub fn upload_mix(&self, ms: &[&QMat], extra: &[&[f32]]) -> Option<CMat> {
        use super::gguf_q::QType;
        let cols = ms[0].cols;
        let native = |m: &QMat| m.ty == QType::IQ4XS && cols % 256 == 0 && cols <= 6144;
        if native_mode() == 0 || !ms.iter().any(|m| native(m)) { return self.upload(ms, extra); }
        let q8: Vec<&QMat> = ms.iter().copied().filter(|m| !native(m)).collect();
        let nq8 = q8.iter().map(|m| m.rows).sum::<usize>() + extra.len();
        let nnat: usize = ms.iter().filter(|m| native(m)).map(|m| m.rows).sum();
        // mostly-Q8 mixes (fin: 640 of 1793 rows) cost an extra launch per segment for little VRAM: keep them all Q8
        if nnat * 2 < nnat + nq8 && std::env::var("PYRE_QN_NATMIX").map(|v| v != "1").unwrap_or(true) { return self.upload(ms, extra); }
        if !self.take(nq8 * cols * 9 / 8 + nnat * cols / 256 * 140) { return None; }
        let (mut words, mut sc): (Vec<u32>, Vec<f32>) = (vec![], vec![]);
        if !q8.is_empty() { let q = super::qn_gpu::to_q8_0(&q8);
            for blk in q.chunks_exact(34) { sc.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32()); for c in blk[2..].chunks_exact(4) { words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } } }
        let nb = cols / 32;
        for e in extra { for b in 0..nb { let x = &e[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs())); let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            sc.push(d); for c in x.chunks_exact(4) { let qb: Vec<u8> = c.iter().map(|v| ((v * id).round().clamp(-127.0, 127.0) as i8) as u8).collect(); words.push(u32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]])); } } }
        if words.is_empty() { words.push(0); sc.push(0.0); }
        let (mut nat, mut qseg) = (vec![], vec![]); let (mut r0, mut qb) = (0usize, 0usize);
        for m in ms {
            if native(m) {
                let raw = m.raw_rows(0, m.rows); let mut v: Vec<u32> = Vec::with_capacity(raw.len() / 136 * 35);
                for b in raw.chunks_exact(136) { v.push(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32().to_bits()); v.push(u16::from_le_bytes([b[2], b[3]]) as u32); v.push(u32::from_le_bytes([b[4], b[5], b[6], b[7]]));
                    for c in b[8..].chunks_exact(4) { v.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } }
                nat.push((r0, m.rows, self.hup(u32::as_bytes(&v))));
            } else { match qseg.last_mut() { Some((a, n, _)) if *a + *n == r0 => *n += m.rows, _ => qseg.push((r0, m.rows, qb)) } qb += m.rows; }
            r0 += m.rows;
        }
        for _ in extra { match qseg.last_mut() { Some((a, n, _)) if *a + *n == r0 => *n += 1, _ => qseg.push((r0, 1, qb)) } qb += 1; r0 += 1; }
        let mut splits = vec![0]; for m in ms { splits.push(splits.last().unwrap() + m.rows); } for _ in extra { splits.push(splits.last().unwrap() + 1); }
        let kv: Vec<f32> = super::iq_tables::KVALUES_IQ4NL.iter().map(|&v| v as f32).collect();
        let kvh = if native_mode() >= 2 { let kb = |i: usize| (super::iq_tables::KVALUES_IQ4NL[i] as i32 as u32) & 255; let t2: Vec<u32> = (0..256).map(|b| kb(b & 15) | (kb(b >> 4) << 8)).collect(); self.hup(u32::as_bytes(&t2)) } else { self.hup(f32::as_bytes(&kv)) };
        NATIVE_STAT.fetch_add((nnat * cols) as u64, std::sync::atomic::Ordering::Relaxed); NATIVE_STAT.fetch_add(((nq8 * cols) as u64) << 40, std::sync::atomic::Ordering::Relaxed);
        Some(CMat { w: self.hup(u32::as_bytes(&words)), s: self.hup(f32::as_bytes(&sc)), rows: r0, cols, splits, nat, qseg, kv: Some(kvh) })
    }    pub fn buf(&self, v: &[f32]) -> Handle { self.hup(f32::as_bytes(v)) }
    pub fn empty(&self, n: usize) -> Handle { self.client.empty(n * 4) }
    /// CPU reference for the first native segment of m (first 64 rows, 3 columns, f64 from the packed IQ4_XS words),
    /// against gemv_m with iq4d_m and with iq4s_m. The GPU side quantizes x to Q8, so expect ~1e-2 relative error.
    pub fn natcheck(&self, nm: &str, m: &CMat) {
        let Some((r0, n, wh)) = m.nat.first() else { eprintln!("natcheck {nm}: not native"); return };
        let (cols, rows) = (m.cols, m.rows);
        eprintln!("natcheck {nm}: {}x{} nat {:?} qseg {:?}", rows, cols, m.nat.iter().map(|(a, b, _)| (*a, *b)).collect::<Vec<_>>(), m.qseg);
        let wb = self.client.read_one(wh.clone()).expect("gpu readback");
        let w: Vec<u32> = wb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let x: Vec<f32> = (0..3 * cols).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
        let xh = self.buf(&x);
        const LV: [f64; 16] = [-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0];
        let nb = cols / 256; let nr = (*n).min(64);
        let mut rf = vec![0f64; 3 * nr];
        for r in 0..nr { for b in 0..nb { let b0 = r * nb * 35 + b * 35; let d = f32::from_bits(w[b0]) as f64; let (sh, sl) = (w[b0 + 1], w[b0 + 2]);
            for ib in 0..8 { let ls = ((((sl >> (4 * ib)) & 15) | (((sh >> (2 * ib)) & 3) << 4)) as i32 - 32) as f64;
                for k in 0..4 { let q = w[b0 + 3 + ib * 4 + k];
                    for by in 0..4 { let v = (q >> (8 * by)) & 255; let (lo, hi) = (LV[(v & 15) as usize], LV[(v >> 4) as usize]);
                        let cl = b * 256 + ib * 32 + 4 * k + by;
                        for t in 0..3 { rf[t * nr + r] += d * ls * (lo * x[t * cols + cl] as f64 + hi * x[t * cols + cl + 16] as f64); } } } } } }
        for ov in [1u32, 2, 3] {
            NATS_OV.store(ov, std::sync::atomic::Ordering::Relaxed);
            let o = self.empty(3 * rows); self.gemv_m(m, &xh, 3 * cols, 0, cols, &o, 3 * rows, 0, rows, 3); let g = self.read(&o);
            let (mut me, mut mx) = (0f64, 0f64);
            for t in 0..3 { for r in 0..nr { let a = g[t * rows + r0 + r] as f64; me = me.max((a - rf[t * nr + r]).abs()); mx = mx.max(rf[t * nr + r].abs()); } }
            eprintln!("natcheck {nm} {}: maxerr {me:.4e} maxref {mx:.4e}", match ov { 1 => "iq4d_m", 2 => "iq4s_m", _ => "iq4d_r" });
        }
        NATS_OV.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    /// Host-to-device for a small buffer (activations, expert tables): through CubeCL's page-locked staging pool when
    /// PYRE_QN_HPIN=1, so the driver copies by DMA instead of staging a pageable buffer on this thread.
    pub fn hup(&self, b: &[u8]) -> Handle {
        if hpin() && b.len() <= (1 << 20) { let mut by = cubecl::bytes::Bytes::from_bytes_vec(b.to_vec()); self.client.staging(std::iter::once(&mut by), false); return self.client.create(by); }
        self.client.create_from_slice(b)
    }
    /// Start a readback now (its fence follows the work enqueued so far) and finish it later with `read_finish`,
    /// so work enqueued in between does not delay it (the verify wavefront).
    pub fn read_start(&self, h: &Handle) -> impl std::future::Future<Output = Result<Vec<burn_cubecl::cubecl::bytes::Bytes>, burn_cubecl::cubecl::server::ServerError>> + Send + '_ { self.client.read_async(vec![h.clone()]) }
    pub fn read_finish(&self, f: impl std::future::Future<Output = Result<Vec<burn_cubecl::cubecl::bytes::Bytes>, burn_cubecl::cubecl::server::ServerError>>) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        struct W(std::thread::Thread); impl std::task::Wake for W { fn wake(self: std::sync::Arc<Self>) { self.0.unpark(); } }
        let wk = std::task::Waker::from(std::sync::Arc::new(W(std::thread::current()))); let mut cx = std::task::Context::from_waker(&wk);
        let mut f = std::pin::pin!(f);
        let b = loop { if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) { break v.expect("gpu readback").remove(0); } std::thread::park(); };
        let v = f32::from_bytes(&b).to_vec();
        super::qn_gpu::GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); super::qn_gpu::GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
    pub fn read(&self, h: &Handle) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let b = self.client.read_one(h.clone()).expect("gpu readback");
        let v = f32::from_bytes(&b).to_vec();
        super::qn_gpu::GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); super::qn_gpu::GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
    /// out[ooff..ooff+rows] = W x[xoff..xoff+cols]
    pub fn gemv(&self, m: &CMat, x: &Handle, xn: usize, xoff: usize, out: &Handle, on: usize, ooff: usize) {
        if !m.nat.is_empty() && native_mode() == 3 {
            for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4x::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            return;
        }
        if !m.nat.is_empty() && native_mode() == 2 {
            let nb = m.cols / 32; let xq = self.empty(m.cols / 4); let xs = self.empty(nb);
            unsafe { k_q8pack::launch_unchecked(&self.client, CubeCount::Static((nb as u32).div_ceil(64), 1, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(xq.clone(), m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nb), xoff as u32, nb as u32) }
            for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4d::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32) } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            return;
        }
        if !m.nat.is_empty() {
            for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 16),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            return;
        }
        // PYRE_QN_GEMV1R=1: one-column k_gemv4r (4 rows per warp share each activation load; same per-row order)
        if gemv1r() {
            let rpc = gemv_rpc(m.rows); unsafe { k_gemv4r::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, 0u32, ooff as u32, 0u32, 1u32, false, false, false) }
            return;
        }
        unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
            BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
            BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, ooff as u32) }
    }
    /// `gemv` for `nt` activation columns: x[c] at xoff + c*xst (xn = x buffer length), out[c] at ooff + c*ost.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_m(&self, m: &CMat, x: &Handle, xn: usize, xoff: usize, xst: usize, out: &Handle, on: usize, ooff: usize, ost: usize, nt: usize) {
        // native IQ4_XS, 2-4 columns: one activation pack and one launch per native segment for all columns (weights
        // read once; each column's sums as the per-column path), the Q8 rows in one k_gemv4r per segment (the all-Q8
        // path's per-row order). PYRE_QN_NATM=0 reverts to the per-column launches.
        if !m.nat.is_empty() && native_mode() == 2 && (2..=4).contains(&nt) && natm() {
            let nb = m.cols / 32; let xq = self.empty(nt * m.cols / 4); let xs = self.empty(nt * nb);
            unsafe { k_q8pack_m::launch_unchecked(&self.client, CubeCount::Static((nb as u32).div_ceil(64), nt as u32, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), xoff as u32, xst as u32, nb as u32) }
            if nats_on() { for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4s_m::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } } } else {
            for (r0, n, w) in &m.nat { if natr() && NATS_OV.load(std::sync::atomic::Ordering::Relaxed) != 1 { unsafe { k_gemv_iq4d_r::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE * 4), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } continue; }
                unsafe { k_gemv_iq4d_m::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); let rpc = gemv_rpc(*n);
                unsafe { k_gemv4r::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, xst as u32, (ooff + r0) as u32, ost as u32, nt as u32, nt > 1, nt > 2, nt > 3) } }
            return;
        }
        if !m.nat.is_empty() || nt == 1 || std::env::var("PYRE_QN_GEMV4").map(|v| v == "0").unwrap_or(false) {
            for c in 0..nt { self.gemv(m, x, xn, xoff + c * xst, out, on, ooff + c * ost); }
            return;
        }
        let mut c0 = 0;
        // PYRE_QN_GEMV4R=0: the one-row-per-warp k_gemv4
        if std::env::var("PYRE_QN_GEMV4R").map(|v| v == "8").unwrap_or(false) {
            while c0 < nt { let n = (nt - c0).min(4);
                unsafe { k_gemv4r8::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE * 8), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                    BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                    BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32) }
                c0 += n; }
            return;
        }
        // PYRE_QN_GEMV4R=v: k_gemv4v (vector activation loads; needs 4-aligned offsets)
        if std::env::var("PYRE_QN_GEMV4R").map(|v| v == "v").unwrap_or(false) && xoff % 4 == 0 && xst % 4 == 0 && m.cols % 4 == 0 {
            while c0 < nt { let n = (nt - c0).min(4);
                unsafe { k_gemv4v::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE * 4), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE), 4,
                    BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                    BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32, n > 1, n > 2, n > 3) }
                c0 += n; }
            return;
        }
        if std::env::var("PYRE_QN_GEMV4R").map(|v| v != "0").unwrap_or(true) {
            while c0 < nt { let n = (nt - c0).min(4);
                let rpc = gemv_rpc(m.rows); unsafe { k_gemv4r::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                    BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                    BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32, n > 1, n > 2, n > 3) }
                c0 += n; }
            return;
        }
        while c0 < nt { let n = (nt - c0).min(4);
            unsafe { k_gemv4::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32) }
            c0 += n; }
    }
    /// rms of t tokens (separate x buffers) into xn at token stride hc*d; one launch for up to 4 (PYRE_QN_RMS4=0: per token)
    pub fn rms_t(&self, xs: &[Handle], norm: &Handle, xn: &Handle, hc: usize, d: usize, eps: f32) {
        let t = xs.len(); let w = hc * d;
        if t >= 2 && t <= 4 && std::env::var("PYRE_QN_RMS4").map(|v| v != "0").unwrap_or(true) {
            let g = |i: usize| xs[i.min(t - 1)].clone();
            unsafe { k_rms4::launch_unchecked(&self.client, CubeCount::Static(hc as u32, t as u32, 1), CubeDim::new_1d(256),
                BufferArg::from_raw_parts(g(0), w), BufferArg::from_raw_parts(g(1), w), BufferArg::from_raw_parts(g(2), w), BufferArg::from_raw_parts(g(3), w),
                BufferArg::from_raw_parts(norm.clone(), w), BufferArg::from_raw_parts(xn.clone(), t * w), d as u32, w as u32, eps) }
            return;
        }
        for (ti, x) in xs.iter().enumerate() { self.rms(x, norm, &xn.clone().offset_start((ti * w * 4) as u64), hc, d, eps); }
    }
    pub fn rms(&self, x: &Handle, norm: &Handle, xn: &Handle, hc: usize, d: usize, eps: f32) {
        unsafe { k_rms::launch_unchecked(&self.client, CubeCount::Static(hc as u32, 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(norm.clone(), hc * d), BufferArg::from_raw_parts(xn.clone(), hc * d), d as u32, eps) }
    }
    /// k_gate over t tokens at strides xs / ds / os (elements); false when t is not 2..=4 (caller loops k_gate)
    pub fn gate_m(&self, xn: &Handle, dn: &Handle, up: &CMat, out: &Handle, on: usize, ooff: usize, hc: usize, d: usize, t: usize, xs: usize, ds: usize, os: usize) -> bool {
        if !(2..=4).contains(&t) { return false; }
        let lr = up.cols;
        unsafe { k_gate_m::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8),
            BufferArg::from_raw_parts(xn.clone(), t * xs), BufferArg::from_raw_parts(dn.clone(), t * ds), BufferArg::from_raw_parts(up.w.clone(), up.rows * lr / 4),
            BufferArg::from_raw_parts(up.s.clone(), up.rows * lr / 32), BufferArg::from_raw_parts(out.clone(), on), d as u32, ooff as u32, xs as u32, ds as u32, os as u32, lr, hc as u32, t as u32) }
        true
    }
    pub fn gate(&self, xn: &Handle, dn: &Handle, up: &CMat, out: &Handle, on: usize, ooff: usize, hc: usize, d: usize) {
        let lr = up.cols;
        unsafe { k_gate::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8),
            BufferArg::from_raw_parts(xn.clone(), hc * d), BufferArg::from_raw_parts(dn.clone(), lr + hc), BufferArg::from_raw_parts(up.w.clone(), up.rows * lr / 4),
            BufferArg::from_raw_parts(up.s.clone(), up.rows * lr / 32), BufferArg::from_raw_parts(out.clone(), on), d as u32, ooff as u32, lr, hc as u32) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn comb(&self, x: &Handle, a: &Handle, b: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) { self.comb3(x, a, b, None, g, dn, lr, hc, d) }
    #[allow(clippy::too_many_arguments)]
    pub fn comb3(&self, x: &Handle, a: &Handle, b: Option<&Handle>, c3: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) {
        let (gb, gn, goff) = g.map(|(h, n, o)| (h.clone(), n, o)).unwrap_or((a.clone(), d, 0));
        let bb = b.cloned().unwrap_or_else(|| a.clone()); let cc = c3.cloned().unwrap_or_else(|| a.clone());
        unsafe { k_comb::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(a.clone(), d), BufferArg::from_raw_parts(bb, d), BufferArg::from_raw_parts(cc, d), BufferArg::from_raw_parts(gb, gn),
            BufferArg::from_raw_parts(dn.clone(), lr + hc), d as u32, goff as u32, lr as u32, hc as u32, g.is_some(), b.is_some(), c3.is_some()) }
    }
    pub fn conv(&self, proj: &Handle, pn: usize, w: &Handle, cs: &Handle, co: &Handle, ch: usize, kk: usize) {
        unsafe { k_conv::launch_unchecked(&self.client, CubeCount::Static((ch as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(proj.clone(), pn), BufferArg::from_raw_parts(w.clone(), ch * kk), BufferArg::from_raw_parts(cs.clone(), ch * (kk - 1)),
            BufferArg::from_raw_parts(co.clone(), ch), ch as u32, kk as u32) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn delta(&self, co: &Handle, ch: usize, proj: &Handle, pn: usize, w: &GdnW, s: &Handle, out: &Handle, nk: usize, nv: usize, dk: usize, zoff: usize, boff: usize, aoff: usize, eps: f32) {
        unsafe { k_delta::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
            BufferArg::from_raw_parts(co.clone(), ch), BufferArg::from_raw_parts(proj.clone(), pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
            BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), nv * dk * dk), BufferArg::from_raw_parts(out.clone(), nv * dk),
            nk as u32, nv as u32, zoff as u32, boff as u32, aoff as u32, eps, dk as u32) }
    }
    /// `delta` for nt (1..=4) tokens: co/proj/out at strides ch/pn/ostr; checkpoint t (< 3) written to cks[t] when bit t of ckm is set.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_m(&self, co: &Handle, ch: usize, proj: &Handle, pn: usize, w: &GdnW, s: &Handle, out: &Handle, ostr: usize, cks: &[Option<Handle>], ckm: u32, nk: usize, nv: usize, dk: usize, zoff: usize, boff: usize, aoff: usize, eps: f32, nt: usize) {
        let sn = nv * dk * dk; let dummy = self.empty(1);
        let ck = |i: usize| match cks.get(i) { Some(Some(h)) if (ckm >> i) & 1 == 1 => (h.clone(), sn), _ => (dummy.clone(), 1) };
        let (k0, k1, k2, k3) = (ck(0), ck(1), ck(2), ck(3));
        unsafe { k_delta_m::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
            BufferArg::from_raw_parts(co.clone(), nt * ch), BufferArg::from_raw_parts(proj.clone(), nt * pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
            BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), sn), BufferArg::from_raw_parts(out.clone(), nt * ostr),
            BufferArg::from_raw_parts(k0.0, k0.1), BufferArg::from_raw_parts(k1.0, k1.1), BufferArg::from_raw_parts(k2.0, k2.1), BufferArg::from_raw_parts(k3.0, k3.1),
            nk as u32, ch as u32, pn as u32, ostr as u32, ckm, zoff as u32, boff as u32, aoff as u32, eps, dk as u32, nt as u32) }
    }
    /// Decode attention for one token (see k_attn_prep / k_attn); `p` = the fused q|k|v projection.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(&self, p: &Handle, pn: usize, ko: usize, vo: usize, qn: &Handle, kn: &Handle, cs: &Handle, kc: &Handle, vc: &Handle, cap: usize, out: &Handle, nh: usize, nkv: usize, hd: usize, half: usize, pos: usize, eps: f32) {
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        let q = self.empty(nh * hd); let sc = self.empty(nh * cap);
        unsafe {
            k_attn_prep::launch_unchecked(&self.client, CubeCount::Static((nh + nkv) as u32, 1, 1), CubeDim::new_1d(32),
                a(p, pn), a(qn, hd), a(kn, hd), a(cs, 2 * half), a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), ko as u32, vo as u32, nh as u32, nkv as u32, hd as u32, half as u32, pos as u32, eps);
            k_attn::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(32),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip());
        }
    }    pub fn xtabs(&self) -> XTabs {
        use super::iq_tables::*;
        let w64 = |t: &[u64]| -> Vec<u32> { t.iter().flat_map(|v| [*v as u32, (*v >> 32) as u32]).collect() };
        let ks: Vec<u32> = KSIGNS_IQ2XS.iter().map(|&v| v as u32).collect();
        XTabs { g2s: self.hup(u32::as_bytes(&w64(&IQ2S_GRID))), g2x: self.hup(u32::as_bytes(&w64(&IQ2XXS_GRID))),
            ks: self.hup(u32::as_bytes(&ks)), g1: self.hup(u32::as_bytes(&w64(&IQ1S_GRID))),
            sm: self.hup(u32::as_bytes(&(0..256u32).flat_map(|b| [0u32, 4].map(|o| (0..4).map(|j| if (b >> (o + j)) & 1 == 1 { 0xFFu32 << (8 * j) } else { 0 }).sum::<u32>())).collect::<Vec<u32>>())),
            q2l: self.hup(u32::as_bytes(&(0..256u32).map(|b| (0..4).map(|j| ((b >> (2 * j)) & 3) << (8 * j)).sum::<u32>()).collect::<Vec<u32>>())) }
    }
    pub fn xarena(&self, bytes: usize) -> Handle { self.client.empty(bytes) }
    /// Repack one expert into arena slot `slot` (slot_bytes each; the arena is ar_bytes long).
    #[allow(clippy::too_many_arguments)]
    pub fn xupload(&self, ar: &Handle, ar_bytes: usize, slot: usize, slot_bytes: usize, g: &[u8], u: &[u8], d: &[u8], tys: (u32, u32, u32)) -> XEnt {
        let t0 = std::time::Instant::now();
        let mut v = repack(tys.0, g); let uo = v.len(); v.extend(repack(tys.1, u)); let dof = v.len(); v.extend(repack(tys.2, d));
        let t1 = std::time::Instant::now();
        let base = slot * slot_bytes; let len = v.len(); assert!(len <= slot_bytes && base + len <= ar_bytes);
        let hd = ar.clone().offset_start(base as u64).offset_end((ar_bytes - base - len) as u64);
        self.client.write(&hd, cubecl::bytes::Bytes::from_elems(v));
        XUP_NS[0].fetch_add((t1 - t0).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); XUP_NS[1].fetch_add(t1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let b = (base / 4) as u32;
        XEnt { off: [b, b + uo as u32, b + dof as u32], arena: 0, slot, bytes: slot_bytes }
    }
    /// Upload one pre-packed expert (`xpacked`: gate words, then up at `uo`, down at `dof`) into slot `slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn xupload_pk(&self, ar: &Handle, ar_bytes: usize, slot: usize, slot_bytes: usize, v: &cubecl::bytes::Bytes, uo: usize, dof: usize) -> XEnt {
        let t0 = std::time::Instant::now();
        let base = slot * slot_bytes; let len = v.len(); assert!(len <= slot_bytes && base + len <= ar_bytes);
        let hd = ar.clone().offset_start(base as u64).offset_end((ar_bytes - base - len) as u64);
        // pinned staging (CubeCL's page-locked pool) so the copy is a real async DMA, like Strata's staging memcpyAsync
        let mut by = v.clone(); if xpin() { self.client.staging(std::iter::once(&mut by), false); }
        if xs2() {
            // PYRE_QN_XS2=1: issue the copy from a dedicated thread, which CubeCL gives its own CUDA stream, so the
            // transfer overlaps the compute stream (Strata copies on a separate stream); CubeCL orders the later
            // kernel that reads the slot after it. This thread waits only until the copy is enqueued.
            type Job = (Handle, cubecl::bytes::Bytes, std::sync::mpsc::Sender<()>);
            static TX: std::sync::OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<Job>>> = std::sync::OnceLock::new();
            let tx = TX.get_or_init(|| { let (tx, rx) = std::sync::mpsc::channel::<Job>(); let cl = self.client.clone();
                std::thread::Builder::new().name("pyre-xup".into()).spawn(move || { for (h, b, ack) in rx { cl.write(&h, b); let _ = cl.flush(); let _ = ack.send(()); } }).expect("upload thread");
                std::sync::Mutex::new(tx) });
            let (at, ar_) = std::sync::mpsc::channel(); tx.lock().unwrap().send((hd, by, at)).expect("upload thread gone"); let _ = ar_.recv();
        } else { self.client.write(&hd, by); }
        XUP_NS[1].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let b = (base / 4) as u32;
        XEnt { off: [b, b + uo as u32, b + dof as u32], arena: 0, slot, bytes: slot_bytes }
    }
    /// All cached experts of one layer: h[e] = silu(G_e x) * (U_e x) -> Q8 grid -> y += sum_e w_e D_e h[e].
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_b(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, ents: &[(&XEnt, f32)], x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff));
        let ne = ents.len();
        let meta: Vec<u32> = ents.iter().flat_map(|(e, w)| [e.off[0], e.off[1], e.off[2], w.to_bits()]).collect();
        let mh = self.hup(u32::as_bytes(&meta));
        self.xrun_m(tb, ar, ar_bytes, &mh, ne, x, d, ff, tys, h, y, 0);
    }
    /// The dp4a expert path (gate/up IQ2_S or IQ2_XXS, down Q2_0); false when the types don't fit.
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_q(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, mh: &Handle, ne: usize, x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle) -> bool { self.xrun_qs(tb, ar, ar_bytes, mh, ne, x, d, ff, tys, h, y, 7) }
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_qs(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, mh: &Handle, ne: usize, x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle, st: u32) -> bool {
        if tys.0 > 1 || tys.1 > 1 || tys.2 != 3 || d % 32 != 0 || ff % 32 != 0 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let ml = ne * 4; let aw = ar_bytes / 4; let rpc = 8u32;
        let (xq, xs) = (self.client.empty(d), self.client.empty(d / 8)); let (hq, hs) = (self.client.empty(ne * ff), self.client.empty(ne * ff / 8));
        unsafe {
            if st & 1 != 0 { k_q8pack::launch_unchecked(&self.client, CubeCount::Static((d as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(x, d), a(&xq, d / 4), a(&xs, d / 32), 0, (d / 32) as u32);
            if d % 256 == 0 && tys.0 == tys.1 && xsh() { let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) }; let rp = xshr();
                k_xgu_s::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rp), ne as u32, 1), CubeDim::new_2d(32, rp),
                a(&xq, d / 4), a(&xs, d / 32), a(ar, aw), a(ar, aw), a(mh, ml), a(gsrc, gn as usize), a(h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, (d / 256) as u32, gn, tys.0); }
            else if d % 256 == 0 && xrw() > 0 { let rw = xrw(); let nw = (ff as u32).div_ceil(rw); k_xgu_q4::launch_unchecked(&self.client, CubeCount::Static(nw.div_ceil(4), ne as u32, 1), CubeDim::new_2d(32, 4),
                a(&xq, d / 4), a(&xs, d / 32), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.sm, 512), a(h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, (d / 256) as u32, rw, tys.0, tys.1); }
            else { k_xgu_q::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rpc), ne as u32, 1), CubeDim::new_2d(32, rpc),
                a(&xq, d / 4), a(&xs, d / 32), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.sm, 512), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); } }
            if st & 4 != 0 { k_q8pack::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(h, ne * ff), a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), 0, (ne * ff / 32) as u32); }
            if st & 2 != 0 { k_xdown_q::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), 1, 1), CubeDim::new_2d(32, rpc),
                a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), a(ar, aw), a(ar, aw), a(mh, ml), a(&tb.q2l, 256), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, 1u32); }
        }
        true
    }
    pub fn xscratch(&self, d: usize, ff: usize, max_ne: usize) -> XScr {
        XScr { xq: self.client.empty(d), xs: self.client.empty(d / 8), hq: self.client.empty(max_ne * ff), hs: self.client.empty(max_ne * ff / 8), h: self.client.empty(4 * max_ne * ff), y: self.client.empty(4 * d), max_ne }
    }
    /// One layer's cached experts from one arena: x is read straight from `src` (its first d values),
    /// y = sum_e w_e D_e silu(G_e x)*(U_e x) is written to scr.y. Four launches, one small upload.
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_fast(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, ents: &[(&XEnt, f32)], src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), scr: &XScr, acc: bool) -> bool { self.xrun_fast_to(tb, std::slice::from_ref(ar), ar_bytes, ents, src, sn, d, ff, tys, scr, acc, &scr.y) }
    /// xrun_fast into a caller-owned output (multi-token verify keeps one y per token; scratch is reused in stream order).
    #[allow(clippy::too_many_arguments)]
    /// The per-expert table xrun_fast_to uploads (gate/up/down offsets, bit 31 = second arena, weight).
    pub fn xmeta(ars: &[Handle], ents: &[(&XEnt, f32)]) -> Vec<u32> {
        let fl = |e: &XEnt| if ars.len() == 2 && e.arena == 1 { 1u32 << 31 } else { 0 };
        ents.iter().flat_map(|(e, w)| [e.off[0] | fl(e), e.off[1], e.off[2] | fl(e), w.to_bits()]).collect()
    }
    pub fn xrun_fast_to(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, ents: &[(&XEnt, f32)], src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), scr: &XScr, acc: bool, yo: &Handle) -> bool {
        self.xrun_fast_to_m(tb, ars, ar_bytes, ents, src, sn, d, ff, tys, scr, acc, yo, None)
    }
    /// xrun_fast_to with its table already on the device (`mh`, e.g. a slice of one upload for the whole window):
    /// each small host write costs tens of microseconds on the stream, so a window's tables go up in one.
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_fast_to_m(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, ents: &[(&XEnt, f32)], src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), scr: &XScr, acc: bool, yo: &Handle, mhp: Option<Handle>) -> bool {
        let ne = ents.len();
        if ne == 0 || ne > scr.max_ne || tys.0 != tys.1 || tys.0 > 1 || tys.2 != 3 || d % 256 != 0 || ff % 32 != 0 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let aw = ar_bytes / 4; let rpc = 8u32;
        // up to two arenas in one launch: bit 31 of the gate and down offsets selects the second (with one arena, all go to it)
        if ars.is_empty() || ars.len() > 2 { return false; }
        let mh = match mhp { Some(h) => h, None => self.hup(u32::as_bytes(&Self::xmeta(ars, ents))) }; let ml = ne * 4;
        let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) }; let rp = xshr();
        let (ar, ar1) = (&ars[0], ars.last().unwrap());
        unsafe {
            k_q8pack::launch_unchecked(&self.client, CubeCount::Static((d as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(src, sn), a(&scr.xq, d / 4), a(&scr.xs, d / 32), 0, (d / 32) as u32);
            k_xgu_s::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rp), ne as u32, 1), CubeDim::new_2d(32, rp),
                a(&scr.xq, d / 4), a(&scr.xs, d / 32), a(ar, aw), a(ar1, aw), a(&mh, ml), a(gsrc, gn as usize), a(&scr.h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, (d / 256) as u32, gn, tys.0);
            k_q8pack::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(&scr.h, ne * ff), a(&scr.hq, ne * ff / 4), a(&scr.hs, ne * ff / 32), 0, (ne * ff / 32) as u32);
            k_xdown_q::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), 1, 1), CubeDim::new_2d(32, rpc),
                a(&scr.hq, ne * ff / 4), a(&scr.hs, ne * ff / 32), a(ar, aw), a(ar1, aw), a(&mh, ml), a(&tb.q2l, 256), a(yo, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, acc as u32);
        }
        true
    }
    /// gate-up + q8 + down with an explicit kernel choice (xbench).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn xrun_sel(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, mh: &Handle, ne: usize, x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle, old: bool) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let ml = ne * 4; let aw = ar_bytes / 4; let rpc = 8u32;
        let cc = CubeCount::Static((ff as u32).div_ceil(rpc), ne as u32, 1); let cd = CubeCount::Static((d as u32).div_ceil(rpc), 1, 1);
        if old { k_xgu_b::launch_unchecked(&self.client, cc, CubeDim::new_2d(32, rpc), a(x, d), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); }
        else { k_xgu_b4::launch_unchecked(&self.client, cc, CubeDim::new_2d(32, rpc), a(x, d), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); }
        k_q8sim::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8), a(h, ne * ff), (ne * ff) as u32);
        if old { k_xdown_b::launch_unchecked(&self.client, cd, CubeDim::new_2d(32, rpc), a(h, ne * ff), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, tys.2); }
        else { k_xdown_b4::launch_unchecked(&self.client, cd, CubeDim::new_2d(32, rpc), a(h, ne * ff), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, tys.2); }
    }
    /// xrun_b with the per-expert meta already on the GPU; skip bits 1/2/4 drop gate-up / q8 / down (benchmarks).
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_m(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, mh: &Handle, ne: usize, x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle, skip: u32) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let ml = ne * 4;
        let aw = ar_bytes / 4;
        static RPC: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); let rpc = *RPC.get_or_init(|| std::env::var("PYRE_QN_XRPC").ok().and_then(|v| v.parse().ok()).unwrap_or(8));
        unsafe {
            if skip == 0 && xkq() && self.xrun_q(tb, ar, ar_bytes, mh, ne, x, d, ff, tys, h, y) { return; }
            let v4 = xk4(); if skip & 1 == 0 && v4 { k_xgu_b4::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rpc), ne as u32, 1), CubeDim::new_2d(32, rpc),
                a(x, d), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); }
            if skip & 1 == 0 && !v4 { k_xgu_b::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rpc), ne as u32, 1), CubeDim::new_2d(32, rpc),
                a(x, d), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); }
            if skip & 2 == 0 { k_q8sim::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8), a(h, ne * ff), (ne * ff) as u32); }
            if skip & 4 == 0 && v4 { k_xdown_b4::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), 1, 1), CubeDim::new_2d(32, rpc),
                a(h, ne * ff), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, tys.2); }
            if skip & 4 == 0 && !v4 { k_xdown_b::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), 1, 1), CubeDim::new_2d(32, rpc),
                a(h, ne * ff), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, tys.2); }
        }
    }    /// dst = Q8-grid copy of src[soff..soff+n]; z[..n] = 0.
    pub fn q8copy(&self, src: &Handle, sn: usize, soff: usize, dst: &Handle, z: &Handle, n: usize) {
        unsafe { k_q8copy::launch_unchecked(&self.client, CubeCount::Static((n as u32 / 32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8),
            BufferArg::from_raw_parts(src.clone(), sn), BufferArg::from_raw_parts(dst.clone(), n), BufferArg::from_raw_parts(z.clone(), n), soff as u32, n as u32) }
    }
    /// dst[dof..dof+n] = src[so..so+n] (srcn / dstn: buffer lengths in floats)
    #[allow(clippy::too_many_arguments)]
    pub fn copy(&self, src: &Handle, srcn: usize, so: usize, dst: &Handle, dstn: usize, dof: usize, n: usize) {
        unsafe { k_copy::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(src.clone(), srcn), BufferArg::from_raw_parts(dst.clone(), dstn), so as u32, dof as u32, n as u32) }
    }
    pub fn silu_mul(&self, f: &Handle, fnn: usize, og: usize, ou: usize, h: &Handle, n: usize) {
        unsafe { k_silu_mul::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(f.clone(), fnn), BufferArg::from_raw_parts(h.clone(), n), og as u32, ou as u32, n as u32) }
    }
}

/// A hyper-connection mixer: norm weights, down(+inject) rows, up rows.
pub struct CHc { pub norm: Handle, pub di: CMat, pub up: CMat }
/// One layer's dense half (see `Model::layer_cube`). `fin` = router | shared gate | shared up | shared-gate row.
pub struct GdnW { pub conv: Handle, pub a: Handle, pub dt: Handle, pub norm: Handle }
pub struct CLayer { pub an: Option<(Handle, Handle)>, pub gdn: Option<GdnW>, pub hca: CHc, pub min: CMat, pub mout: CMat, pub hcf: CHc, pub fin: CMat, pub sdown: CMat }

/// PYRE_QN_XBENCH: time the batched expert kernels on random IQ2_XXS/IQ2_XXS/Q2_0 experts (d 2560, ff 640),
/// old vs new, and compare their outputs.
pub fn xbench() -> anyhow::Result<()> {
    let cb = Cube::new()?; let tb = cb.xtabs();
    let (d, ff) = (2560usize, 640usize); for tys in [(1u32, 1u32, 3u32), (0u32, 0u32, 3u32)] {
    let mut rng = 0x9e3779b97f4a7c15u64; let mut rb = |n: usize, blk: usize| -> Vec<u8> { (0..n).map(|i| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; if i % blk < 2 { if i % blk == 0 { 0x00 } else { 0x20 } } else { rng as u8 } }).collect() };
    let nexp = 40usize; let g = |n: usize| n; let _ = g;
    let (gbk, gw) = if tys.0 == 0 { (82usize, 21usize) } else { (66, 17) }; let gb = ff * d / 256 * gbk; let db = d * ff / 64 * 18;
    let slot = ((2 * ff * d / 256 * gw + d * ff / 64 * 5) * 4).next_multiple_of(256);
    let ar_bytes = slot * nexp; let ar = cb.xarena(ar_bytes);
    let ents: Vec<XEnt> = (0..nexp).map(|i| cb.xupload(&ar, ar_bytes, i, slot, &rb(gb, gbk), &rb(gb, gbk), &rb(db, 18), tys)).collect();
    let xs: Vec<f32> = (0..d).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect(); let x = cb.buf(&xs);
    let bytes_per = (gb * 2 + db) as f64;
    for ne in [4usize, 10, 20] {
        let sel: Vec<(&XEnt, f32)> = (0..ne).map(|i| (&ents[(i * 7) % nexp], 0.1 + i as f32 * 0.01)).collect();
        let mut outs = vec![];
        for v in ["1", "q"] {
            std::env::set_var("PYRE_QN_XK", v);
            let meta: Vec<u32> = sel.iter().flat_map(|(e, w)| [e.off[0], e.off[1], e.off[2], w.to_bits()]).collect();
            let mh = cb.client.create_from_slice(u32::as_bytes(&meta));
            let h = cb.empty(ne * ff); let y = cb.buf(&vec![0f32; d]);
            // xk4 is cached per process: call the kernels through the explicit selector
            let run = |old: bool| unsafe { if old { cb.xrun_sel(&tb, &ar, ar_bytes, &mh, ne, &x, d, ff, tys, &h, &y, true) } else { assert!(cb.xrun_q(&tb, &ar, ar_bytes, &mh, ne, &x, d, ff, tys, &h, &y)); } };
            cb.client.write(&y, cubecl::bytes::Bytes::from_elems(vec![0f32; d])); run(v == "1"); let r = cb.read(&y); outs.push(r);
            let t0 = std::time::Instant::now(); for _ in 0..200 { run(v == "1"); } cb.read(&y);
            let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
            eprintln!("xbench ne={ne} {}: {us:.1} us ({:.0} GB/s)", if v == "1" { "old" } else { "new" }, bytes_per * ne as f64 / us / 1e3);
        }
        { let meta: Vec<u32> = sel.iter().flat_map(|(e, w)| [e.off[0], e.off[1], e.off[2], w.to_bits()]).collect(); let mh = cb.client.create_from_slice(u32::as_bytes(&meta)); let h = cb.empty(ne * ff); let y = cb.buf(&vec![0f32; d]);
          for (st, nm) in [(1u32, "pack+gu"), (4, "pack h"), (2, "down"), (0, "none")] { cb.xrun_qs(&tb, &ar, ar_bytes, &mh, ne, &x, d, ff, tys, &h, &y, st); cb.read(&y); let t0 = std::time::Instant::now(); for _ in 0..200 { cb.xrun_qs(&tb, &ar, ar_bytes, &mh, ne, &x, d, ff, tys, &h, &y, st); } cb.read(&y); eprintln!("xbench   stage {nm}: {:.1} us", t0.elapsed().as_secs_f64() * 1e6 / 200.0); } }
        let md = outs[0].iter().zip(&outs[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max); let mx = outs[0].iter().fold(0f32, |a, b| a.max(b.abs()));
        eprintln!("xbench tys={tys:?} ne={ne}: max |old-new| {md:.3e} (max |y| {mx:.3e})");
    } }
    Ok(())
}

/// xupload time split: [repack on the host, client.write] in ns.
pub static XUP_NS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
