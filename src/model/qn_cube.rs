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
/// k_gemv4s shared-memory activation capacity (f32): nt * cols must fit (48 KB)
const GEMV4S_MAX: u32 = 12288;
fn gemv_rpc(rows: usize) -> u32 { static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); let thr = *T.get_or_init(|| std::env::var("PYRE_QN_GEMV_RPC").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)); if rows >= thr { ROWS_PER_CUBE } else if rows >= 1024 { 4 } else { 2 } }
/// dense IQ4_XS kernel: PYRE_QN_NATIVE=0 Q8_0 repack, 1 f32 IQ4 kernel, 2 dp4a IQ4 kernel (int8 activations), 3 = 2 with x quantised in-kernel
/// PYRE_QN_NATS=0: the per-8-values k_gemv_iq4d_m instead of the sub-block k_gemv_iq4s_m
/// PYRE_QN_NATX=1: every native IQ4_XS GEMV (1-4 columns) as Strata / ggml vec_dot_iq4_xs_q8_1 arithmetic (sub-block integer sums, half activation scale)
/// PYRE_QN_GATEM2 (default on, =0 off): k_gate_m2 (one warp per (output, stream), identical results)
pub fn gatem2() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_GATEM2").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_GEMV4RP=1 (opt-in): k_gemv4rp prefetching Q8 GEMV for < 1024 rows (identical results, no measured gain e69)
pub fn gemv4rp() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_GEMV4RP").map(|v| v == "1").unwrap_or(false)) }
pub fn natx() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATX").map(|v| v == "1").unwrap_or(false)) }
pub fn nats() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATS").map(|v| v == "1").unwrap_or(false)) }
/// Test override for nats(): 0 = env, 1 = iq4d_m, 2 = iq4s_m.
pub static NATS_OV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn nats_on() -> bool { match NATS_OV.load(std::sync::atomic::Ordering::Relaxed) { 1 | 3 => false, 2 => true, _ => nats() } }
/// PYRE_QN_NATR_MIN=rows: segments with fewer rows use k_gemv_iq4d_m (1 row per warp, more warps; same per-row arithmetic)
pub fn natr_min() -> usize { static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) }
/// PYRE_QN_READNB=1: read_start queues the readback without waiting for the device thread to drain
pub fn readnb() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_READNB").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_NATR_PRMT=1: k_gemv_iq4d_r2 decodes levels with byte permutes instead of the shared table (same integers)
pub fn natr_prmt() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR_PRMT").map(|v| v == "1").unwrap_or(false)) }
/// PYRE_QN_NATR_U: blocks whose loads k_gemv_iq4d_r2 issues together (default 2; 1 = the old k_gemv_iq4d_r; 2 or 4). Bit-identical.
pub fn natr_u() -> u32 { static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR_U").ok().and_then(|v| v.parse().ok()).map(|v: u32| if v >= 4 { 4 } else if v >= 2 { 2 } else { 1 }).unwrap_or(2)) }
/// PYRE_QN_NATR3=rows (1, 2, 4 or 8; default 4; 0 = k_gemv_iq4d_r2): k_gemv_iq4d_r3 (software-pipelined loads after Strata's il_pf kernel). Bit-identical to r2.
pub fn natr3() -> u32 { static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR3").ok().and_then(|v| v.parse().ok()).map(|v: u32| if v >= 8 { 8 } else if v >= 4 { 4 } else if v >= 2 { 2 } else { v }).unwrap_or(4)) }
/// PYRE_QN_NATR3_PL=0: k_gemv_iq4d_r3 decodes levels with the shared table instead of byte permutes (same integers; default permutes, s2: 119 -> 114 us)
pub fn natr3_pl() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR3_PL").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_DELTA2 (default on, =0 k_delta_m): k_delta_m2 (one state pass per token, identical results; s3: 64.5 -> 56.3 us T=3)
pub fn delta2() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_DELTA2").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_NATR_Y: warps per cube for k_gemv_iq4d_r (default 4; b307 sweep -> 4 best)
pub fn natr_y() -> u32 { static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR_Y").ok().and_then(|v| v.parse().ok()).unwrap_or(4)) }
pub fn natr() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATR").map(|v| v != "0").unwrap_or(true)) }
pub fn natm() -> bool { static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATM").map(|v| v != "0").unwrap_or(true)) }
pub fn native_mode() -> u32 { static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *M.get_or_init(|| std::env::var("PYRE_QN_NATIVE").ok().and_then(|v| v.parse().ok()).unwrap_or(if std::env::var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false) { 2 } else { 0 })) }
/// NATIVE=2 profile (explicit PYRE_QN_NATIVE=2, or the default under PYRE_QN_XEXACT=1: IQ4_XS dense + LM head stay native)
pub fn native2() -> bool { native_mode() == 2 }
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

/// PYRE_QN_BPF prompt-chunk GEMM (batched prefill): out[c * ost + ooff + r] = sum_k W[r, k] * x[(c0 + c) * cols + k] for a
/// tile of 64 rows x 64 tokens per cube (256 units, 4x4 outputs each). Every 32-column step dequantizes the tile's weights
/// (Q8_0 words + per-32 scales, or native IQ4_XS blocks when `nat`) into shared memory once for all 64 tokens, so a chunk
/// reads each dense weight once per 64 tokens instead of once per 1-4 (gemv_m). Technique (weights read once per prompt
/// chunk, shared-memory tiles) as in Strata's prefill (MIT, github.com/Niko1221/Strata); this kernel is pyre's own.
#[cube(launch_unchecked)]
fn k_gemm_pf(x: &[f32], w: &[u32], s: &[f32], kv: &[f32], out: &mut [f32], rows: u32, cols: u32, nt: u32, kch: u32, ooff: u32, ost: u32, pst: u32, xst: u32, #[comptime] nat: bool) {
    // tile 128 rows x 64 tokens, 256 units = 16 (rows, 8 each) x 16 (tokens, 4 each); split-K over CUBE_POS_Z (kch columns
    // each, partial z written at out + z * pst). Shared: [k][row] 129 per k, [k][token] 65 per k (padding vs bank conflicts)
    let mut ws = Shared::<[f32]>::new_slice(4128usize);
    let mut xs = Shared::<[f32]>::new_slice(2080usize);
    let mut tab = Shared::<[f32]>::new_slice(16usize);
    let tid = UNIT_POS;
    if nat { if tid < 16u32 { tab[tid as usize] = kv[tid as usize]; } }
    let r0 = CUBE_POS_X * 128u32; let t0 = CUBE_POS_Y * 64u32;
    let ty = tid / 16u32; let tx = tid % 16u32;
    let mut acc = Array::<f32>::new(32usize);
    #[unroll]
    for i in 0..32u32 { acc[i as usize] = 0.0f32; }
    let mut wv = Array::<f32>::new(8usize); let mut xv = Array::<f32>::new(4usize);
    let lr = tid / 2u32; let lh = tid % 2u32;
    let mut grow = r0 + lr; if grow >= rows { grow = rows - 1u32; }
    let xr = tid / 4u32; let lq = tid % 4u32;
    let mut tk = t0 + xr; if tk >= nt { tk = nt - 1u32; }
    let kb = CUBE_POS_Z * kch; let mut ke = kb + kch; if ke > cols { ke = cols; }
    let mut k0 = kb;
    while k0 < ke {
        sync_cube();
        if nat {
            let blk = k0 / 256u32; let ib = (k0 / 32u32) % 8u32;
            let b = grow * (cols / 256u32) * 35u32 + blk * 35u32;
            let d = f32::reinterpret(w[b as usize]); let sh = w[(b + 1u32) as usize]; let sl = w[(b + 2u32) as usize];
            let ls = ((sl >> (4u32 * ib)) & 15u32) | (((sh >> (2u32 * ib)) & 3u32) << 4u32);
            let dl = d * (f32::cast_from(ls) - 32.0f32);
            #[unroll]
            for h in 0..2u32 {
                let wq = 2u32 * lh + h;
                let q = w[(b + 3u32 + 4u32 * ib + wq) as usize];
                #[unroll]
                for j in 0..4u32 {
                    ws[((4u32 * wq + j) * 129u32 + lr) as usize] = dl * tab[((q >> (8u32 * j)) & 15u32) as usize];
                    ws[((16u32 + 4u32 * wq + j) * 129u32 + lr) as usize] = dl * tab[((q >> (8u32 * j + 4u32)) & 15u32) as usize];
                }
            }
        } else {
            let sc = s[(grow * (cols / 32u32) + k0 / 32u32) as usize];
            let wb = grow * (cols / 4u32) + k0 / 4u32 + 4u32 * lh;
            #[unroll]
            for h in 0..4u32 {
                let wd = w[(wb + h) as usize];
                #[unroll]
                for j in 0..4u32 { ws[((16u32 * lh + 4u32 * h + j) * 129u32 + lr) as usize] = sc * dec(wd, j); }
            }
        }
        let xb = tk * xst + k0 + 8u32 * lq;
        #[unroll]
        for j in 0..8u32 { xs[((8u32 * lq + j) * 65u32 + xr) as usize] = x[(xb + j) as usize]; }
        sync_cube();
        #[unroll]
        for k in 0..32u32 {
            #[unroll]
            for i in 0..8u32 { wv[i as usize] = ws[(k * 129u32 + ty * 8u32 + i) as usize]; }
            #[unroll]
            for i in 0..4u32 { xv[i as usize] = xs[(k * 65u32 + tx * 4u32 + i) as usize]; }
            #[unroll]
            for i in 0..8u32 {
                #[unroll]
                for j in 0..4u32 { acc[(i * 4u32 + j) as usize] += wv[i as usize] * xv[j as usize]; }
            }
        }
        k0 += 32u32;
    }
    let pb = CUBE_POS_Z * pst;
    #[unroll]
    for i in 0..8u32 {
        let r = r0 + ty * 8u32 + i;
        #[unroll]
        for j in 0..4u32 {
            let c = t0 + tx * 4u32 + j;
            if r < rows && c < nt { out[(pb + c * ost + ooff + r) as usize] = acc[(i * 4u32 + j) as usize]; }
        }
    }
}
/// k_gemm_pf on tensor cores: tile 64 rows x 64 tokens per cube (4 warps, each 32 tokens x 32 rows = 2x2 16x16x16 f16
/// fragments with f32 accumulators). Each 32-column step dequantizes the tile's weights (Q8_0 / native IQ4_XS) and the
/// tokens' activations to f16 in shared memory; same output/split-K interface as k_gemm_pf. Tensor-core prompt GEMM as in
/// Strata's prefill (MIT, github.com/Niko1221/Strata); pyre's own kernel.
#[cube(launch_unchecked)]
fn k_gemm_tc(x: &[f32], w: &[u32], s: &[f32], kv: &[f32], out: &mut [f32], rows: u32, cols: u32, nt: u32, kch: u32, ooff: u32, ost: u32, pst: u32, xst: u32, #[comptime] nat: bool) {
    let mut ws = Shared::<[half::f16]>::new_aligned_slice(2048usize, 32usize);
    let mut xs = Shared::<[half::f16]>::new_aligned_slice(2048usize, 32usize);
    let mut os = Shared::<[f32]>::new_aligned_slice(4096usize, 32usize);
    let mut tab = Shared::<[f32]>::new_slice(16usize);
    let tid = UNIT_POS;
    if nat { if tid < 16u32 { tab[tid as usize] = kv[tid as usize]; } }
    let r0 = CUBE_POS_X * 64u32; let t0 = CUBE_POS_Y * 64u32;
    let wp = tid / 32u32; let wr = (wp % 2u32) as usize; let wt = (wp / 2u32) as usize;
    let c00 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c01 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c10 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c11 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut a0 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let mut a1 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let mut b0 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let mut b1 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let lr = tid / 2u32; let lh = tid % 2u32;
    let mut grow = r0 + lr; if grow >= rows { grow = rows - 1u32; }
    let mut tk = t0 + lr; if tk >= nt { tk = nt - 1u32; }
    let kb = CUBE_POS_Z * kch; let mut ke = kb + kch; if ke > cols { ke = cols; }
    let mut k0 = kb;
    while k0 < ke {
        sync_cube();
        if nat {
            let blk = k0 / 256u32; let ib = (k0 / 32u32) % 8u32;
            let b = grow * (cols / 256u32) * 35u32 + blk * 35u32;
            let d = f32::reinterpret(w[b as usize]); let sh = w[(b + 1u32) as usize]; let sl = w[(b + 2u32) as usize];
            let ls = ((sl >> (4u32 * ib)) & 15u32) | (((sh >> (2u32 * ib)) & 3u32) << 4u32);
            let dl = d * (f32::cast_from(ls) - 32.0f32);
            #[unroll]
            for wq in 0..4u32 {
                let q = w[(b + 3u32 + 4u32 * ib + wq) as usize];
                #[unroll]
                for j in 0..4u32 {
                    ws[(lr * 32u32 + 16u32 * lh + 4u32 * wq + j) as usize] = half::f16::cast_from(dl * tab[((q >> (8u32 * j + 4u32 * lh)) & 15u32) as usize]);
                }
            }
        } else {
            let sc = s[(grow * (cols / 32u32) + k0 / 32u32) as usize];
            let wb = grow * (cols / 4u32) + k0 / 4u32 + 4u32 * lh;
            #[unroll]
            for h in 0..4u32 {
                let wd = w[(wb + h) as usize];
                #[unroll]
                for j in 0..4u32 { ws[(lr * 32u32 + 16u32 * lh + 4u32 * h + j) as usize] = half::f16::cast_from(sc * dec(wd, j)); }
            }
        }
        let xb = tk * xst + k0 + 16u32 * lh;
        #[unroll]
        for j in 0..16u32 { xs[(lr * 32u32 + 16u32 * lh + j) as usize] = half::f16::cast_from(x[(xb + j) as usize]); }
        sync_cube();
        #[unroll]
        for kk in 0..2usize {
            let xa = wt * 1024usize + kk * 16usize; let wa = wr * 1024usize + kk * 16usize;
            cmma::load(&mut a0, &xs[xa..xa + 496usize], 32u32);
            cmma::load(&mut a1, &xs[xa + 512usize..xa + 1008usize], 32u32);
            cmma::load(&mut b0, &ws[wa..wa + 496usize], 32u32);
            cmma::load(&mut b1, &ws[wa + 512usize..wa + 1008usize], 32u32);
            cmma::execute(&a0, &b0, &c00, &c00);
            cmma::execute(&a0, &b1, &c01, &c01);
            cmma::execute(&a1, &b0, &c10, &c10);
            cmma::execute(&a1, &b1, &c11, &c11);
        }
        k0 += 32u32;
    }
    let ob = wt * 2048usize + wr * 32usize;
    cmma::store(&mut os.slice_mut(ob, ob + 1008usize), &c00, 64u32, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut os.slice_mut(ob + 16usize, ob + 1024usize), &c01, 64u32, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut os.slice_mut(ob + 1024usize, ob + 2032usize), &c10, 64u32, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut os.slice_mut(ob + 1040usize, ob + 2048usize), &c11, 64u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    let pb = CUBE_POS_Z * pst;
    let mut i = tid;
    while i < 4096u32 { let c = t0 + i / 64u32; let r = r0 + i % 64u32;
        if r < rows && c < nt { out[(pb + c * ost + ooff + r) as usize] = os[i as usize]; }
        i += 128u32; }
}
/// split-K reduction of k_gemm_pf: out[i] = sum_z part[z * n + i] (z in order)
#[cube(launch_unchecked)]
fn k_gemm_sum(part: &[f32], out: &mut [f32], rows: u32, n: u32, ks: u32, ooff: u32, ost: u32) {
    // part planes [z][token][row] (rows per token) -> out[token * ost + ooff + row]
    let i = ABSOLUTE_POS_X;
    if i < n { let mut a = 0.0f32; let mut z = 0u32; while z < ks { a += part[(z * n + i) as usize]; z += 1u32; } let c = i / rows; let r = i % rows; out[(c * ost + ooff + r) as usize] = a; }
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

/// k_gemv4r for few rows (rows < 1024: ~2 warps per SM, latency-bound on one dependent 128-word step per
/// iteration) with the next step's weight words, scales and activations loaded before the current step's math
/// (software prefetch, two register sets). Per (row, column) the same products, the same dotx / `+= dotx * s`
/// sequence in ascending word order and the same plane_sum as k_gemv4r: identical results. Needs words % 128 == 0
/// (no k_gemv4r tail loop). PYRE_QN_GEMV4RP=0 reverts to k_gemv4r.
#[cube(launch_unchecked)]
fn k_gemv4rp(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, #[comptime] nt: u32) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 4u32;
    if row0 < rows {
        let words = cols / 4; let nb = cols / 32;
        let mut wb = Array::<u32>::new(4usize); let mut sb = Array::<u32>::new(4usize);
        #[unroll]
        for r in 0..4u32 { let mut rr = row0 + r; if rr >= rows { rr = rows - 1u32; } wb[r as usize] = rr * words; sb[r as usize] = rr * nb; }
        let mut acc = Array::<f32>::new(comptime!((4 * nt) as usize));
        #[unroll]
        for i in 0..comptime!(4 * nt) { acc[i as usize] = 0.0f32; }
        let mut cw = Array::<u32>::new(16usize); let mut cs = Array::<f32>::new(16usize); let mut cx = Array::<f32>::new(comptime!((16 * nt) as usize));
        let mut nw = Array::<u32>::new(16usize); let mut ns = Array::<f32>::new(16usize); let mut nx = Array::<f32>::new(comptime!((16 * nt) as usize));
        let mut wi = UNIT_POS_X;
        {
            let k = wi * 4;
            #[unroll]
            for r in 0..4u32 {
                #[unroll]
                for m in 0..4u32 { cw[(r * 4 + m) as usize] = w[(wb[r as usize] + wi + m * 32u32) as usize]; cs[(r * 4 + m) as usize] = s[(sb[r as usize] + k / 32 + m * 4u32) as usize]; }
            }
            #[unroll]
            for c in 0..nt {
                #[unroll]
                for q in 0..16u32 { cx[(c * 16 + q) as usize] = x[(xoff + c * xst + k + (q / 4) * 128u32 + q % 4) as usize]; }
            }
        }
        while wi < words {
            let wn = wi + 128u32;
            if wn < words {
                let k = wn * 4;
                #[unroll]
                for r in 0..4u32 {
                    #[unroll]
                    for m in 0..4u32 { nw[(r * 4 + m) as usize] = w[(wb[r as usize] + wn + m * 32u32) as usize]; ns[(r * 4 + m) as usize] = s[(sb[r as usize] + k / 32 + m * 4u32) as usize]; }
                }
                #[unroll]
                for c in 0..nt {
                    #[unroll]
                    for q in 0..16u32 { nx[(c * 16 + q) as usize] = x[(xoff + c * xst + k + (q / 4) * 128u32 + q % 4) as usize]; }
                }
            }
            #[unroll]
            for r in 0..4u32 {
                #[unroll]
                for m in 0..4u32 {
                    let wd = cw[(r * 4 + m) as usize];
                    let v0 = dec(wd, 0u32); let v1 = dec(wd, 1u32); let v2 = dec(wd, 2u32); let v3 = dec(wd, 3u32);
                    let sc = cs[(r * 4 + m) as usize];
                    #[unroll]
                    for c in 0..nt {
                        let xb = c * 16 + m * 4;
                        acc[(r * nt + c) as usize] += dotx(v0, v1, v2, v3, cx[xb as usize], cx[(xb + 1) as usize], cx[(xb + 2) as usize], cx[(xb + 3) as usize]) * sc;
                    }
                }
            }
            #[unroll]
            for i in 0..16u32 { cw[i as usize] = nw[i as usize]; cs[i as usize] = ns[i as usize]; }
            #[unroll]
            for i in 0..comptime!(16 * nt) { cx[i as usize] = nx[i as usize]; }
            wi = wn;
        }
        #[unroll]
        for r in 0..4u32 {
            #[unroll]
            for c in 0..nt {
                let tv = plane_sum(acc[(r * nt + c) as usize]);
                if UNIT_POS_X == 0 && row0 + r < rows { out[(ooff + c * ost + row0 + r) as usize] = tv; }
            }
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

/// k_gemv4r for 5-8 activation columns in one launch (PYRE_QN_GEMV8=1): weights read once; every column's loads,
/// products and per-lane / plane_sum order are k_gemv4r's (bit-identical per column).
#[cube(launch_unchecked)]
fn k_gemv8r(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32, #[comptime] c1: bool, #[comptime] c2: bool, #[comptime] c3: bool, #[comptime] c4: bool, #[comptime] c5: bool, #[comptime] c6: bool, #[comptime] c7: bool) {
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
        let mut cc3 = 3u32; if cc3 >= nt { cc3 = nt - 1u32; } let xb3 = xoff + cc3 * xst; let mut cc4 = 4u32; if cc4 >= nt { cc4 = nt - 1u32; } let xb4 = xoff + cc4 * xst; let mut cc5 = 5u32; if cc5 >= nt { cc5 = nt - 1u32; } let xb5 = xoff + cc5 * xst; let mut cc6 = 6u32; if cc6 >= nt { cc6 = nt - 1u32; } let xb6 = xoff + cc6 * xst; let mut cc7 = 7u32; if cc7 >= nt { cc7 = nt - 1u32; } let xb7 = xoff + cc7 * xst;
        let mut a0_0 = 0.0f32; let mut a0_1 = 0.0f32; let mut a0_2 = 0.0f32; let mut a0_3 = 0.0f32; let mut a0_4 = 0.0f32; let mut a0_5 = 0.0f32; let mut a0_6 = 0.0f32; let mut a0_7 = 0.0f32;
        let mut a1_0 = 0.0f32; let mut a1_1 = 0.0f32; let mut a1_2 = 0.0f32; let mut a1_3 = 0.0f32; let mut a1_4 = 0.0f32; let mut a1_5 = 0.0f32; let mut a1_6 = 0.0f32; let mut a1_7 = 0.0f32;
        let mut a2_0 = 0.0f32; let mut a2_1 = 0.0f32; let mut a2_2 = 0.0f32; let mut a2_3 = 0.0f32; let mut a2_4 = 0.0f32; let mut a2_5 = 0.0f32; let mut a2_6 = 0.0f32; let mut a2_7 = 0.0f32;
        let mut a3_0 = 0.0f32; let mut a3_1 = 0.0f32; let mut a3_2 = 0.0f32; let mut a3_3 = 0.0f32; let mut a3_4 = 0.0f32; let mut a3_5 = 0.0f32; let mut a3_6 = 0.0f32; let mut a3_7 = 0.0f32;
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
            if c4 {
                let x4_0_0 = x[(xb4 + k + 0u32) as usize]; let x4_0_1 = x[(xb4 + k + 1u32) as usize]; let x4_0_2 = x[(xb4 + k + 2u32) as usize]; let x4_0_3 = x[(xb4 + k + 3u32) as usize];
                let x4_1_0 = x[(xb4 + k + 128u32) as usize]; let x4_1_1 = x[(xb4 + k + 129u32) as usize]; let x4_1_2 = x[(xb4 + k + 130u32) as usize]; let x4_1_3 = x[(xb4 + k + 131u32) as usize];
                let x4_2_0 = x[(xb4 + k + 256u32) as usize]; let x4_2_1 = x[(xb4 + k + 257u32) as usize]; let x4_2_2 = x[(xb4 + k + 258u32) as usize]; let x4_2_3 = x[(xb4 + k + 259u32) as usize];
                let x4_3_0 = x[(xb4 + k + 384u32) as usize]; let x4_3_1 = x[(xb4 + k + 385u32) as usize]; let x4_3_2 = x[(xb4 + k + 386u32) as usize]; let x4_3_3 = x[(xb4 + k + 387u32) as usize];
                a0_4 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s0_0; a0_4 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x4_1_0, x4_1_1, x4_1_2, x4_1_3) * s0_1; a0_4 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x4_2_0, x4_2_1, x4_2_2, x4_2_3) * s0_2; a0_4 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x4_3_0, x4_3_1, x4_3_2, x4_3_3) * s0_3;
                a1_4 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s1_0; a1_4 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x4_1_0, x4_1_1, x4_1_2, x4_1_3) * s1_1; a1_4 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x4_2_0, x4_2_1, x4_2_2, x4_2_3) * s1_2; a1_4 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x4_3_0, x4_3_1, x4_3_2, x4_3_3) * s1_3;
                a2_4 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s2_0; a2_4 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x4_1_0, x4_1_1, x4_1_2, x4_1_3) * s2_1; a2_4 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x4_2_0, x4_2_1, x4_2_2, x4_2_3) * s2_2; a2_4 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x4_3_0, x4_3_1, x4_3_2, x4_3_3) * s2_3;
                a3_4 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s3_0; a3_4 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x4_1_0, x4_1_1, x4_1_2, x4_1_3) * s3_1; a3_4 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x4_2_0, x4_2_1, x4_2_2, x4_2_3) * s3_2; a3_4 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x4_3_0, x4_3_1, x4_3_2, x4_3_3) * s3_3;
            }
            if c5 {
                let x5_0_0 = x[(xb5 + k + 0u32) as usize]; let x5_0_1 = x[(xb5 + k + 1u32) as usize]; let x5_0_2 = x[(xb5 + k + 2u32) as usize]; let x5_0_3 = x[(xb5 + k + 3u32) as usize];
                let x5_1_0 = x[(xb5 + k + 128u32) as usize]; let x5_1_1 = x[(xb5 + k + 129u32) as usize]; let x5_1_2 = x[(xb5 + k + 130u32) as usize]; let x5_1_3 = x[(xb5 + k + 131u32) as usize];
                let x5_2_0 = x[(xb5 + k + 256u32) as usize]; let x5_2_1 = x[(xb5 + k + 257u32) as usize]; let x5_2_2 = x[(xb5 + k + 258u32) as usize]; let x5_2_3 = x[(xb5 + k + 259u32) as usize];
                let x5_3_0 = x[(xb5 + k + 384u32) as usize]; let x5_3_1 = x[(xb5 + k + 385u32) as usize]; let x5_3_2 = x[(xb5 + k + 386u32) as usize]; let x5_3_3 = x[(xb5 + k + 387u32) as usize];
                a0_5 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s0_0; a0_5 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x5_1_0, x5_1_1, x5_1_2, x5_1_3) * s0_1; a0_5 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x5_2_0, x5_2_1, x5_2_2, x5_2_3) * s0_2; a0_5 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x5_3_0, x5_3_1, x5_3_2, x5_3_3) * s0_3;
                a1_5 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s1_0; a1_5 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x5_1_0, x5_1_1, x5_1_2, x5_1_3) * s1_1; a1_5 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x5_2_0, x5_2_1, x5_2_2, x5_2_3) * s1_2; a1_5 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x5_3_0, x5_3_1, x5_3_2, x5_3_3) * s1_3;
                a2_5 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s2_0; a2_5 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x5_1_0, x5_1_1, x5_1_2, x5_1_3) * s2_1; a2_5 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x5_2_0, x5_2_1, x5_2_2, x5_2_3) * s2_2; a2_5 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x5_3_0, x5_3_1, x5_3_2, x5_3_3) * s2_3;
                a3_5 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s3_0; a3_5 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x5_1_0, x5_1_1, x5_1_2, x5_1_3) * s3_1; a3_5 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x5_2_0, x5_2_1, x5_2_2, x5_2_3) * s3_2; a3_5 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x5_3_0, x5_3_1, x5_3_2, x5_3_3) * s3_3;
            }
            if c6 {
                let x6_0_0 = x[(xb6 + k + 0u32) as usize]; let x6_0_1 = x[(xb6 + k + 1u32) as usize]; let x6_0_2 = x[(xb6 + k + 2u32) as usize]; let x6_0_3 = x[(xb6 + k + 3u32) as usize];
                let x6_1_0 = x[(xb6 + k + 128u32) as usize]; let x6_1_1 = x[(xb6 + k + 129u32) as usize]; let x6_1_2 = x[(xb6 + k + 130u32) as usize]; let x6_1_3 = x[(xb6 + k + 131u32) as usize];
                let x6_2_0 = x[(xb6 + k + 256u32) as usize]; let x6_2_1 = x[(xb6 + k + 257u32) as usize]; let x6_2_2 = x[(xb6 + k + 258u32) as usize]; let x6_2_3 = x[(xb6 + k + 259u32) as usize];
                let x6_3_0 = x[(xb6 + k + 384u32) as usize]; let x6_3_1 = x[(xb6 + k + 385u32) as usize]; let x6_3_2 = x[(xb6 + k + 386u32) as usize]; let x6_3_3 = x[(xb6 + k + 387u32) as usize];
                a0_6 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s0_0; a0_6 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x6_1_0, x6_1_1, x6_1_2, x6_1_3) * s0_1; a0_6 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x6_2_0, x6_2_1, x6_2_2, x6_2_3) * s0_2; a0_6 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x6_3_0, x6_3_1, x6_3_2, x6_3_3) * s0_3;
                a1_6 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s1_0; a1_6 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x6_1_0, x6_1_1, x6_1_2, x6_1_3) * s1_1; a1_6 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x6_2_0, x6_2_1, x6_2_2, x6_2_3) * s1_2; a1_6 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x6_3_0, x6_3_1, x6_3_2, x6_3_3) * s1_3;
                a2_6 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s2_0; a2_6 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x6_1_0, x6_1_1, x6_1_2, x6_1_3) * s2_1; a2_6 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x6_2_0, x6_2_1, x6_2_2, x6_2_3) * s2_2; a2_6 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x6_3_0, x6_3_1, x6_3_2, x6_3_3) * s2_3;
                a3_6 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s3_0; a3_6 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x6_1_0, x6_1_1, x6_1_2, x6_1_3) * s3_1; a3_6 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x6_2_0, x6_2_1, x6_2_2, x6_2_3) * s3_2; a3_6 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x6_3_0, x6_3_1, x6_3_2, x6_3_3) * s3_3;
            }
            if c7 {
                let x7_0_0 = x[(xb7 + k + 0u32) as usize]; let x7_0_1 = x[(xb7 + k + 1u32) as usize]; let x7_0_2 = x[(xb7 + k + 2u32) as usize]; let x7_0_3 = x[(xb7 + k + 3u32) as usize];
                let x7_1_0 = x[(xb7 + k + 128u32) as usize]; let x7_1_1 = x[(xb7 + k + 129u32) as usize]; let x7_1_2 = x[(xb7 + k + 130u32) as usize]; let x7_1_3 = x[(xb7 + k + 131u32) as usize];
                let x7_2_0 = x[(xb7 + k + 256u32) as usize]; let x7_2_1 = x[(xb7 + k + 257u32) as usize]; let x7_2_2 = x[(xb7 + k + 258u32) as usize]; let x7_2_3 = x[(xb7 + k + 259u32) as usize];
                let x7_3_0 = x[(xb7 + k + 384u32) as usize]; let x7_3_1 = x[(xb7 + k + 385u32) as usize]; let x7_3_2 = x[(xb7 + k + 386u32) as usize]; let x7_3_3 = x[(xb7 + k + 387u32) as usize];
                a0_7 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s0_0; a0_7 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x7_1_0, x7_1_1, x7_1_2, x7_1_3) * s0_1; a0_7 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x7_2_0, x7_2_1, x7_2_2, x7_2_3) * s0_2; a0_7 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x7_3_0, x7_3_1, x7_3_2, x7_3_3) * s0_3;
                a1_7 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s1_0; a1_7 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x7_1_0, x7_1_1, x7_1_2, x7_1_3) * s1_1; a1_7 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x7_2_0, x7_2_1, x7_2_2, x7_2_3) * s1_2; a1_7 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x7_3_0, x7_3_1, x7_3_2, x7_3_3) * s1_3;
                a2_7 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s2_0; a2_7 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x7_1_0, x7_1_1, x7_1_2, x7_1_3) * s2_1; a2_7 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x7_2_0, x7_2_1, x7_2_2, x7_2_3) * s2_2; a2_7 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x7_3_0, x7_3_1, x7_3_2, x7_3_3) * s2_3;
                a3_7 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s3_0; a3_7 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x7_1_0, x7_1_1, x7_1_2, x7_1_3) * s3_1; a3_7 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x7_2_0, x7_2_1, x7_2_2, x7_2_3) * s3_2; a3_7 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x7_3_0, x7_3_1, x7_3_2, x7_3_3) * s3_3;
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
            if c4 {
                let x4_0_0 = x[(xb4 + k + 0u32) as usize]; let x4_0_1 = x[(xb4 + k + 1u32) as usize]; let x4_0_2 = x[(xb4 + k + 2u32) as usize]; let x4_0_3 = x[(xb4 + k + 3u32) as usize];
                a0_4 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s0_0;
                a1_4 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s1_0;
                a2_4 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s2_0;
                a3_4 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x4_0_0, x4_0_1, x4_0_2, x4_0_3) * s3_0;
            }
            if c5 {
                let x5_0_0 = x[(xb5 + k + 0u32) as usize]; let x5_0_1 = x[(xb5 + k + 1u32) as usize]; let x5_0_2 = x[(xb5 + k + 2u32) as usize]; let x5_0_3 = x[(xb5 + k + 3u32) as usize];
                a0_5 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s0_0;
                a1_5 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s1_0;
                a2_5 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s2_0;
                a3_5 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x5_0_0, x5_0_1, x5_0_2, x5_0_3) * s3_0;
            }
            if c6 {
                let x6_0_0 = x[(xb6 + k + 0u32) as usize]; let x6_0_1 = x[(xb6 + k + 1u32) as usize]; let x6_0_2 = x[(xb6 + k + 2u32) as usize]; let x6_0_3 = x[(xb6 + k + 3u32) as usize];
                a0_6 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s0_0;
                a1_6 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s1_0;
                a2_6 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s2_0;
                a3_6 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x6_0_0, x6_0_1, x6_0_2, x6_0_3) * s3_0;
            }
            if c7 {
                let x7_0_0 = x[(xb7 + k + 0u32) as usize]; let x7_0_1 = x[(xb7 + k + 1u32) as usize]; let x7_0_2 = x[(xb7 + k + 2u32) as usize]; let x7_0_3 = x[(xb7 + k + 3u32) as usize];
                a0_7 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s0_0;
                a1_7 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s1_0;
                a2_7 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s2_0;
                a3_7 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x7_0_0, x7_0_1, x7_0_2, x7_0_3) * s3_0;
            }
            wi += 32;
        }
        let t0_0 = plane_sum(a0_0); let t0_1 = plane_sum(a0_1); let t0_2 = plane_sum(a0_2); let t0_3 = plane_sum(a0_3); let t0_4 = plane_sum(a0_4); let t0_5 = plane_sum(a0_5); let t0_6 = plane_sum(a0_6); let t0_7 = plane_sum(a0_7);
        let t1_0 = plane_sum(a1_0); let t1_1 = plane_sum(a1_1); let t1_2 = plane_sum(a1_2); let t1_3 = plane_sum(a1_3); let t1_4 = plane_sum(a1_4); let t1_5 = plane_sum(a1_5); let t1_6 = plane_sum(a1_6); let t1_7 = plane_sum(a1_7);
        let t2_0 = plane_sum(a2_0); let t2_1 = plane_sum(a2_1); let t2_2 = plane_sum(a2_2); let t2_3 = plane_sum(a2_3); let t2_4 = plane_sum(a2_4); let t2_5 = plane_sum(a2_5); let t2_6 = plane_sum(a2_6); let t2_7 = plane_sum(a2_7);
        let t3_0 = plane_sum(a3_0); let t3_1 = plane_sum(a3_1); let t3_2 = plane_sum(a3_2); let t3_3 = plane_sum(a3_3); let t3_4 = plane_sum(a3_4); let t3_5 = plane_sum(a3_5); let t3_6 = plane_sum(a3_6); let t3_7 = plane_sum(a3_7);
        if UNIT_POS_X == 0 {
            if row0 + 0u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 0u32) as usize] = t0_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 0u32) as usize] = t0_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 0u32) as usize] = t0_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 0u32) as usize] = t0_3; } if nt > 4u32 { out[(ooff + 4u32 * ost + row0 + 0u32) as usize] = t0_4; } if nt > 5u32 { out[(ooff + 5u32 * ost + row0 + 0u32) as usize] = t0_5; } if nt > 6u32 { out[(ooff + 6u32 * ost + row0 + 0u32) as usize] = t0_6; } if nt > 7u32 { out[(ooff + 7u32 * ost + row0 + 0u32) as usize] = t0_7; }
            }
            if row0 + 1u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 1u32) as usize] = t1_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 1u32) as usize] = t1_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 1u32) as usize] = t1_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 1u32) as usize] = t1_3; } if nt > 4u32 { out[(ooff + 4u32 * ost + row0 + 1u32) as usize] = t1_4; } if nt > 5u32 { out[(ooff + 5u32 * ost + row0 + 1u32) as usize] = t1_5; } if nt > 6u32 { out[(ooff + 6u32 * ost + row0 + 1u32) as usize] = t1_6; } if nt > 7u32 { out[(ooff + 7u32 * ost + row0 + 1u32) as usize] = t1_7; }
            }
            if row0 + 2u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 2u32) as usize] = t2_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 2u32) as usize] = t2_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 2u32) as usize] = t2_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 2u32) as usize] = t2_3; } if nt > 4u32 { out[(ooff + 4u32 * ost + row0 + 2u32) as usize] = t2_4; } if nt > 5u32 { out[(ooff + 5u32 * ost + row0 + 2u32) as usize] = t2_5; } if nt > 6u32 { out[(ooff + 6u32 * ost + row0 + 2u32) as usize] = t2_6; } if nt > 7u32 { out[(ooff + 7u32 * ost + row0 + 2u32) as usize] = t2_7; }
            }
            if row0 + 3u32 < rows {
                if nt > 0u32 { out[(ooff + 0u32 * ost + row0 + 3u32) as usize] = t3_0; }
                if nt > 1u32 { out[(ooff + 1u32 * ost + row0 + 3u32) as usize] = t3_1; }
                if nt > 2u32 { out[(ooff + 2u32 * ost + row0 + 3u32) as usize] = t3_2; }
                if nt > 3u32 { out[(ooff + 3u32 * ost + row0 + 3u32) as usize] = t3_3; } if nt > 4u32 { out[(ooff + 4u32 * ost + row0 + 3u32) as usize] = t3_4; } if nt > 5u32 { out[(ooff + 5u32 * ost + row0 + 3u32) as usize] = t3_5; } if nt > 6u32 { out[(ooff + 6u32 * ost + row0 + 3u32) as usize] = t3_6; } if nt > 7u32 { out[(ooff + 7u32 * ost + row0 + 3u32) as usize] = t3_7; }
            }
        }
    }
}

/// k_gemv4r with the activations read as 4-wide vectors (one 16-byte load instead of four); same arithmetic.
#[cube(launch_unchecked)]
fn k_gemv4s(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32, #[comptime] c1: bool, #[comptime] c2: bool, #[comptime] c3: bool) {
    let mut sx = Shared::<[f32]>::new_slice(12288usize);
    let mut li = UNIT_POS_Y * 32u32 + UNIT_POS_X; let tot = nt * cols;
    while li < tot { let cc = li / cols; sx[li as usize] = x[(xoff + cc * xst + li - cc * cols) as usize]; li += CUBE_DIM_Y * 32u32; }
    sync_cube();
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * 4u32;
    if row0 < rows {
        let words = cols / 4; let nb = cols / 32;
        let mut rr0 = row0 + 0u32; if rr0 >= rows { rr0 = rows - 1u32; } let wb0 = rr0 * words; let sb0 = rr0 * nb;
        let mut rr1 = row0 + 1u32; if rr1 >= rows { rr1 = rows - 1u32; } let wb1 = rr1 * words; let sb1 = rr1 * nb;
        let mut rr2 = row0 + 2u32; if rr2 >= rows { rr2 = rows - 1u32; } let wb2 = rr2 * words; let sb2 = rr2 * nb;
        let mut rr3 = row0 + 3u32; if rr3 >= rows { rr3 = rows - 1u32; } let wb3 = rr3 * words; let sb3 = rr3 * nb;
        let mut cc0 = 0u32; if cc0 >= nt { cc0 = nt - 1u32; } let xb0 = cc0 * cols;
        let mut cc1 = 1u32; if cc1 >= nt { cc1 = nt - 1u32; } let xb1 = cc1 * cols;
        let mut cc2 = 2u32; if cc2 >= nt { cc2 = nt - 1u32; } let xb2 = cc2 * cols;
        let mut cc3 = 3u32; if cc3 >= nt { cc3 = nt - 1u32; } let xb3 = cc3 * cols;
        let mut a0_0 = 0.0f32; let mut a0_1 = 0.0f32; let mut a0_2 = 0.0f32; let mut a0_3 = 0.0f32;
        let mut a1_0 = 0.0f32; let mut a1_1 = 0.0f32; let mut a1_2 = 0.0f32; let mut a1_3 = 0.0f32;
        let mut a2_0 = 0.0f32; let mut a2_1 = 0.0f32; let mut a2_2 = 0.0f32; let mut a2_3 = 0.0f32;
        let mut a3_0 = 0.0f32; let mut a3_1 = 0.0f32; let mut a3_2 = 0.0f32; let mut a3_3 = 0.0f32;
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let k = wi * 4;
            let x0_0_0 = sx[(xb0 + k + 0u32) as usize]; let x0_0_1 = sx[(xb0 + k + 1u32) as usize]; let x0_0_2 = sx[(xb0 + k + 2u32) as usize]; let x0_0_3 = sx[(xb0 + k + 3u32) as usize];
            let x0_1_0 = sx[(xb0 + k + 128u32) as usize]; let x0_1_1 = sx[(xb0 + k + 129u32) as usize]; let x0_1_2 = sx[(xb0 + k + 130u32) as usize]; let x0_1_3 = sx[(xb0 + k + 131u32) as usize];
            let x0_2_0 = sx[(xb0 + k + 256u32) as usize]; let x0_2_1 = sx[(xb0 + k + 257u32) as usize]; let x0_2_2 = sx[(xb0 + k + 258u32) as usize]; let x0_2_3 = sx[(xb0 + k + 259u32) as usize];
            let x0_3_0 = sx[(xb0 + k + 384u32) as usize]; let x0_3_1 = sx[(xb0 + k + 385u32) as usize]; let x0_3_2 = sx[(xb0 + k + 386u32) as usize]; let x0_3_3 = sx[(xb0 + k + 387u32) as usize];
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
                let x1_0_0 = sx[(xb1 + k + 0u32) as usize]; let x1_0_1 = sx[(xb1 + k + 1u32) as usize]; let x1_0_2 = sx[(xb1 + k + 2u32) as usize]; let x1_0_3 = sx[(xb1 + k + 3u32) as usize];
                let x1_1_0 = sx[(xb1 + k + 128u32) as usize]; let x1_1_1 = sx[(xb1 + k + 129u32) as usize]; let x1_1_2 = sx[(xb1 + k + 130u32) as usize]; let x1_1_3 = sx[(xb1 + k + 131u32) as usize];
                let x1_2_0 = sx[(xb1 + k + 256u32) as usize]; let x1_2_1 = sx[(xb1 + k + 257u32) as usize]; let x1_2_2 = sx[(xb1 + k + 258u32) as usize]; let x1_2_3 = sx[(xb1 + k + 259u32) as usize];
                let x1_3_0 = sx[(xb1 + k + 384u32) as usize]; let x1_3_1 = sx[(xb1 + k + 385u32) as usize]; let x1_3_2 = sx[(xb1 + k + 386u32) as usize]; let x1_3_3 = sx[(xb1 + k + 387u32) as usize];
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0; a0_1 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s0_1; a0_1 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s0_2; a0_1 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s0_3;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0; a1_1 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s1_1; a1_1 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s1_2; a1_1 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s1_3;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0; a2_1 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s2_1; a2_1 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s2_2; a2_1 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s2_3;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0; a3_1 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x1_1_0, x1_1_1, x1_1_2, x1_1_3) * s3_1; a3_1 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x1_2_0, x1_2_1, x1_2_2, x1_2_3) * s3_2; a3_1 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x1_3_0, x1_3_1, x1_3_2, x1_3_3) * s3_3;
            }
            if c2 {
                let x2_0_0 = sx[(xb2 + k + 0u32) as usize]; let x2_0_1 = sx[(xb2 + k + 1u32) as usize]; let x2_0_2 = sx[(xb2 + k + 2u32) as usize]; let x2_0_3 = sx[(xb2 + k + 3u32) as usize];
                let x2_1_0 = sx[(xb2 + k + 128u32) as usize]; let x2_1_1 = sx[(xb2 + k + 129u32) as usize]; let x2_1_2 = sx[(xb2 + k + 130u32) as usize]; let x2_1_3 = sx[(xb2 + k + 131u32) as usize];
                let x2_2_0 = sx[(xb2 + k + 256u32) as usize]; let x2_2_1 = sx[(xb2 + k + 257u32) as usize]; let x2_2_2 = sx[(xb2 + k + 258u32) as usize]; let x2_2_3 = sx[(xb2 + k + 259u32) as usize];
                let x2_3_0 = sx[(xb2 + k + 384u32) as usize]; let x2_3_1 = sx[(xb2 + k + 385u32) as usize]; let x2_3_2 = sx[(xb2 + k + 386u32) as usize]; let x2_3_3 = sx[(xb2 + k + 387u32) as usize];
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0; a0_2 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s0_1; a0_2 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s0_2; a0_2 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s0_3;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0; a1_2 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s1_1; a1_2 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s1_2; a1_2 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s1_3;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0; a2_2 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s2_1; a2_2 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s2_2; a2_2 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s2_3;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0; a3_2 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x2_1_0, x2_1_1, x2_1_2, x2_1_3) * s3_1; a3_2 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x2_2_0, x2_2_1, x2_2_2, x2_2_3) * s3_2; a3_2 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x2_3_0, x2_3_1, x2_3_2, x2_3_3) * s3_3;
            }
            if c3 {
                let x3_0_0 = sx[(xb3 + k + 0u32) as usize]; let x3_0_1 = sx[(xb3 + k + 1u32) as usize]; let x3_0_2 = sx[(xb3 + k + 2u32) as usize]; let x3_0_3 = sx[(xb3 + k + 3u32) as usize];
                let x3_1_0 = sx[(xb3 + k + 128u32) as usize]; let x3_1_1 = sx[(xb3 + k + 129u32) as usize]; let x3_1_2 = sx[(xb3 + k + 130u32) as usize]; let x3_1_3 = sx[(xb3 + k + 131u32) as usize];
                let x3_2_0 = sx[(xb3 + k + 256u32) as usize]; let x3_2_1 = sx[(xb3 + k + 257u32) as usize]; let x3_2_2 = sx[(xb3 + k + 258u32) as usize]; let x3_2_3 = sx[(xb3 + k + 259u32) as usize];
                let x3_3_0 = sx[(xb3 + k + 384u32) as usize]; let x3_3_1 = sx[(xb3 + k + 385u32) as usize]; let x3_3_2 = sx[(xb3 + k + 386u32) as usize]; let x3_3_3 = sx[(xb3 + k + 387u32) as usize];
                a0_3 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s0_0; a0_3 += dotx(v0_1_0, v0_1_1, v0_1_2, v0_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s0_1; a0_3 += dotx(v0_2_0, v0_2_1, v0_2_2, v0_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s0_2; a0_3 += dotx(v0_3_0, v0_3_1, v0_3_2, v0_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s0_3;
                a1_3 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s1_0; a1_3 += dotx(v1_1_0, v1_1_1, v1_1_2, v1_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s1_1; a1_3 += dotx(v1_2_0, v1_2_1, v1_2_2, v1_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s1_2; a1_3 += dotx(v1_3_0, v1_3_1, v1_3_2, v1_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s1_3;
                a2_3 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s2_0; a2_3 += dotx(v2_1_0, v2_1_1, v2_1_2, v2_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s2_1; a2_3 += dotx(v2_2_0, v2_2_1, v2_2_2, v2_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s2_2; a2_3 += dotx(v2_3_0, v2_3_1, v2_3_2, v2_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s2_3;
                a3_3 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x3_0_0, x3_0_1, x3_0_2, x3_0_3) * s3_0; a3_3 += dotx(v3_1_0, v3_1_1, v3_1_2, v3_1_3, x3_1_0, x3_1_1, x3_1_2, x3_1_3) * s3_1; a3_3 += dotx(v3_2_0, v3_2_1, v3_2_2, v3_2_3, x3_2_0, x3_2_1, x3_2_2, x3_2_3) * s3_2; a3_3 += dotx(v3_3_0, v3_3_1, v3_3_2, v3_3_3, x3_3_0, x3_3_1, x3_3_2, x3_3_3) * s3_3;
            }
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let x0_0_0 = sx[(xb0 + k + 0u32) as usize]; let x0_0_1 = sx[(xb0 + k + 1u32) as usize]; let x0_0_2 = sx[(xb0 + k + 2u32) as usize]; let x0_0_3 = sx[(xb0 + k + 3u32) as usize];
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
                let x1_0_0 = sx[(xb1 + k + 0u32) as usize]; let x1_0_1 = sx[(xb1 + k + 1u32) as usize]; let x1_0_2 = sx[(xb1 + k + 2u32) as usize]; let x1_0_3 = sx[(xb1 + k + 3u32) as usize];
                a0_1 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s0_0;
                a1_1 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s1_0;
                a2_1 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s2_0;
                a3_1 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x1_0_0, x1_0_1, x1_0_2, x1_0_3) * s3_0;
            }
            if c2 {
                let x2_0_0 = sx[(xb2 + k + 0u32) as usize]; let x2_0_1 = sx[(xb2 + k + 1u32) as usize]; let x2_0_2 = sx[(xb2 + k + 2u32) as usize]; let x2_0_3 = sx[(xb2 + k + 3u32) as usize];
                a0_2 += dotx(v0_0_0, v0_0_1, v0_0_2, v0_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s0_0;
                a1_2 += dotx(v1_0_0, v1_0_1, v1_0_2, v1_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s1_0;
                a2_2 += dotx(v2_0_0, v2_0_1, v2_0_2, v2_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s2_0;
                a3_2 += dotx(v3_0_0, v3_0_1, v3_0_2, v3_0_3, x2_0_0, x2_0_1, x2_0_2, x2_0_3) * s3_0;
            }
            if c3 {
                let x3_0_0 = sx[(xb3 + k + 0u32) as usize]; let x3_0_1 = sx[(xb3 + k + 1u32) as usize]; let x3_0_2 = sx[(xb3 + k + 2u32) as usize]; let x3_0_3 = sx[(xb3 + k + 3u32) as usize];
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
/// k_q8pack_m as ggml quantize_q8_1 (PYRE_QN_NATX): q = roundf(x / d) (IEEE divide, halves away from zero), 0 for an all-zero block
#[cube(launch_unchecked)]
fn k_q8pack_mx(src: &[f32], q: &mut [u32], sc: &mut [f32], soff: u32, sst: u32, nb: u32) {
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
                let qv = src[(base + w * 4 + j) as usize] / d; let fa = f32::floor(f32::abs(qv)); let mut r = fa; if f32::abs(qv) - fa >= 0.5f32 { r = fa + 1.0f32; } if qv < 0.0f32 { r = 0.0f32 - r; } if d == 0.0f32 { r = 0.0f32; }
                let v = i32::cast_from(r);
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

/// k_gemv_iq4d_r with the loads of U consecutive 256-blocks (all 4 rows' headers + quant words, the x words) issued
/// before any of their math, so U x more loads are in flight per warp (the r kernel ran at ~48% of DRAM bandwidth,
/// latency-bound on one dependent block per iteration). Per (row, column) the products and the accumulation order
/// (ascending blocks, then plane_sum) are k_gemv_iq4d_r's: bit-identical output.
#[cube(launch_unchecked)]
fn k_gemv_iq4d_r2(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32, #[comptime] u: u32, #[comptime] pl: bool) {
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
        let mut xl = Array::<i32>::new(comptime!((u * 4) as usize));
        let mut xh = Array::<i32>::new(comptime!((u * 4) as usize));
        let mut xv = Array::<f32>::new(comptime!((u * 4) as usize));
        let mut wd = Array::<f32>::new(comptime!((u * 4) as usize));
        let mut wh = Array::<u32>::new(comptime!((u * 4) as usize));
        let mut wl = Array::<u32>::new(comptime!((u * 4) as usize));
        let mut wq = Array::<u32>::new(comptime!((u * 4) as usize));
        let mut blk = 0u32;
        while blk < nb {
            #[unroll]
            for j in 0..u {
                let bb = blk + j;
                if bb < nb {
                    let xw = bb * 64 + ib * 8 + k;
                    let xi = bb * 8 + ib;
                    #[unroll]
                    for c in 0..4u32 {
                        if c < nt {
                            xl[(j * 4 + c) as usize] = i32::cast_from(xq[(c * cq + xw) as usize]);
                            xh[(j * 4 + c) as usize] = i32::cast_from(xq[(c * cq + xw + 4) as usize]);
                            xv[(j * 4 + c) as usize] = xs[(c * cs + xi) as usize];
                        }
                    }
                    #[unroll]
                    for r in 0..4u32 {
                        let rr = row0 + r;
                        if rr < rows {
                            let b0 = rr * rw + bb * 35;
                            wd[(j * 4 + r) as usize] = f32::reinterpret(w[b0 as usize]);
                            wh[(j * 4 + r) as usize] = w[(b0 + 1) as usize];
                            wl[(j * 4 + r) as usize] = w[(b0 + 2) as usize];
                            wq[(j * 4 + r) as usize] = w[(b0 + 3 + UNIT_POS_X) as usize];
                        }
                    }
                }
            }
            #[unroll]
            for j in 0..u {
                if blk + j < nb {
                    #[unroll]
                    for r in 0..4u32 {
                        if row0 + r < rows {
                            let q = wq[(j * 4 + r) as usize];
                            let ls = i32::cast_from(((wl[(j * 4 + r) as usize] >> (4 * ib)) & 15u32) | (((wh[(j * 4 + r) as usize] >> (2 * ib)) & 3u32) << 4)) - 32i32;
                            let mut lo = 0i32;
                            let mut hi = 0i32;
                            if comptime!(pl) {
                                // register-only levels (iq4_lookup, Strata/llama.cpp byte permutes): the same int8 values as the table
                                let (lu, hu) = iq4_lookup(q);
                                lo = i32::cast_from(lu);
                                hi = i32::cast_from(hu);
                            } else {
                                let u0 = tb[(q & 255u32) as usize];
                                let u1 = tb[((q >> 8) & 255u32) as usize];
                                let u2 = tb[((q >> 16) & 255u32) as usize];
                                let u3 = tb[(q >> 24) as usize];
                                lo = i32::cast_from((u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24));
                                hi = i32::cast_from(((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24));
                            }
                            let dl = wd[(j * 4 + r) as usize] * f32::cast_from(ls);
                            #[unroll]
                            for c in 0..4u32 {
                                if c < nt {
                                    let p = dp4a(hi, xh[(j * 4 + c) as usize], dp4a(lo, xl[(j * 4 + c) as usize], 0i32));
                                    acc[(r * 4 + c) as usize] += dl * xv[(j * 4 + c) as usize] * f32::cast_from(p);
                                }
                            }
                        }
                    }
                }
            }
            blk += u;
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

/// k_gemv_iq4d_r2 software-pipelined after Strata's native_mmvq_il_pf_kernel (MIT, src/kernels/cuda/native_mmvq.cu:1998-2240,
/// "P1" hoisted group loads + DB ping-pong): the next block's loads (all `rr` rows' header + quant words, the x words) are issued
/// with no branch (address clamped to the last block / last row, like Strata's tail clamp) before the current block's math, so a
/// warp always has one block per row in flight while it computes the previous one. `rr` rows a warp (1, 2, 4). Per (row, column)
/// the lane mapping (lane = sub-block*4 + word), the products, the ascending-block accumulation and the plane_sum are
/// k_gemv_iq4d_r2's: bit-identical output. PYRE_QN_NATR3=rr (default 4).
#[cube(launch_unchecked)]
fn k_gemv_iq4d_r3(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32, #[comptime] rr: u32, #[comptime] pl: bool) {
    let row0 = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) * rr;
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
        let mut acc = Array::<f32>::new(comptime!((rr * 4) as usize));
        #[unroll]
        for i in 0..rr * 4 { acc[i as usize] = 0.0f32; }
        let mut rb = Array::<u32>::new(comptime!(rr as usize));
        #[unroll]
        for r in 0..rr { let mut rx = row0 + r; if rx >= rows { rx = rows - 1; } rb[r as usize] = rx * rw; }
        let mut xl = Array::<i32>::new(4usize);
        let mut xh = Array::<i32>::new(4usize);
        let mut xv = Array::<f32>::new(4usize);
        let mut wd = Array::<f32>::new(comptime!(rr as usize));
        let mut wh = Array::<u32>::new(comptime!(rr as usize));
        let mut wl = Array::<u32>::new(comptime!(rr as usize));
        let mut wq = Array::<u32>::new(comptime!(rr as usize));
        let mut nl = Array::<i32>::new(4usize);
        let mut nh = Array::<i32>::new(4usize);
        let mut nv = Array::<f32>::new(4usize);
        let mut nd = Array::<f32>::new(comptime!(rr as usize));
        let mut nhh = Array::<u32>::new(comptime!(rr as usize));
        let mut nll = Array::<u32>::new(comptime!(rr as usize));
        let mut nq = Array::<u32>::new(comptime!(rr as usize));
        // block 0 (columns >= nt read column 0's words: always in bounds, never used)
        #[unroll]
        for c in 0..4u32 {
            let mut cc = 0u32; if c < nt { cc = c; }
            xl[c as usize] = i32::cast_from(xq[(cc * cq + ib * 8 + k) as usize]);
            xh[c as usize] = i32::cast_from(xq[(cc * cq + ib * 8 + k + 4) as usize]);
            xv[c as usize] = xs[(cc * cs + ib) as usize];
        }
        #[unroll]
        for r in 0..rr {
            let b0 = rb[r as usize];
            wd[r as usize] = f32::reinterpret(w[b0 as usize]);
            wh[r as usize] = w[(b0 + 1) as usize];
            wl[r as usize] = w[(b0 + 2) as usize];
            wq[r as usize] = w[(b0 + 3 + UNIT_POS_X) as usize];
        }
        let mut blk = 0u32;
        while blk < nb {
            let mut bn = blk + 1; if bn >= nb { bn = nb - 1; }
            let xw = bn * 64 + ib * 8 + k;
            let xi = bn * 8 + ib;
            #[unroll]
            for c in 0..4u32 {
                let mut cc = 0u32; if c < nt { cc = c; }
                nl[c as usize] = i32::cast_from(xq[(cc * cq + xw) as usize]);
                nh[c as usize] = i32::cast_from(xq[(cc * cq + xw + 4) as usize]);
                nv[c as usize] = xs[(cc * cs + xi) as usize];
            }
            #[unroll]
            for r in 0..rr {
                let b0 = rb[r as usize] + bn * 35;
                nd[r as usize] = f32::reinterpret(w[b0 as usize]);
                nhh[r as usize] = w[(b0 + 1) as usize];
                nll[r as usize] = w[(b0 + 2) as usize];
                nq[r as usize] = w[(b0 + 3 + UNIT_POS_X) as usize];
            }
            #[unroll]
            for r in 0..rr {
                let q = wq[r as usize];
                let ls = i32::cast_from(((wl[r as usize] >> (4 * ib)) & 15u32) | (((wh[r as usize] >> (2 * ib)) & 3u32) << 4)) - 32i32;
                let mut lo = 0i32;
                let mut hi = 0i32;
                if comptime!(pl) {
                    let (lu, hu) = iq4_lookup(q);
                    lo = i32::cast_from(lu);
                    hi = i32::cast_from(hu);
                } else {
                    let u0 = tb[(q & 255u32) as usize];
                    let u1 = tb[((q >> 8) & 255u32) as usize];
                    let u2 = tb[((q >> 16) & 255u32) as usize];
                    let u3 = tb[(q >> 24) as usize];
                    lo = i32::cast_from((u0 & 255u32) | ((u1 & 255u32) << 8) | ((u2 & 255u32) << 16) | ((u3 & 255u32) << 24));
                    hi = i32::cast_from(((u0 >> 8) & 255u32) | (u1 & 65280u32) | (((u2 >> 8) & 255u32) << 16) | (((u3 >> 8) & 255u32) << 24));
                }
                let dl = wd[r as usize] * f32::cast_from(ls);
                #[unroll]
                for c in 0..4u32 {
                    if c < nt {
                        let p = dp4a(hi, xh[c as usize], dp4a(lo, xl[c as usize], 0i32));
                        acc[(r * 4 + c) as usize] += dl * xv[c as usize] * f32::cast_from(p);
                    }
                }
            }
            #[unroll]
            for c in 0..4u32 { xl[c as usize] = nl[c as usize]; xh[c as usize] = nh[c as usize]; xv[c as usize] = nv[c as usize]; }
            #[unroll]
            for r in 0..rr { wd[r as usize] = nd[r as usize]; wh[r as usize] = nhh[r as usize]; wl[r as usize] = nll[r as usize]; wq[r as usize] = nq[r as usize]; }
            blk += 1u32;
        }
        #[unroll]
        for r in 0..rr {
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
/// warp_sum as Strata / ggml: xor butterfly 16, 8, 4, 2, 1
#[cube]
fn wsum(x: f32) -> f32 { let mut v = x; v += plane_shuffle_xor(v, 16u32); v += plane_shuffle_xor(v, 8u32); v += plane_shuffle_xor(v, 4u32); v += plane_shuffle_xor(v, 2u32); v += plane_shuffle_xor(v, 1u32); v }
/// q8_1 keeps the activation scale as a half (ggml block_q8_1.ds): the scale as that half, back in f32
#[cube]
fn hround(v: f32) -> f32 { let u = u32::reinterpret(v); f32::reinterpret((u + 4095u32 + ((u >> 13) & 1u32)) & 4294959104u32) } // round-to-nearest-even to an 11-bit significand (normal half range)
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
            a0 += (d * hround(xs[xi as usize])) * f32::cast_from(s0 * ls);
            if nt > 1u32 { a1 += (d * hround(xs[(cs + xi) as usize])) * f32::cast_from(s1 * ls); }
            if nt > 2u32 { a2 += (d * hround(xs[(2 * cs + xi) as usize])) * f32::cast_from(s2 * ls); }
            if nt > 3u32 { a3 += (d * hround(xs[(3 * cs + xi) as usize])) * f32::cast_from(s3 * ls); }
            blk += 4u32;
        }
        let t0 = wsum(a0);
        let t1 = wsum(a1);
        let t2s = wsum(a2);
        let t3 = wsum(a3);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t0;
            if nt > 1u32 { out[(ooff + ost + row) as usize] = t1; }
            if nt > 2u32 { out[(ooff + 2 * ost + row) as usize] = t2s; }
            if nt > 3u32 { out[(ooff + 3 * ost + row) as usize] = t3; }
        }
    }
}

/// IQ4_XS GEMV with int8 activations (x packed by k_q8pack) and dp4a. t2[b] = IQ4_NL level of the low
/// k_gemv_iq4s_m in ggml/Strata native_iq4_xs_mmvq order for any row length (PYRE_QN_NATX): lane l keeps one partial per
/// virtual warp v (blocks 4v + l/8 + 16k), summed ((p0 + p1) + p2) + p3 then xor butterfly - the 4-warp kernel's bits on
/// one warp (Strata, MIT, native_mmvq.cu note at the ROWS kernels).
#[cube(launch_unchecked)]
fn k_gemv_iq4s_mx(xq: &[u32], xs: &[f32], w: &[u32], t2: &[u32], out: &mut [f32], rows: u32, cols: u32, ooff: u32, ost: u32, nt: u32, #[comptime] fm: bool) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nb = cols / 256;
        let rw = nb * 35;
        let ib = UNIT_POS_X % 8;
        let cq = cols / 4;
        let cs = cols / 32;
        let mut a0 = 0.0f32; let mut a0b = 0.0f32; let mut a0c = 0.0f32; let mut a0d = 0.0f32;
        let mut a1 = 0.0f32; let mut a1b = 0.0f32; let mut a1c = 0.0f32; let mut a1d = 0.0f32;
        let mut a2 = 0.0f32; let mut a2b = 0.0f32; let mut a2c = 0.0f32; let mut a2d = 0.0f32;
        let mut a3 = 0.0f32; let mut a3b = 0.0f32; let mut a3c = 0.0f32; let mut a3d = 0.0f32;
        let mut blk = UNIT_POS_X / 8; let mut m = 0u32;
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
            let vd = d * hround(xs[xi as usize]); let vs = f32::cast_from(s0 * ls); let vv = vd * vs; let vw = m % 4u32; if vw == 0u32 { if fm { a0 = fma(vd, vs, a0); } else { a0 += vv; } } else if vw == 1u32 { if fm { a0b = fma(vd, vs, a0b); } else { a0b += vv; } } else if vw == 2u32 { if fm { a0c = fma(vd, vs, a0c); } else { a0c += vv; } } else { if fm { a0d = fma(vd, vs, a0d); } else { a0d += vv; } }
            if nt > 1u32 { let vd = d * hround(xs[(cs + xi) as usize]); let vs = f32::cast_from(s1 * ls); let vv = vd * vs; let vw = m % 4u32; if vw == 0u32 { if fm { a1 = fma(vd, vs, a1); } else { a1 += vv; } } else if vw == 1u32 { if fm { a1b = fma(vd, vs, a1b); } else { a1b += vv; } } else if vw == 2u32 { if fm { a1c = fma(vd, vs, a1c); } else { a1c += vv; } } else { if fm { a1d = fma(vd, vs, a1d); } else { a1d += vv; } } }
            if nt > 2u32 { let vd = d * hround(xs[(2 * cs + xi) as usize]); let vs = f32::cast_from(s2 * ls); let vv = vd * vs; let vw = m % 4u32; if vw == 0u32 { if fm { a2 = fma(vd, vs, a2); } else { a2 += vv; } } else if vw == 1u32 { if fm { a2b = fma(vd, vs, a2b); } else { a2b += vv; } } else if vw == 2u32 { if fm { a2c = fma(vd, vs, a2c); } else { a2c += vv; } } else { if fm { a2d = fma(vd, vs, a2d); } else { a2d += vv; } } }
            if nt > 3u32 { let vd = d * hround(xs[(3 * cs + xi) as usize]); let vs = f32::cast_from(s3 * ls); let vv = vd * vs; let vw = m % 4u32; if vw == 0u32 { if fm { a3 = fma(vd, vs, a3); } else { a3 += vv; } } else if vw == 1u32 { if fm { a3b = fma(vd, vs, a3b); } else { a3b += vv; } } else if vw == 2u32 { if fm { a3c = fma(vd, vs, a3c); } else { a3c += vv; } } else { if fm { a3d = fma(vd, vs, a3d); } else { a3d += vv; } } }
            blk += 4u32; m += 1u32;
        }
        let t0 = wsum(((a0 + a0b) + a0c) + a0d);
        let t1 = wsum(((a1 + a1b) + a1c) + a1d);
        let t2s = wsum(((a2 + a2b) + a2c) + a2d);
        let t3 = wsum(((a3 + a3b) + a3c) + a3d);
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
/// 8-buffer copies of k_rms4 / k_comb_m / k_comb3_m / k_delta_m (PYRE_QN_W8 under T8): token ti's arithmetic is unchanged.
#[cube(launch_unchecked)]
fn k_rms8(x0: &[f32], x1: &[f32], x2: &[f32], x3: &[f32], x4: &[f32], x5: &[f32], x6: &[f32], x7: &[f32], norm: &[f32], xn: &mut [f32], d: u32, w: u32, eps: f32) {
    let base = CUBE_POS_X * d; let tk = CUBE_POS_Y; let ob = tk * w;
    let mut acc = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d {
        let j = (base + i) as usize;
        let mut v = x0[j]; if tk == 1u32 { v = x1[j]; } if tk == 2u32 { v = x2[j]; } if tk == 3u32 { v = x3[j]; } if tk == 4u32 { v = x4[j]; } if tk == 5u32 { v = x5[j]; } if tk == 6u32 { v = x6[j]; } if tk == 7u32 { v = x7[j]; }
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
        let mut v = x0[k]; if tk == 1u32 { v = x1[k]; } if tk == 2u32 { v = x2[k]; } if tk == 3u32 { v = x3[k]; } if tk == 4u32 { v = x4[k]; } if tk == 5u32 { v = x5[k]; } if tk == 6u32 { v = x6[k]; } if tk == 7u32 { v = x7[k]; }
        xn[(ob + base + i2) as usize] = v * r * norm[k];
        i2 += CUBE_DIM_X;
    }
}
#[cube(launch_unchecked)]
fn k_comb_m8(x0: &mut [f32], x1: &mut [f32], x2: &mut [f32], x3: &mut [f32], x4: &mut [f32], x5: &mut [f32], x6: &mut [f32], x7: &mut [f32], a: &[f32], dn: &[f32], d: u32, ds: u32, lr: u32, #[comptime] hc: u32) {
    let k = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let o = a[(ti * d + i) as usize];
        let inj = dn[(ti * ds + lr + st) as usize] / f32::cast_from(hc);
        let v = o * 2.0f32 / (1.0f32 + f32::exp(-inj));
        if ti == 0u32 { x0[k as usize] += v; } else if ti == 1u32 { x1[k as usize] += v; } else if ti == 2u32 { x2[k as usize] += v; } else if ti == 3u32 { x3[k as usize] += v; } else if ti == 4u32 { x4[k as usize] += v; } else if ti == 5u32 { x5[k as usize] += v; } else if ti == 6u32 { x6[k as usize] += v; } else { x7[k as usize] += v; }
    }
}
#[cube(launch_unchecked)]
fn k_comb3_m8(x0: &mut [f32], x1: &mut [f32], x2: &mut [f32], x3: &mut [f32], x4: &mut [f32], x5: &mut [f32], x6: &mut [f32], x7: &mut [f32], c0: &[f32], c1: &[f32], c2: &[f32], c3: &[f32], c4: &[f32], c5: &[f32], c6: &[f32], c7: &[f32], a: &[f32], b: &[f32], gb: &[f32], dn: &[f32], d: u32, gs: u32, goff: u32, ds: u32, lr: u32, cm: u32, #[comptime] hc: u32) {
    let k = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let mut o = a[(ti * d + i) as usize];
        o = o / (1.0f32 + f32::exp(-gb[(ti * gs + goff) as usize]));
        o += b[(ti * d + i) as usize];
        if ((cm >> ti) & 1u32) == 1u32 {
            if ti == 0u32 { o += c0[i as usize]; } else if ti == 1u32 { o += c1[i as usize]; } else if ti == 2u32 { o += c2[i as usize]; } else if ti == 3u32 { o += c3[i as usize]; } else if ti == 4u32 { o += c4[i as usize]; } else if ti == 5u32 { o += c5[i as usize]; } else if ti == 6u32 { o += c6[i as usize]; } else { o += c7[i as usize]; }
        }
        let inj = dn[(ti * ds + lr + st) as usize] / f32::cast_from(hc);
        let v = o * 2.0f32 / (1.0f32 + f32::exp(-inj));
        if ti == 0u32 { x0[k as usize] += v; } else if ti == 1u32 { x1[k as usize] += v; } else if ti == 2u32 { x2[k as usize] += v; } else if ti == 3u32 { x3[k as usize] += v; } else if ti == 4u32 { x4[k as usize] += v; } else if ti == 5u32 { x5[k as usize] += v; } else if ti == 6u32 { x6[k as usize] += v; } else { x7[k as usize] += v; }
    }
}
#[cube(launch_unchecked)]
fn k_delta_m8(co: &[f32], proj: &[f32], a: &[f32], dt: &[f32], norm: &[f32], s: &mut [f32], out: &mut [f32], c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], c3: &mut [f32], c4: &mut [f32], c5: &mut [f32], c6: &mut [f32], c7: &mut [f32],
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
                if t == 0u32 { c0[idx] = st[i as usize]; } else if t == 1u32 { c1[idx] = st[i as usize]; } else if t == 2u32 { c2[idx] = st[i as usize]; } else if t == 3u32 { c3[idx] = st[i as usize]; } else if t == 4u32 { c4[idx] = st[i as usize]; } else if t == 5u32 { c5[idx] = st[i as usize]; } else if t == 6u32 { c6[idx] = st[i as usize]; } else { c7[idx] = st[i as usize]; }
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

/// k_gate_m with one warp per (output i, stream st) instead of one warp per output looping over the hc streams
/// (8/hc outputs per cube, hc x more warps in flight), each lane's weight words / scales loaded before the silu
/// prologue's sync. Every (i, st, token) dot is k_gate_m's: same lane split, same per-lane order, same plane_sum;
/// the hc stream terms xn / (1 + exp(-g)) are added in st order from 0.0 by one thread, as k_gate_m's acc
/// (identical results). Needs 8 % hc == 0. PYRE_QN_GATEM2=0 reverts to k_gate_m.
#[cube(launch_unchecked)]
fn k_gate_m2(xn: &[f32], dn: &[f32], w: &[u32], s: &[f32], out: &mut [f32], d: u32, ooff: u32, xs: u32, ds: u32, os: u32, #[comptime] lr: usize, #[comptime] hc: u32, #[comptime] t: u32) {
    let mut lo = Shared::<[f32]>::new_slice(comptime!(lr * t as usize));
    let mut tr = Shared::<[f32]>::new_slice(comptime!(8 * t as usize));
    let hcf = f32::cast_from(hc);
    let tid = UNIT_POS_Y * 32 + UNIT_POS_X;
    let words = comptime!(lr as u32 / 4);
    let nb = comptime!(lr as u32 / 32);
    let nk = comptime!((lr as u32 / 4).div_ceil(32));
    let i = CUBE_POS_X * comptime!(8 / hc) + UNIT_POS_Y / hc;
    let st = UNIT_POS_Y % hc;
    let mut ii = i; if ii >= d { ii = d - 1u32; }
    let row = st * d + ii;
    let mut wv = Array::<u32>::new(comptime!(nk as usize));
    let mut sv = Array::<f32>::new(comptime!(nk as usize));
    #[unroll]
    for k in 0..nk {
        let wi = UNIT_POS_X + k * 32u32;
        let mut wd = 0u32; let mut sc = 0.0f32;
        if wi < words { wd = w[(row * words + wi) as usize]; sc = s[(row * nb + wi / 8) as usize]; }
        wv[k as usize] = wd; sv[k as usize] = sc;
    }
    let mut xv = Array::<f32>::new(comptime!(t as usize));
    #[unroll]
    for tk in 0..t { xv[tk as usize] = xn[(tk * xs + row) as usize]; }
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
    let mut g = Array::<f32>::new(comptime!(t as usize));
    #[unroll]
    for tk in 0..t { g[tk as usize] = 0.0f32; }
    #[unroll]
    for k in 0..nk {
        let wi = UNIT_POS_X + k * 32u32;
        if wi < words {
            let word = wv[k as usize];
            let sc = sv[k as usize];
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
        }
    }
    #[unroll]
    for tk in 0..t {
        let gt = plane_sum(g[tk as usize]);
        if UNIT_POS_X == 0 { tr[(UNIT_POS_Y * t + tk) as usize] = xv[tk as usize] / (1.0f32 + f32::exp(-gt)); }
    }
    sync_cube();
    if UNIT_POS_X == 0 && st == 0 && i < d {
        #[unroll]
        for tk in 0..t {
            let mut acc = 0.0f32;
            #[unroll]
            for s2 in 0..hc { acc += tr[((UNIT_POS_Y + s2) * t + tk) as usize]; }
            out[(tk * os + ooff + i) as usize] = acc / hcf;
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

/// k_comb (no gate, no b/c terms) for up to 4 tokens in one launch: token ti = CUBE_POS_Y updates x_ti from
/// a[ti*d..] and dn[ti*ds..]; per element the same arithmetic as k_comb (PYRE_QN_COMBM, default on).
#[cube(launch_unchecked)]
fn k_comb_m(x0: &mut [f32], x1: &mut [f32], x2: &mut [f32], x3: &mut [f32], a: &[f32], dn: &[f32], d: u32, ds: u32, lr: u32, #[comptime] hc: u32) {
    let k = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let o = a[(ti * d + i) as usize];
        let inj = dn[(ti * ds + lr + st) as usize] / f32::cast_from(hc);
        let v = o * 2.0f32 / (1.0f32 + f32::exp(-inj));
        if ti == 0u32 { x0[k as usize] += v; } else if ti == 1u32 { x1[k as usize] += v; } else if ti == 2u32 { x2[k as usize] += v; } else { x3[k as usize] += v; }
    }
}
/// The post-MoE k_comb3 (gate on, b on, c per token where cm bit ti) for up to 4 tokens in one launch; token ti =
/// CUBE_POS_Y; a, b, gb, dn are contiguous per token (strides d, d, gs, ds); per element k_comb's arithmetic.
#[cube(launch_unchecked)]
fn k_comb3_m(x0: &mut [f32], x1: &mut [f32], x2: &mut [f32], x3: &mut [f32], c0: &[f32], c1: &[f32], c2: &[f32], c3: &[f32], a: &[f32], b: &[f32], gb: &[f32], dn: &[f32], d: u32, gs: u32, goff: u32, ds: u32, lr: u32, cm: u32, #[comptime] hc: u32) {
    let k = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let mut o = a[(ti * d + i) as usize];
        o = o / (1.0f32 + f32::exp(-gb[(ti * gs + goff) as usize]));
        o += b[(ti * d + i) as usize];
        if ((cm >> ti) & 1u32) == 1u32 {
            if ti == 0u32 { o += c0[i as usize]; } else if ti == 1u32 { o += c1[i as usize]; } else if ti == 2u32 { o += c2[i as usize]; } else { o += c3[i as usize]; }
        }
        let inj = dn[(ti * ds + lr + st) as usize] / f32::cast_from(hc);
        let v = o * 2.0f32 / (1.0f32 + f32::exp(-inj));
        if ti == 0u32 { x0[k as usize] += v; } else if ti == 1u32 { x1[k as usize] += v; } else if ti == 2u32 { x2[k as usize] += v; } else { x3[k as usize] += v; }
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

/// k_silu_mul for t rows in one launch (row ti = CUBE_POS_Y: f at ti * fs, h at ti * n)
#[cube(launch_unchecked)]
fn k_silu_mul_m(f: &[f32], h: &mut [f32], og: u32, ou: u32, n: u32, fs: u32) {
    let j = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if j < n {
        let g = f[(ti * fs + og + j) as usize];
        h[(ti * n + j) as usize] = g / (1.0f32 + f32::exp(-g)) * f[(ti * fs + ou + j) as usize];
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

/// k_conv for all t window tokens in one launch (PYRE_QN_CONVM, default on): per channel the same arithmetic in
/// the same token order as t k_conv launches, and the state after token ti copied to its checkpoint (ckm bit ti),
/// as the per-token k_copy did.
#[cube(launch_unchecked)]
fn k_conv_m(proj: &[f32], w: &[f32], cs: &mut [f32], co: &mut [f32], ck0: &mut [f32], ck1: &mut [f32], ck2: &mut [f32], ck3: &mut [f32], ch: u32, pr: u32, nt: u32, ckm: u32, #[comptime] kk: u32) {
    let c = ABSOLUTE_POS as u32;
    if c < ch {
        for ti in 0..nt {
            let cur = proj[(ti * pr + c) as usize];
            let mut o = w[(c * kk + kk - 1) as usize] * cur;
            #[unroll]
            for k in 0..kk - 1 {
                o += w[(c * kk + k) as usize] * cs[(k * ch + c) as usize];
            }
            co[(ti * ch + c) as usize] = o / (1.0f32 + f32::exp(-o));
            #[unroll]
            for k in 0..kk - 2 {
                cs[(k * ch + c) as usize] = cs[((k + 1) * ch + c) as usize];
            }
            cs[((kk - 2) * ch + c) as usize] = cur;
            if ((ckm >> ti) & 1u32) == 1u32 {
                #[unroll]
                for k in 0..kk - 1 {
                    let v = cs[(k * ch + c) as usize];
                    if ti == 0u32 { ck0[(k * ch + c) as usize] = v; } else if ti == 1u32 { ck1[(k * ch + c) as usize] = v; } else if ti == 2u32 { ck2[(k * ch + c) as usize] = v; } else { ck3[(k * ch + c) as usize] = v; }
                }
            }
        }
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

/// k_delta_m with one pass over the state per token instead of two: the next token's memory read-out
/// mem(t+1) = sum_i S_i(t+1) k_i(t+1) is accumulated in the same loop that writes S(t+1) (= the update of token t), and
/// all tokens' normalised q/k and gates are computed before the loop (inputs only). Per element the products, the
/// ascending-i sums and every float op are k_delta_m's (same values, same order): identical results. Fewer passes over
/// the (spilled) 128-entry state column and two independent chains per pass. Idea of fusing the GDN step after Strata's
/// gdn_step_norm_kernel (MIT, src/kernels/cuda/fused_gdn.cu); Strata's warp-per-column layout itself reorders the sums,
/// so it is not used. PYRE_QN_DELTA2=0 reverts.
#[cube(launch_unchecked)]
fn k_delta_m2(co: &[f32], proj: &[f32], a: &[f32], dt: &[f32], norm: &[f32], s: &mut [f32], out: &mut [f32], c0: &mut [f32], c1: &mut [f32], c2: &mut [f32], c3: &mut [f32],
    nk: u32, ch: u32, pn: u32, ostr: u32, ckm: u32, zoff: u32, boff: u32, aoff: u32, eps: f32, #[comptime] dk: u32, #[comptime] nt: u32) {
    let h = CUBE_POS_X;
    let j = UNIT_POS_X;
    let kh = h % nk;
    let mut qs = Shared::<[f32]>::new_slice(comptime!((dk * nt) as usize));
    let mut ks = Shared::<[f32]>::new_slice(comptime!((dk * nt) as usize));
    let mut red = Shared::<[f32]>::new_slice(comptime!((dk / 32) as usize));
    let base = h * dk * dk;
    let mut st = Array::<f32>::new(comptime!(dk as usize));
    #[unroll]
    for i in 0..dk {
        st[i as usize] = s[(base + i * dk + j) as usize];
    }
    let mut gg = Array::<f32>::new(comptime!(nt as usize));
    let mut bb = Array::<f32>::new(comptime!(nt as usize));
    #[unroll]
    for t in 0..nt {
        let cbs = t * ch;
        let pb = t * pn;
        let qj = co[(cbs + kh * dk + j) as usize];
        let kj = co[(cbs + nk * dk + kh * dk + j) as usize];
        let sq = cube_sum(qj * qj, &mut red, comptime!(dk / 32));
        let sk = cube_sum(kj * kj, &mut red, comptime!(dk / 32));
        qs[(t * dk + j) as usize] = qj / f32::sqrt(sq + 1e-6f32) / f32::sqrt(f32::cast_from(dk));
        ks[(t * dk + j) as usize] = kj / f32::sqrt(sk + 1e-6f32);
        let x = proj[(pb + aoff + h) as usize] + dt[h as usize];
        let mut sp = x;
        if x <= 20.0f32 {
            sp = f32::ln(1.0f32 + f32::exp(x));
        }
        gg[t as usize] = f32::exp(sp * a[h as usize]);
        bb[t as usize] = 1.0f32 / (1.0f32 + f32::exp(-proj[(pb + boff + h) as usize]));
    }
    sync_cube();
    let mut mem = 0.0f32;
    #[unroll]
    for i in 0..dk {
        mem += st[i as usize] * ks[i as usize];
    }
    #[unroll]
    for t in 0..nt {
        let cbs = t * ch;
        let pb = t * pn;
        let g = gg[t as usize];
        let vj = co[(cbs + 2 * nk * dk + h * dk + j) as usize];
        mem *= g;
        let delta = (vj - mem) * bb[t as usize];
        let ck = ((ckm >> t) & 1u32) == 1u32;
        let mut o = 0.0f32;
        let mut mn = 0.0f32;
        #[unroll]
        for i in 0..dk {
            let v = st[i as usize] * g + ks[(t * dk + i) as usize] * delta;
            st[i as usize] = v;
            o += v * qs[(t * dk + i) as usize];
            if t + 1 < nt {
                mn += v * ks[((t + 1) * dk + i) as usize];
            }
            if ck {
                let idx = (base + i * dk + j) as usize;
                if t == 0u32 { c0[idx] = v; } else if t == 1u32 { c1[idx] = v; } else if t == 2u32 { c2[idx] = v; } else { c3[idx] = v; }
            }
        }
        mem = mn;
        let so = cube_sum(o * o, &mut red, comptime!(dk / 32));
        let r = 1.0f32 / f32::sqrt(so / f32::cast_from(dk) + eps);
        let z = proj[(pb + zoff + h * dk + j) as usize];
        out[(t * ostr + h * dk + j) as usize] = o * r * norm[j as usize] / (1.0f32 + f32::exp(-z));
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
    // PYRE_QN_REPACK_OLD=1: the previous growing-Vec loop (same words; the new one only reserves the exact size up front, p3)
    if !repack_old() { return repack_fast(ty, raw); }
    repack_slow(ty, raw)
}
fn repack_old() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_REPACK_OLD").map(|v| v == "1").unwrap_or(false)) }
/// Same output as `repack_slow`, word for word: output sized once, each block's words copied straight from the bytes.
pub fn repack_fast(ty: u32, raw: &[u8]) -> Vec<u32> {
    let f = |b: &[u8], o: usize| half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
    let (bs, nw) = match ty { 0 => (82usize, 21usize), 1 => (66, 17), 2 => (56, 15), _ => (18, 5) };
    let nb = raw.len() / bs; let mut out: Vec<u32> = vec![0; nb * nw];
    for (b, o) in raw.chunks_exact(bs).zip(out.chunks_exact_mut(nw)) {
        match ty {
            0 | 1 | 3.. => { o[0] = (if ty == 0 || ty == 1 { f(b, 0) * 0.25 } else { f(b, 0) }).to_bits();
                for (w, c) in o[1..].iter_mut().zip(b[2..].chunks_exact(4)) { *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]); } }
            _ => { let s = |k: usize| u16::from_le_bytes([b[48 + 2 * k], b[49 + 2 * k]]);
                o[0] = half::f16::from_bits((s(0) >> 12) | ((s(1) >> 8) & 0x00f0) | ((s(2) >> 4) & 0x0f00) | (s(3) & 0xf000)).to_f32().to_bits();
                for (w, c) in o[1..].iter_mut().zip(b.chunks_exact(4)) { *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]); } }
        }
    }
    out
}
pub fn repack_slow(ty: u32, raw: &[u8]) -> Vec<u32> {
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
fn xex8() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XEX8").map(|v| v != "0").unwrap_or(true)) }
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
/// Lossy draft-head GEMV (only the MTP drafter uses it): Q8_0 weights x int8-quantised activations (k_q8pack),
/// one lane per 32-wide block, 8 dp4a per block. Not bit-identical to k_gemv; drafts only (verify decides output).
#[cube(launch_unchecked)]
fn k_gemv_q8d(xq: &[u32], xs: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nb = cols / 32;
        let wb = row * (cols / 4);
        let sb = row * nb;
        let mut acc = 0.0f32;
        let mut b = UNIT_POS_X;
        while b < nb {
            let mut p = 0i32;
            #[unroll]
            for j in 0..8u32 { p = dp4a(i32::cast_from(w[(wb + b * 8 + j) as usize]), i32::cast_from(xq[(b * 8 + j) as usize]), p); }
            acc += f32::cast_from(p) * s[(sb + b) as usize] * xs[b as usize];
            b += 32u32;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 { out[row as usize] = t; }
    }
}
/// Draft-head argmax + top probability (mtp::best on the GPU, one 1024-thread cube): out[0] = index (as f32, exact
/// below 2^24; ties keep the lowest index like best()), out[1] = 1 / sum exp(v - max).
#[cube(launch_unchecked)]
fn k_best(lg: &[f32], out: &mut [f32], n: u32) {
    let mut sv = Shared::<[f32]>::new_slice(1024usize);
    let mut si = Shared::<[u32]>::new_slice(1024usize);
    let tid = UNIT_POS_X;
    let mut bv = f32::new(-3.0e38);
    let mut bi = 0u32;
    let mut i = tid;
    while i < n { let v = lg[i as usize]; if v > bv { bv = v; bi = i; } i += 1024u32; }
    sv[tid as usize] = bv; si[tid as usize] = bi;
    sync_cube();
    let mut h = 512u32;
    while h > 0u32 {
        if tid < h {
            let (a, b) = (sv[tid as usize], sv[(tid + h) as usize]);
            if b > a || (b == a && si[(tid + h) as usize] < si[tid as usize]) { sv[tid as usize] = b; si[tid as usize] = si[(tid + h) as usize]; }
        }
        sync_cube();
        h = h / 2u32;
    }
    let mx = sv[0];
    let bix = si[0];
    sync_cube();
    let mut z = 0.0f32;
    let mut j = tid;
    while j < n { z += f32::exp(lg[j as usize] - mx); j += 1024u32; }
    sv[tid as usize] = z;
    sync_cube();
    let mut h2 = 512u32;
    while h2 > 0u32 {
        if tid < h2 { sv[tid as usize] = sv[tid as usize] + sv[(tid + h2) as usize]; }
        sync_cube();
        h2 = h2 / 2u32;
    }
    if tid == 0u32 { out[0] = f32::cast_from(bix); out[1] = 1.0f32 / sv[0]; }
}
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

/// k_xgu_s over the entries of up to 4 window tokens in one launch: entry e belongs to token
/// (e >= o1) + (e >= o2) + (e >= o3), whose packed activation starts at ti * d/4 words (scales ti * d/32).
/// Per entry and row the same arithmetic as k_xgu_s (PYRE_QN_XBATCH, default on).
/// k_xgu_st / k_xdown_qt for up to 8 window tokens: entry offsets eo[0..=t] from a buffer; per entry/row arithmetic unchanged.
#[cube(launch_unchecked)]
fn k_xgu_st8(xq: &[u32], xs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, eo: &[u32], nt: u32, dq: u32,
    #[comptime] ng: u32, #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let lane = UNIT_POS_X;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let mut ti = 0u32;
    for k in 1..nt { if e >= eo[k as usize] { ti += 1u32; } }
    let xo = ti * dq;
    let so = ti * (dq / 8);
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
            let x0 = i32::cast_from(xq[(xo + q * 2) as usize]);
            let x1 = i32::cast_from(xq[(xo + q * 2 + 1) as usize]);
            let xsc = xs[(so + q / 4) as usize];
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
#[cube(launch_unchecked)]
fn k_xdown_qt8(hq: &[u32], hs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], q2l: &[u32], y: &mut [f32], rows: u32, cols: u32, rbd: u32, eo: &[u32]) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let ti = CUBE_POS_Y;
    let e0 = eo[ti as usize];
    let e1 = eo[(ti + 1) as usize];
    if row < rows {
        let nsb = cols / 8;
        let tot = (e1 - e0) * nsb;
        let mut s = 0.0f32;
        let mut j = UNIT_POS_X;
        while j < tot {
            let el = j / nsb;
            let q = j - el * nsb;
            let e = e0 + el;
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
            y[(ti * rows + row) as usize] = t;
        }
    }
}
#[cube(launch_unchecked)]
fn k_xgu_st(xq: &[u32], xs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, o1: u32, o2: u32, o3: u32, dq: u32,
    #[comptime] ng: u32, #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let lane = UNIT_POS_X;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let mut ti = 0u32;
    if e >= o1 { ti += 1u32; }
    if e >= o2 { ti += 1u32; }
    if e >= o3 { ti += 1u32; }
    let xo = ti * dq;
    let so = ti * (dq / 8);
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
            let x0 = i32::cast_from(xq[(xo + q * 2) as usize]);
            let x1 = i32::cast_from(xq[(xo + q * 2 + 1) as usize]);
            let xsc = xs[(so + q / 4) as usize];
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

/// k_xdown_q per window token (CUBE_POS_Y = ti over its entries [eo[ti], eo[ti+1])), y at ti * rows; per row
/// the same lane mapping and summation order as k_xdown_q over that token's entries.
#[cube(launch_unchecked)]
fn k_xdown_qt(hq: &[u32], hs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], q2l: &[u32], y: &mut [f32], rows: u32, cols: u32, rbd: u32, o1: u32, o2: u32, o3: u32, o4: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let ti = CUBE_POS_Y;
    let mut e0 = 0u32;
    let mut e1 = o1;
    if ti == 1u32 { e0 = o1; e1 = o2; } else if ti == 2u32 { e0 = o2; e1 = o3; } else if ti == 3u32 { e0 = o3; e1 = o4; }
    if row < rows {
        let nsb = cols / 8;
        let tot = (e1 - e0) * nsb;
        let mut s = 0.0f32;
        let mut j = UNIT_POS_X;
        while j < tot {
            let el = j / nsb;
            let q = j - el * nsb;
            let e = e0 + el;
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
            y[(ti * rows + row) as usize] = t;
        }
    }
}
/// PYRE_QN_XEXACT: qdot::xround / qdot::xexp, op for op
#[cube]
fn xround_g(x: f32) -> f32 {
    let v = f32::floor(f32::abs(x) + 0.5f32);
    let mut r = v;
    if x < 0.0f32 { r = -v; }
    r
}
#[cube]
fn xexp_g(x0: f32) -> f32 {
    let x = f32::min(f32::max(x0, -87.0f32), 88.0f32);
    let n = f32::floor(fma(x, f32::reinterpret(1069066811u32), 0.5f32));
    let r1 = fma(-n, f32::reinterpret(1060205056u32), x);
    let r = fma(-n, f32::reinterpret(901758606u32), r1);
    let mut p = f32::reinterpret(961547521u32);
    p = fma(p, r, f32::reinterpret(985008993u32)); p = fma(p, r, f32::reinterpret(1007192201u32)); p = fma(p, r, f32::reinterpret(1026206379u32)); p = fma(p, r, f32::reinterpret(1042983595u32)); p = fma(p, r, 0.5f32); p = fma(p, r, 1.0f32); p = fma(p, r, 1.0f32);
    p * f32::reinterpret(u32::cast_from(i32::cast_from(n) + 127i32) << 23u32)
}
/// PYRE_QN_XEXACT: Q8K activation quantization exactly as qdot::quantize_k_into (one scale per 256, sequential amax,
/// d = amax/127, id = 1/d, q = round(x * id)); CUBE_POS_Y = token (row stride sst), one thread per 256-block.
#[cube(launch_unchecked)]
fn k_q8k_m(src: &[f32], q: &mut [u32], sc: &mut [f32], sst: u32, nb: u32) {
    let b = ABSOLUTE_POS_X;
    let c = CUBE_POS_Y;
    if b < nb {
        let base = c * sst + b * 256;
        let mut amax = 0.0f32;
        for i in 0..256u32 { amax = f32::max(amax, f32::abs(src[(base + i) as usize])); }
        let d = amax / 127.0f32;
        let mut id = 0.0f32;
        if d > 0.0f32 { id = 1.0f32 / d; }
        sc[(c * nb + b) as usize] = d;
        for w in 0..64u32 {
            let mut word = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let v = i32::cast_from(xround_g(src[(base + w * 4 + j) as usize] * id));
                word |= (u32::cast_from(v) & 255u32) << (8 * j);
            }
            q[(c * nb * 64 + b * 64 + w) as usize] = word;
        }
    }
}
/// PYRE_QN_XEXACT: Q8D quantization of n values (one scale per 64, as qdot::quantize_d_into) kept in natural order,
/// with the per-block integer sums; one thread per 64-block.
#[cube(launch_unchecked)]
fn k_q8d(src: &[f32], q: &mut [u32], sc: &mut [f32], ss: &mut [u32], nb: u32) {
    let b = ABSOLUTE_POS_X;
    if b < nb {
        let base = b * 64;
        let mut amax = 0.0f32;
        for i in 0..64u32 { amax = f32::max(amax, f32::abs(src[(base + i) as usize])); }
        let d = amax / 127.0f32;
        let mut id = 0.0f32;
        if d > 0.0f32 { id = 1.0f32 / d; }
        sc[b as usize] = d;
        let mut sm = 0i32;
        for w in 0..16u32 {
            let mut word = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let v = i32::cast_from(xround_g(src[(base + w * 4 + j) as usize] * id));
                sm += v;
                word |= (u32::cast_from(v) & 255u32) << (8 * j);
            }
            q[(b * 16 + w) as usize] = word;
        }
        ss[b as usize] = u32::cast_from(sm);
    }
}
/// signed byte k of a packed word
#[cube]
fn sbyte(w: u32, k: u32) -> i32 {
    let v = i32::cast_from((w >> (8 * k)) & 255u32);
    let mut r = v;
    if v >= 128 { r = v - 256; }
    r
}
/// The integer part of qdot's IQ2 AVX lane L (values 4L..4L+3 of every 32-chunk, times 2*ls+1) for one 256-block.
#[cube]
fn iq2_lane_int(w: &[u32], blk: u32, xq: &[u32], xo: u32, s: u32, lane: u32, grid: &Shared<[u32]>, #[comptime] ty: u32) -> i32 {
    let l = lane / 2;
    let hf = lane % 2;
    let mut acc = 0i32;
    for ib in 0..8u32 {
        let xw = i32::cast_from(xq[(xo + (s * 8 + ib) * 8 + l * 2 + hf) as usize]);
        let mut g = 0i32;
        let mut scl = 0i32;
        if comptime!(ty == 0) {
            let qs = w[(blk + 1 + ib) as usize];
            let sg = w[(blk + 9 + ib) as usize];
            let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
            let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
            let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
            let s8 = (sg >> (8 * l)) & 255u32;
            let m = smask4(s8 >> (4 * hf));
            g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
            scl = 2 * i32::cast_from(sc & 15u32) + 1;
            if lane >= 4 { scl = 2 * i32::cast_from(sc >> 4) + 1; }
        }
        if comptime!(ty == 1) {
            let qw = w[(blk + 1 + 2 * ib) as usize];
            let a1 = w[(blk + 2 + 2 * ib) as usize];
            scl = 2 * i32::cast_from(a1 >> 28) + 1;
            let idx = (qw >> (8 * l)) & 255u32;
            let k7 = (a1 >> (7 * l)) & 127u32;
            let s8 = k7 | ((u32::count_ones(k7) & 1u32) << 7);
            let m = smask4(s8 >> (4 * hf));
            g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
        }
        acc += dp4a(g, xw, 0i32) * scl;
    }
    acc
}
/// iq2_lane_int reading staged (shared) weights and activations: the same integer ops
#[cube]
fn iq2_lane_int_s(w: &Shared<[u32]>, blk: u32, xq: &Shared<[u32]>, xo: u32, s: u32, lane: u32, grid: &Shared<[u32]>, #[comptime] ty: u32) -> i32 {
    let l = lane / 2;
    let hf = lane % 2;
    let mut acc = 0i32;
    for ib in 0..8u32 {
        let xw = i32::cast_from(xq[(xo + (s * 8 + ib) * 8 + l * 2 + hf) as usize]);
        let mut g = 0i32;
        let mut scl = 0i32;
        if comptime!(ty == 0) {
            let qs = w[(blk + 1 + ib) as usize];
            let sg = w[(blk + 9 + ib) as usize];
            let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
            let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
            let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
            let s8 = (sg >> (8 * l)) & 255u32;
            let m = smask4(s8 >> (4 * hf));
            g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
            scl = 2 * i32::cast_from(sc & 15u32) + 1;
            if lane >= 4 { scl = 2 * i32::cast_from(sc >> 4) + 1; }
        }
        if comptime!(ty == 1) {
            let qw = w[(blk + 1 + 2 * ib) as usize];
            let a1 = w[(blk + 2 + 2 * ib) as usize];
            scl = 2 * i32::cast_from(a1 >> 28) + 1;
            let idx = (qw >> (8 * l)) & 255u32;
            let k7 = (a1 >> (7 * l)) & 127u32;
            let s8 = k7 | ((u32::count_ones(k7) & 1u32) << 7);
            let m = smask4(s8 >> (4 * hf));
            g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
        }
        acc += dp4a(g, xw, 0i32) * scl;
    }
    acc
}
/// k_xgu_ex16 with the cube's 16 gate and up rows and the activations staged in shared memory by coalesced loads;
/// identical integer sums and fma order.
#[cube(launch_unchecked)]
fn k_xgu_ex16s(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, nsb: u32, nx: u32,
    #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut fa = Shared::<[f32]>::new_slice(256usize);
    let mut ws = Shared::<[u32]>::new_slice(6720usize);
    let mut xs = Shared::<[u32]>::new_slice(2560usize);
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    let e = CUBE_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let half = ux / 16u32;
    let r = ux % 16u32;
    let lr = wy * 2u32 + half;
    let row = CUBE_POS_X * 16u32 + lr;
    let ein = e < oc[3];
    let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
    let seg = nsb * bw;
    let mut sel = 0u32;
    if ein {
        let m0 = meta[(e * 4) as usize];
        sel = m0 >> 31;
        let g0 = (m0 & 2147483647u32) + CUBE_POS_X * 16u32 * rbg;
        let u0 = meta[(e * 4 + 1) as usize] + CUBE_POS_X * 16u32 * rbu;
        let mut k = UNIT_POS;
        while k < 32u32 * seg {
            let sg = k / seg;
            let c = k - sg * seg;
            let rr = sg / 2u32;
            let mut src = g0 + rr * rbg + c;
            if sg % 2u32 == 1u32 { src = u0 + rr * rbu + c; }
            if CUBE_POS_X * 16u32 + rr < rows {
                if sel == 0u32 { ws[k as usize] = ar[src as usize]; } else { ws[k as usize] = ar1[src as usize]; }
            }
            k += CUBE_DIM;
        }
        let mut k2 = UNIT_POS;
        while k2 < nx { xs[k2 as usize] = xq[k2 as usize]; k2 += CUBE_DIM; }
    }
    sync_cube();
    let act = ein && row < rows;
    if act {
        let mut ti = 0u32;
        if e >= oc[0] { ti += 1u32; }
        if e >= oc[1] { ti += 1u32; }
        if e >= oc[2] { ti += 1u32; }
        let xo = ti * nsb * 64;
        let isu = r / 8u32;
        let base = (lr * 2u32 + isu) * seg;
        let lane = r % 8u32;
        let mut acc = 0.0f32;
        for s in 0..nsb {
            let iv = iq2_lane_int_s(&ws, base + s * bw, &xs, xo, s, lane, &grid, ty);
            let w0 = f32::reinterpret(ws[(base + s * bw) as usize]);
            let d = (w0 * 0.5f32) * xd[(ti * nsb + s) as usize];
            acc = fma(f32::cast_from(iv), d, acc);
        }
        fa[(wy * 32 + ux) as usize] = acc;
    }
    sync_cube();
    if act && r == 0u32 {
        let f = wy * 32 + half * 16;
        let g = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let u = ((fa[(f + 8) as usize] + fa[(f + 12) as usize]) + (fa[(f + 10) as usize] + fa[(f + 14) as usize])) + ((fa[(f + 9) as usize] + fa[(f + 13) as usize]) + (fa[(f + 11) as usize] + fa[(f + 15) as usize]));
        let ex = xexp_g(-g);
        let sg2 = 1.0f32 / (1.0f32 + ex);
        let sl = g * sg2;
        h[(e * rows + row) as usize] = sl * u;
    }
}
/// PYRE_QN_XEXACT gate/up: h = silu(g) * u with g, u computed as qdot::iq2s_k_b / iq2xxs_k_b do (8 integer lanes per
/// 256-block, lane-wise fma over the blocks in order, the AVX horizontal-sum tree), silu as x * (1 / (1 + exp(-x))).
/// One 32-thread cube per (row, entry); CUBE_POS_Y = entry; entries >= oc[3] are padding.
#[cube(launch_unchecked)]
fn k_xgu_ex(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, nsb: u32,
    #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut ig = Shared::<[i32]>::new_slice(256usize);
    let mut fa = Shared::<[f32]>::new_slice(16usize);
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X;
    let act = e < oc[3];
    let mut ti = 0u32;
    if e >= oc[0] { ti += 1u32; }
    if e >= oc[1] { ti += 1u32; }
    if e >= oc[2] { ti += 1u32; }
    let xo = ti * nsb * 64;
    let mut go = 0u32;
    let mut uo = 0u32;
    let mut sel = 0u32;
    if act {
        let m0 = meta[(e * 4) as usize];
        sel = m0 >> 31;
        go = (m0 & 2147483647u32) + row * rbg;
        uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let lane = UNIT_POS % 8;
        let mut s = UNIT_POS / 8;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        while s < nsb {
            if sel == 0u32 {
                ig[(s * 8 + lane) as usize] = iq2_lane_int(ar, go + s * bw, xq, xo, s, lane, &grid, ty);
                ig[(128 + s * 8 + lane) as usize] = iq2_lane_int(ar, uo + s * bw, xq, xo, s, lane, &grid, ty);
            } else {
                ig[(s * 8 + lane) as usize] = iq2_lane_int(ar1, go + s * bw, xq, xo, s, lane, &grid, ty);
                ig[(128 + s * 8 + lane) as usize] = iq2_lane_int(ar1, uo + s * bw, xq, xo, s, lane, &grid, ty);
            }
            s += 4u32;
        }
    }
    sync_cube();
    if act && UNIT_POS < 16 {
        let lane = UNIT_POS % 8;
        let isu = UNIT_POS / 8;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        let mut base = go;
        if isu == 1u32 { base = uo; }
        let mut acc = 0.0f32;
        for s in 0..nsb {
            let mut w0 = 0.0f32;
            if sel == 0u32 { w0 = f32::reinterpret(ar[(base + s * bw) as usize]); } else { w0 = f32::reinterpret(ar1[(base + s * bw) as usize]); }
            let d = (w0 * 0.5f32) * xd[(ti * nsb + s) as usize];
            acc = fma(f32::cast_from(ig[(isu * 128 + s * 8 + lane) as usize]), d, acc);
        }
        fa[UNIT_POS as usize] = acc;
    }
    sync_cube();
    if act && UNIT_POS == 0 {
        let g = ((fa[0] + fa[4]) + (fa[2] + fa[6])) + ((fa[1] + fa[5]) + (fa[3] + fa[7]));
        let u = ((fa[8] + fa[12]) + (fa[10] + fa[14])) + ((fa[9] + fa[13]) + (fa[11] + fa[15]));
        let ex = xexp_g(-g);
        let sg = 1.0f32 / (1.0f32 + ex);
        let sl = g * sg;
        h[(e * rows + row) as usize] = sl * u;
    }
}
/// PYRE_QN_XEXACT down (Q2_0 rows, Q8D h): as qdot::q2_0_d_avx2 (8 integer lanes per 64-block, lane-wise fma, the
/// correction term subtracted block by block, corr + hsum), times the routing weight; one output per (entry, row).
#[cube(launch_unchecked)]
fn k_xdown_ex(hq: &[u32], hd: &[f32], hs: &[u32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], y: &mut [f32], rows: u32, nb: u32, rbd: u32) {
    let mut pi = Shared::<[i32]>::new_slice(512usize);
    let mut fa = Shared::<[f32]>::new_slice(9usize);
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X;
    let act = e < oc[3];
    let mut dof = 0u32;
    let mut sel = 0u32;
    if act {
        let m2 = meta[(e * 4 + 2) as usize];
        sel = m2 >> 31;
        dof = (m2 & 2147483647u32) + row * rbd;
        let m = UNIT_POS % 8;
        let mut i = UNIT_POS / 8;
        let k0 = m / 4;
        let j0 = (m % 4) * 4;
        while i < nb {
            let blk = dof + i * 5;
            let mut p = 0i32;
            let mut wq = 0u32;
            if sel == 0u32 { wq = ar[(blk + 1 + j0 / 4) as usize]; } else { wq = ar1[(blk + 1 + j0 / 4) as usize]; }
            for jj in 0..4u32 {
                let j = j0 + jj;
                let qv = (wq >> (8 * (j % 4))) & 255u32;
                let xw = hq[(e * nb * 16 + i * 16 + j) as usize];
                let pk = (((qv >> (2 * k0)) & 3u32) << (8 * k0)) | (((qv >> (2 * k0 + 4)) & 3u32) << (8 * k0 + 16));
                p += dp4a(i32::cast_from(pk), i32::cast_from(xw), 0i32);
            }
            pi[(i * 8 + m) as usize] = p;
            i += 4u32;
        }
    }
    sync_cube();
    if act && UNIT_POS < 9 {
        let mut acc = 0.0f32;
        for i in 0..nb {
            let mut w0 = 0.0f32;
            if sel == 0u32 { w0 = f32::reinterpret(ar[(dof + i * 5) as usize]); } else { w0 = f32::reinterpret(ar1[(dof + i * 5) as usize]); }
            let dd = w0 * hd[(e * nb + i) as usize];
            if UNIT_POS < 8 {
                acc = fma(f32::cast_from(pi[(i * 8 + UNIT_POS) as usize]), dd, acc);
            } else {
                let pr = fma(dd, f32::cast_from(i32::cast_from(hs[(e * nb + i) as usize])), 0.0f32);
                acc = acc - pr;
            }
        }
        fa[UNIT_POS as usize] = acc;
    }
    sync_cube();
    if act && UNIT_POS == 0 {
        let hsum = ((fa[0] + fa[4]) + (fa[2] + fa[6])) + ((fa[1] + fa[5]) + (fa[3] + fa[7]));
        let v = fa[8] + hsum;
        let wt = f32::reinterpret(meta[(e * 4 + 3) as usize]);
        y[(e * rows + row) as usize] = fma(wt, v, 0.0f32);
    }
}
/// k_xgu_ex / k_xdown_ex with 8 rows (one warp each) per cube: same per-row lane order and float steps.
#[cube(launch_unchecked)]
fn k_xgu_ex8(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, nsb: u32,
    #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut ig = Shared::<[i32]>::new_slice(4096usize);
    let mut fa = Shared::<[f32]>::new_slice(256usize);
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let act = e < oc[3] && row < rows;
    let mut ti = 0u32;
    if e >= oc[0] { ti += 1u32; }
    if e >= oc[1] { ti += 1u32; }
    if e >= oc[2] { ti += 1u32; }
    let xo = ti * nsb * 64;
    let mut go = 0u32;
    let mut uo = 0u32;
    let mut sel = 0u32;
    if act {
        let m0 = meta[(e * 4) as usize];
        sel = m0 >> 31;
        go = (m0 & 2147483647u32) + row * rbg;
        uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let lane = ux % 8;
        let mut s = ux / 8;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        while s < nsb {
            if sel == 0u32 {
                ig[(wy * 256 + s * 8 + lane) as usize] = iq2_lane_int(ar, go + s * bw, xq, xo, s, lane, &grid, ty);
                ig[(wy * 256 + 128 + s * 8 + lane) as usize] = iq2_lane_int(ar, uo + s * bw, xq, xo, s, lane, &grid, ty);
            } else {
                ig[(wy * 256 + s * 8 + lane) as usize] = iq2_lane_int(ar1, go + s * bw, xq, xo, s, lane, &grid, ty);
                ig[(wy * 256 + 128 + s * 8 + lane) as usize] = iq2_lane_int(ar1, uo + s * bw, xq, xo, s, lane, &grid, ty);
            }
            s += 4u32;
        }
    }
    sync_cube();
    if act && ux < 16 {
        let lane = ux % 8;
        let isu = ux / 8;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        let mut base = go;
        if isu == 1u32 { base = uo; }
        let mut acc = 0.0f32;
        for s in 0..nsb {
            let mut w0 = 0.0f32;
            if sel == 0u32 { w0 = f32::reinterpret(ar[(base + s * bw) as usize]); } else { w0 = f32::reinterpret(ar1[(base + s * bw) as usize]); }
            let d = (w0 * 0.5f32) * xd[(ti * nsb + s) as usize];
            acc = fma(f32::cast_from(ig[(wy * 256 + isu * 128 + s * 8 + lane) as usize]), d, acc);
        }
        fa[(wy * 16 + ux) as usize] = acc;
    }
    sync_cube();
    if act && ux == 0 {
        let f = wy * 16;
        let g = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let u = ((fa[(f + 8) as usize] + fa[(f + 12) as usize]) + (fa[(f + 10) as usize] + fa[(f + 14) as usize])) + ((fa[(f + 9) as usize] + fa[(f + 13) as usize]) + (fa[(f + 11) as usize] + fa[(f + 15) as usize]));
        let ex = xexp_g(-g);
        let sg = 1.0f32 / (1.0f32 + ex);
        let sl = g * sg;
        h[(e * rows + row) as usize] = sl * u;
    }
}
/// PYRE_QN_XGU_ROWS (8 or 16, default 8; 16 was slower in nsys e9): rows per cube of k_xgu_ex8
fn xgu_rows() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGU_ROWS").ok().and_then(|v| v.parse().ok()).filter(|&v: &u32| v == 8 || v == 16).unwrap_or(8)) }
/// PYRE_QN_XEXACT gate/up, two rows per warp (16 lanes each: 8 integer lanes x gate/up), no shared integer stage:
/// each lane does its own blocks in order and accumulates at once, the same per-lane fma order and hsum as k_xgu_ex8.
#[cube(launch_unchecked)]
fn k_xgu_ex16(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, nsb: u32,
    #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut fa = Shared::<[f32]>::new_slice(256usize);
    // entry slots past the window's hit count (PYRE_QN_XGR launches t*k slots) leave at once
    if CUBE_POS_Y >= oc[3] { terminate!(); }
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let half = ux / 16u32;
    let r = ux % 16u32;
    let row = (CUBE_POS_X * CUBE_DIM_Y + wy) * 2u32 + half;
    let act = e < oc[3] && row < rows;
    if act {
        let mut ti = 0u32;
        if e >= oc[0] { ti += 1u32; }
        if e >= oc[1] { ti += 1u32; }
        if e >= oc[2] { ti += 1u32; }
        let xo = ti * nsb * 64;
        let m0 = meta[(e * 4) as usize];
        let sel = m0 >> 31;
        let mut base = (m0 & 2147483647u32) + row * rbg;
        if r >= 8u32 { base = meta[(e * 4 + 1) as usize] + row * rbu; }
        let lane = r % 8u32;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        let mut acc = 0.0f32;
        for s in 0..nsb {
            let mut iv = 0i32;
            let mut w0 = 0.0f32;
            if sel == 0u32 { iv = iq2_lane_int(ar, base + s * bw, xq, xo, s, lane, &grid, ty); w0 = f32::reinterpret(ar[(base + s * bw) as usize]); }
            else { iv = iq2_lane_int(ar1, base + s * bw, xq, xo, s, lane, &grid, ty); w0 = f32::reinterpret(ar1[(base + s * bw) as usize]); }
            let d = (w0 * 0.5f32) * xd[(ti * nsb + s) as usize];
            acc = fma(f32::cast_from(iv), d, acc);
        }
        fa[(wy * 32 + ux) as usize] = acc;
    }
    sync_cube();
    if act && r == 0u32 {
        let f = wy * 32 + half * 16;
        let g = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let u = ((fa[(f + 8) as usize] + fa[(f + 12) as usize]) + (fa[(f + 10) as usize] + fa[(f + 14) as usize])) + ((fa[(f + 9) as usize] + fa[(f + 13) as usize]) + (fa[(f + 11) as usize] + fa[(f + 15) as usize]));
        let ex = xexp_g(-g);
        let sg = 1.0f32 / (1.0f32 + ex);
        let sl = g * sg;
        h[(e * rows + row) as usize] = sl * u;
    }
}
/// PYRE_QN_XGU2: 1 (default) k_xgu_ex16, 5 k_xgu_ex16s (staged; e16 same trace 172 vs 114 us, slower), 6 alternate ex16s/ex16, 3 k_xgu_ex64 (fused Q8D; e14 same trace 129 us vs 114+9 us, not faster), 0 k_xgu_ex8, 2 alternate ex16/ex8, 4 alternate ex64/ex16 (one trace)
fn xgu2() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGU2").ok().and_then(|v| v.parse().ok()).unwrap_or(1)) }
static XGU2_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// PYRE_QN_XEXACT down, two rows per warp (16 lanes each: lanes 0..7 the q2_0_d_avx2 lanes, lane 8 the block correction),
/// each lane accumulating its blocks in order (same per-lane fma order and final sum as k_xdown_ex8), no shared integer stage.
#[cube(launch_unchecked)]
fn k_xdown_ex16(hq: &[u32], hd: &[f32], hs: &[u32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], y: &mut [f32], rows: u32, nb: u32, rbd: u32) {
    if CUBE_POS_Y >= oc[3] { terminate!(); }
    let mut fa = Shared::<[f32]>::new_slice(256usize);
    let e = CUBE_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let half = ux / 16u32;
    let r = ux % 16u32;
    let row = (CUBE_POS_X * CUBE_DIM_Y + wy) * 2u32 + half;
    let act = e < oc[3] && row < rows;
    if act && r < 9u32 {
        let m2 = meta[(e * 4 + 2) as usize];
        let sel = m2 >> 31;
        let dof = (m2 & 2147483647u32) + row * rbd;
        let k0 = (r % 8u32) / 4u32;
        let j0 = (r % 4u32) * 4u32;
        let mut acc = 0.0f32;
        for i in 0..nb {
            let blk = dof + i * 5;
            let mut w0 = 0.0f32;
            let mut wq = 0u32;
            if sel == 0u32 { w0 = f32::reinterpret(ar[blk as usize]); wq = ar[(blk + 1 + j0 / 4) as usize]; } else { w0 = f32::reinterpret(ar1[blk as usize]); wq = ar1[(blk + 1 + j0 / 4) as usize]; }
            let dd = w0 * hd[(e * nb + i) as usize];
            if r < 8u32 {
                let mut p = 0i32;
                for jj in 0..4u32 {
                    let j = j0 + jj;
                    let qv = (wq >> (8 * (j % 4))) & 255u32;
                    let xw = hq[(e * nb * 16 + i * 16 + j) as usize];
                    let pk = (((qv >> (2 * k0)) & 3u32) << (8 * k0)) | (((qv >> (2 * k0 + 4)) & 3u32) << (8 * k0 + 16));
                    p += dp4a(i32::cast_from(pk), i32::cast_from(xw), 0i32);
                }
                acc = fma(f32::cast_from(p), dd, acc);
            } else {
                let pr = fma(dd, f32::cast_from(i32::cast_from(hs[(e * nb + i) as usize])), 0.0f32);
                acc = acc - pr;
            }
        }
        fa[(wy * 32 + ux) as usize] = acc;
    }
    sync_cube();
    if act && r == 0u32 {
        let f = wy * 32 + half * 16;
        let hsum = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let v = fa[(f + 8) as usize] + hsum;
        let wt = f32::reinterpret(meta[(e * 4 + 3) as usize]);
        y[(e * rows + row) as usize] = fma(wt, v, 0.0f32);
    }
}
/// PYRE_QN_XGX bit 0: grouped exact gate/up. One cube per (row chunk, distinct expert): the first entry of each expert
/// (leader; later entries with the same gate offset are the other window tokens routed to it) loads every weight word
/// once and applies it to all its tokens. One thread per (row, gate|up) holds the 8 k_xgu_ex16 lane accumulators of
/// each token: per lane the same integer sums and the same fma chain over the 256-blocks, then the same lane tree,
/// so every h value is bit-identical to k_xgu_ex16 (and the CPU pool). Activations staged in shared memory.
#[cube(launch_unchecked)]
fn k_xgu_gx(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32,
    #[comptime] nsb: u32, #[comptime] gn: u32, #[comptime] ty: u32, #[comptime] lp: u32) {
    if CUBE_POS_Y >= oc[3] { terminate!(); }
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut xs = Shared::<[u32]>::new_slice(comptime!((4 * nsb * 64) as usize));
    let mut gm = Shared::<[u32]>::new_slice(8usize);
    let mut fa = Shared::<[f32]>::new_slice(4096usize);
    let e = CUBE_POS_Y;
    if UNIT_POS == 0u32 {
        let n = oc[3];
        let me = meta[(e * 4) as usize];
        let mut lead = 1u32;
        let mut cnt = 0u32;
        for k in 0..n {
            if meta[(k * 4) as usize] == me {
                if k < e { lead = 0u32; } else { if cnt < 4u32 { gm[(1 + cnt) as usize] = k; cnt += 1u32; } }
            }
        }
        gm[0] = cnt * lead;
    }
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let cnt = gm[0];
    if cnt == 0u32 { terminate!(); }
    let nx = nsb * 64u32;
    let mut k = UNIT_POS;
    while k < cnt * nx {
        let j = k / nx;
        let ej = gm[(1 + j) as usize];
        let mut ti = 0u32;
        if ej >= oc[0] { ti += 1u32; }
        if ej >= oc[1] { ti += 1u32; }
        if ej >= oc[2] { ti += 1u32; }
        xs[k as usize] = xq[(ti * nx + k % nx) as usize];
        k += CUBE_DIM;
    }
    sync_cube();
    let mut et = Array::<u32>::new(4usize);
    let mut tt = Array::<u32>::new(4usize);
    #[unroll]
    for j in 0..4u32 {
        let mut ej = e;
        if j < cnt { ej = gm[(1 + j) as usize]; }
        let mut ti = 0u32;
        if ej >= oc[0] { ti += 1u32; }
        if ej >= oc[1] { ti += 1u32; }
        if ej >= oc[2] { ti += 1u32; }
        et[j as usize] = ej;
        tt[j as usize] = ti;
    }
    // thread = (row, gate|up, lane group of lp lanes)
    let ng = comptime!(8u32 / lp);
    let part = UNIT_POS % 2u32;
    let lg = (UNIT_POS / 2u32) % ng;
    let rl = UNIT_POS / (2u32 * ng);
    let rpc = CUBE_DIM / (2u32 * ng);
    let row = CUBE_POS_X * rpc + rl;
    let act = row < rows;
    let mut rr = row;
    if !act { rr = rows - 1u32; }
    let m0 = meta[(e * 4) as usize];
    let sel = m0 >> 31;
    let mut base = (m0 & 2147483647u32) + rr * rbg;
    if part == 1u32 { base = meta[(e * 4 + 1) as usize] + rr * rbu; }
    let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
    let mut acc = Array::<f32>::new(comptime!((4 * lp) as usize));
    #[unroll]
    for q in 0..4 * lp { acc[q as usize] = 0.0f32; }
    let mut iacc = Array::<i32>::new(comptime!((4 * lp) as usize));
    let mut wv = Array::<u32>::new(comptime!(bw as usize));
    for s in 0..nsb {
        let b0 = base + s * bw;
        #[unroll]
        for q in 0..bw {
            if sel == 0u32 { wv[q as usize] = ar[(b0 + q) as usize]; } else { wv[q as usize] = ar1[(b0 + q) as usize]; }
        }
        #[unroll]
        for q in 0..4 * lp { iacc[q as usize] = 0i32; }
        #[unroll]
        for ib in 0..8u32 {
            #[unroll]
            for lq in 0..lp {
                let ln = lg * lp + lq;
                let l = ln / 2u32;
                let hf = ln % 2u32;
                let mut g = 0i32;
                let mut scl = 0i32;
                if comptime!(ty == 0) {
                    let qs = wv[(1 + ib) as usize];
                    let sg = wv[(9 + ib) as usize];
                    let qh = (wv[(17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
                    let sc = (wv[(19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
                    let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
                    let s8 = (sg >> (8 * l)) & 255u32;
                    let m = smask4(s8 >> (4 * hf));
                    g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
                    scl = 2 * i32::cast_from((sc >> ((ln / 4u32) * 4u32)) & 15u32) + 1;
                }
                if comptime!(ty == 1) {
                    let qw = wv[(1 + 2 * ib) as usize];
                    let a1 = wv[(2 + 2 * ib) as usize];
                    scl = 2 * i32::cast_from(a1 >> 28) + 1;
                    let idx = (qw >> (8 * l)) & 255u32;
                    let k7 = (a1 >> (7 * l)) & 127u32;
                    let s8 = k7 | ((u32::count_ones(k7) & 1u32) << 7);
                    let m = smask4(s8 >> (4 * hf));
                    g = i32::cast_from((grid[(idx * 2 + hf) as usize] ^ m) + (m & 16843009u32));
                }
                #[unroll]
                for j in 0..4u32 {
                    if j < cnt {
                        let xw = i32::cast_from(xs[(j * nx + (s * 8 + ib) * 8 + ln) as usize]);
                        iacc[(j * lp + lq) as usize] += dp4a(g, xw, 0i32) * scl;
                    }
                }
            }
        }
        let w0 = f32::reinterpret(wv[0]);
        #[unroll]
        for j in 0..4u32 {
            if j < cnt {
                let d = (w0 * 0.5f32) * xd[(tt[j as usize] * nsb + s) as usize];
                #[unroll]
                for lq in 0..lp {
                    let q = (j * lp + lq) as usize;
                    acc[q] = fma(f32::cast_from(iacc[q]), d, acc[q]);
                }
            }
        }
    }
    // lane values to shared: fa[((j * rpc + rl) * 2 + part) * 8 + lane], then k_xgu_ex16's lane trees per (token, row)
    #[unroll]
    for j in 0..4u32 {
        if j < cnt {
            #[unroll]
            for lq in 0..lp { fa[(((j * rpc + rl) * 2 + part) * 8 + lg * lp + lq) as usize] = acc[(j * lp + lq) as usize]; }
        }
    }
    sync_cube();
    if act && part == 0u32 && lg == 0u32 {
        #[unroll]
        for j in 0..4u32 {
            if j < cnt {
                let f = (j * rpc + rl) * 16;
                let g = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
                let u = ((fa[(f + 8) as usize] + fa[(f + 12) as usize]) + (fa[(f + 10) as usize] + fa[(f + 14) as usize])) + ((fa[(f + 9) as usize] + fa[(f + 13) as usize]) + (fa[(f + 11) as usize] + fa[(f + 15) as usize]));
                let ex = xexp_g(-g);
                let sg = 1.0f32 / (1.0f32 + ex);
                let sl = g * sg;
                h[(et[j as usize] * rows + row) as usize] = sl * u;
            }
        }
    }
}
/// PYRE_QN_XGX bit 1: grouped exact down. One thread per output row and distinct expert, all its window tokens:
/// the 8 k_xdown_ex16 lane accumulators + the block-correction accumulator per token, same integer sums, same fma
/// order over the 64-blocks, same final tree and weight fma -> bit-identical y. hq/hd/hs of the group staged in shared.
#[cube(launch_unchecked)]
fn k_xdown_gx(hq: &[u32], hd: &[f32], hs: &[u32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], y: &mut [f32], rows: u32, rbd: u32, #[comptime] nb: u32) {
    if CUBE_POS_Y >= oc[3] { terminate!(); }
    let mut gm = Shared::<[u32]>::new_slice(8usize);
    let mut xs = Shared::<[u32]>::new_slice(comptime!((4 * nb * 16) as usize));
    let mut ds = Shared::<[f32]>::new_slice(comptime!((4 * nb) as usize));
    let mut ss = Shared::<[i32]>::new_slice(comptime!((4 * nb) as usize));
    let e = CUBE_POS_Y;
    if UNIT_POS == 0u32 {
        let n = oc[3];
        let me = meta[(e * 4) as usize];
        let mut lead = 1u32;
        let mut cnt = 0u32;
        for k in 0..n {
            if meta[(k * 4) as usize] == me {
                if k < e { lead = 0u32; } else { if cnt < 4u32 { gm[(1 + cnt) as usize] = k; cnt += 1u32; } }
            }
        }
        gm[0] = cnt * lead;
    }
    sync_cube();
    let cnt = gm[0];
    if cnt == 0u32 { terminate!(); }
    let nx = nb * 16u32;
    let mut k = UNIT_POS;
    while k < cnt * nx {
        let ej = gm[(1 + k / nx) as usize];
        xs[k as usize] = hq[(ej * nx + k % nx) as usize];
        k += CUBE_DIM;
    }
    let mut k2 = UNIT_POS;
    while k2 < cnt * nb {
        let ej = gm[(1 + k2 / nb) as usize];
        ds[k2 as usize] = hd[(ej * nb + k2 % nb) as usize];
        ss[k2 as usize] = i32::cast_from(hs[(ej * nb + k2 % nb) as usize]);
        k2 += CUBE_DIM;
    }
    sync_cube();
    let row = CUBE_POS_X * CUBE_DIM + UNIT_POS;
    if row < rows {
        let m2 = meta[(e * 4 + 2) as usize];
        let sel = m2 >> 31;
        let dof = (m2 & 2147483647u32) + row * rbd;
        let mut acc = Array::<f32>::new(36usize);
        #[unroll]
        for q in 0..36u32 { acc[q as usize] = 0.0f32; }
        let mut wq = Array::<u32>::new(4usize);
        for i in 0..nb {
            let blk = dof + i * 5;
            let mut w0 = 0.0f32;
            if sel == 0u32 {
                w0 = f32::reinterpret(ar[blk as usize]);
                #[unroll]
                for q in 0..4u32 { wq[q as usize] = ar[(blk + 1 + q) as usize]; }
            } else {
                w0 = f32::reinterpret(ar1[blk as usize]);
                #[unroll]
                for q in 0..4u32 { wq[q as usize] = ar1[(blk + 1 + q) as usize]; }
            }
            #[unroll]
            for r in 0..8u32 {
                let k0 = r / 4u32;
                let wr = wq[(r % 4u32) as usize];
                #[unroll]
                for j in 0..4u32 {
                    if j < cnt {
                        let mut p = 0i32;
                        #[unroll]
                        for jj in 0..4u32 {
                            let qv = (wr >> (8 * jj)) & 255u32;
                            let xw = xs[(j * nx + i * 16 + (r % 4u32) * 4 + jj) as usize];
                            let pk = (((qv >> (2 * k0)) & 3u32) << (8 * k0)) | (((qv >> (2 * k0 + 4)) & 3u32) << (8 * k0 + 16));
                            p += dp4a(i32::cast_from(pk), i32::cast_from(xw), 0i32);
                        }
                        let dd = w0 * ds[(j * nb + i) as usize];
                        let q = (j * 9 + r) as usize;
                        acc[q] = fma(f32::cast_from(p), dd, acc[q]);
                    }
                }
            }
            #[unroll]
            for j in 0..4u32 {
                if j < cnt {
                    let dd = w0 * ds[(j * nb + i) as usize];
                    let pr = fma(dd, f32::cast_from(ss[(j * nb + i) as usize]), 0.0f32);
                    let q = (j * 9 + 8) as usize;
                    acc[q] = acc[q] - pr;
                }
            }
        }
        #[unroll]
        for j in 0..4u32 {
            if j < cnt {
                let f = j * 9;
                let hsum = ((acc[f as usize] + acc[(f + 4) as usize]) + (acc[(f + 2) as usize] + acc[(f + 6) as usize])) + ((acc[(f + 1) as usize] + acc[(f + 5) as usize]) + (acc[(f + 3) as usize] + acc[(f + 7) as usize]));
                let v = acc[(f + 8) as usize] + hsum;
                let ej = gm[(1 + j) as usize];
                let wt = f32::reinterpret(meta[(ej * 4 + 3) as usize]);
                y[(ej * rows + row) as usize] = fma(wt, v, 0.0f32);
            }
        }
    }
}
/// PYRE_QN_XGX: bit 0 k_xgu_gx, bit 1 k_xdown_gx (grouped exact expert kernels, default 3 = both; 0 = old ex16 kernels); XEX_OVR overrides it (xbench)
pub static XEX_OVR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
/// xrun_exact stage mask (xbench timing only): 1 q8k, 2 gate/up, 4 q8d, 8 down
pub static XEX_STG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(15);
fn xgx_mode() -> u32 { let o = XEX_OVR.load(std::sync::atomic::Ordering::Relaxed); if o != u32::MAX { return o; }
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGX").ok().and_then(|v| v.parse().ok()).unwrap_or(3)) }
/// PYRE_QN_XGX_R: k_xgu_gx rows per cube (2 threads a row; 16..256, default 64)
fn xgx_r() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGX_R").ok().and_then(|v| v.parse().ok()).filter(|&v: &u32| (16..=256).contains(&v) && v % 16 == 0).unwrap_or(64)) }
/// PYRE_QN_XGX_LP: k_xgu_gx lanes per thread (8, 4, 2; default 4); XGX_LP_OVR overrides (xbench)
pub static XGX_LP_OVR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn xgx_lp() -> u32 { let o = XGX_LP_OVR.load(std::sync::atomic::Ordering::Relaxed); if o != 0 { return o; }
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGX_LP").ok().and_then(|v| v.parse().ok()).filter(|&v: &u32| v == 8 || v == 4 || v == 2).unwrap_or(4)) }
/// PYRE_QN_XGX_DR: k_xdown_gx rows (threads) per cube (32..512, default 128)
fn xgx_dr() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XGX_DR").ok().and_then(|v| v.parse().ok()).filter(|&v: &u32| (32..=512).contains(&v) && v % 32 == 0).unwrap_or(128)) }
/// PYRE_QN_XDN2: 1 (default) k_xdown_ex16, 3 k_xdown_ex16s (staged; e15 same trace 116 vs 93 us, slower), 0 k_xdown_ex8, 2 alternate ex16/ex8, 4 alternate ex16s/ex16
fn xdn2() -> u32 { static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XDN2").ok().and_then(|v| v.parse().ok()).unwrap_or(1)) }
static XDN2_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// k_xgu_ex16 with 64 rows per cube (grid table loaded once per 64 rows) and the Q8D quantize of those 64 h values
/// (one block, exactly k_q8d's ops) fused into the epilogue, so k_q8d is not launched.
#[cube(launch_unchecked)]
fn k_xgu_ex64(xq: &[u32], xd: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], gsrc: &[u32], h: &mut [f32], hq: &mut [u32], hd: &mut [f32], hs: &mut [u32], rows: u32, rbg: u32, rbu: u32, nsb: u32,
    #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut fa = Shared::<[f32]>::new_slice(1024usize);
    let mut hv = Shared::<[f32]>::new_slice(64usize);
    let mut si = Shared::<[i32]>::new_slice(16usize);
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let half = ux / 16u32;
    let r = ux % 16u32;
    let lr = wy * 2u32 + half;
    let row = CUBE_POS_X * 64u32 + lr;
    let act = e < oc[3];
    if act {
        let mut ti = 0u32;
        if e >= oc[0] { ti += 1u32; }
        if e >= oc[1] { ti += 1u32; }
        if e >= oc[2] { ti += 1u32; }
        let xo = ti * nsb * 64;
        let m0 = meta[(e * 4) as usize];
        let sel = m0 >> 31;
        let mut base = (m0 & 2147483647u32) + row * rbg;
        if r >= 8u32 { base = meta[(e * 4 + 1) as usize] + row * rbu; }
        let lane = r % 8u32;
        let bw = comptime!(if ty == 0 { 21u32 } else { 17u32 });
        let mut acc = 0.0f32;
        for s in 0..nsb {
            let mut iv = 0i32;
            let mut w0 = 0.0f32;
            if sel == 0u32 { iv = iq2_lane_int(ar, base + s * bw, xq, xo, s, lane, &grid, ty); w0 = f32::reinterpret(ar[(base + s * bw) as usize]); }
            else { iv = iq2_lane_int(ar1, base + s * bw, xq, xo, s, lane, &grid, ty); w0 = f32::reinterpret(ar1[(base + s * bw) as usize]); }
            let d = (w0 * 0.5f32) * xd[(ti * nsb + s) as usize];
            acc = fma(f32::cast_from(iv), d, acc);
        }
        fa[(wy * 32 + ux) as usize] = acc;
    }
    sync_cube();
    if act && r == 0u32 {
        let f = wy * 32 + half * 16;
        let g = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let u = ((fa[(f + 8) as usize] + fa[(f + 12) as usize]) + (fa[(f + 10) as usize] + fa[(f + 14) as usize])) + ((fa[(f + 9) as usize] + fa[(f + 13) as usize]) + (fa[(f + 11) as usize] + fa[(f + 15) as usize]));
        let ex = xexp_g(-g);
        let sg = 1.0f32 / (1.0f32 + ex);
        let sl = g * sg;
        let hvv = sl * u;
        hv[lr as usize] = hvv;
        h[(e * rows + row) as usize] = hvv;
    }
    sync_cube();
    let tq = UNIT_POS;
    if act && tq < 16u32 {
        let b = e * (rows / 64u32) + CUBE_POS_X;
        let mut amax = 0.0f32;
        for k in 0..64u32 { amax = f32::max(amax, f32::abs(hv[k as usize])); }
        let dq = amax / 127.0f32;
        let mut id = 0.0f32;
        if dq > 0.0f32 { id = 1.0f32 / dq; }
        if tq == 0u32 { hd[b as usize] = dq; }
        let mut word = 0u32;
        let mut sm = 0i32;
        #[unroll]
        for j in 0..4u32 {
            let v = i32::cast_from(xround_g(hv[(tq * 4 + j) as usize] * id));
            sm += v;
            word |= (u32::cast_from(v) & 255u32) << (8 * j);
        }
        hq[(b * 16 + tq) as usize] = word;
        si[tq as usize] = sm;
    }
    sync_cube();
    if act && UNIT_POS == 0u32 {
        let b = e * (rows / 64u32) + CUBE_POS_X;
        let mut t = 0i32;
        for k in 0..16u32 { t += si[k as usize]; }
        hs[b as usize] = u32::cast_from(t);
    }
}
/// k_xdown_ex16 with the cube's 16 weight rows, the entry's hq words and hd/hs staged in shared memory by coalesced
/// loads (all 256 threads, consecutive words) first; then the identical per-lane integer sums and fma order.
#[cube(launch_unchecked)]
fn k_xdown_ex16s(hq: &[u32], hd: &[f32], hs: &[u32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], y: &mut [f32], rows: u32, nb: u32, rbd: u32) {
    let mut fa = Shared::<[f32]>::new_slice(256usize);
    let mut ws = Shared::<[u32]>::new_slice(5120usize);
    let mut xs = Shared::<[u32]>::new_slice(1024usize);
    let mut ds = Shared::<[f32]>::new_slice(64usize);
    let mut ss = Shared::<[u32]>::new_slice(64usize);
    let e = CUBE_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let half = ux / 16u32;
    let r = ux % 16u32;
    let lr = wy * 2u32 + half;
    let row = CUBE_POS_X * 16u32 + lr;
    let ein = e < oc[3];
    let rw = nb * 5u32;
    if ein {
        let m2 = meta[(e * 4 + 2) as usize];
        let sel = m2 >> 31;
        let b0 = (m2 & 2147483647u32) + CUBE_POS_X * 16u32 * rbd;
        let mut k = UNIT_POS;
        while k < 16u32 * rw {
            let rr = k / rw;
            let c = k - rr * rw;
            if CUBE_POS_X * 16u32 + rr < rows {
                if sel == 0u32 { ws[k as usize] = ar[(b0 + rr * rbd + c) as usize]; } else { ws[k as usize] = ar1[(b0 + rr * rbd + c) as usize]; }
            }
            k += CUBE_DIM;
        }
        let mut k2 = UNIT_POS;
        while k2 < nb * 16u32 { xs[k2 as usize] = hq[(e * nb * 16 + k2) as usize]; k2 += CUBE_DIM; }
        let mut k3 = UNIT_POS;
        while k3 < nb { ds[k3 as usize] = hd[(e * nb + k3) as usize]; ss[k3 as usize] = hs[(e * nb + k3) as usize]; k3 += CUBE_DIM; }
    }
    sync_cube();
    let act = ein && row < rows;
    if act && r < 9u32 {
        let k0 = (r % 8u32) / 4u32;
        let j0 = (r % 4u32) * 4u32;
        let mut acc = 0.0f32;
        for i in 0..nb {
            let blk = lr * rw + i * 5;
            let w0 = f32::reinterpret(ws[blk as usize]);
            let wq = ws[(blk + 1 + j0 / 4) as usize];
            let dd = w0 * ds[i as usize];
            if r < 8u32 {
                let mut p = 0i32;
                for jj in 0..4u32 {
                    let j = j0 + jj;
                    let qv = (wq >> (8 * (j % 4))) & 255u32;
                    let xw = xs[(i * 16 + j) as usize];
                    let pk = (((qv >> (2 * k0)) & 3u32) << (8 * k0)) | (((qv >> (2 * k0 + 4)) & 3u32) << (8 * k0 + 16));
                    p += dp4a(i32::cast_from(pk), i32::cast_from(xw), 0i32);
                }
                acc = fma(f32::cast_from(p), dd, acc);
            } else {
                let pr = fma(dd, f32::cast_from(i32::cast_from(ss[i as usize])), 0.0f32);
                acc = acc - pr;
            }
        }
        fa[(wy * 32 + ux) as usize] = acc;
    }
    sync_cube();
    if act && r == 0u32 {
        let f = wy * 32 + half * 16;
        let hsum = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let v = fa[(f + 8) as usize] + hsum;
        let wt = f32::reinterpret(meta[(e * 4 + 3) as usize]);
        y[(e * rows + row) as usize] = fma(wt, v, 0.0f32);
    }
}
#[cube(launch_unchecked)]
fn k_xdown_ex8(hq: &[u32], hd: &[f32], hs: &[u32], ar: &[u32], ar1: &[u32], meta: &[u32], oc: &[u32], y: &mut [f32], rows: u32, nb: u32, rbd: u32) {
    let mut pi = Shared::<[i32]>::new_slice(4096usize);
    let mut fa = Shared::<[f32]>::new_slice(72usize);
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let wy = UNIT_POS_Y;
    let ux = UNIT_POS_X;
    let act = e < oc[3] && row < rows;
    let mut dof = 0u32;
    let mut sel = 0u32;
    if act {
        let m2 = meta[(e * 4 + 2) as usize];
        sel = m2 >> 31;
        dof = (m2 & 2147483647u32) + row * rbd;
        let m = ux % 8;
        let mut i = ux / 8;
        let k0 = m / 4;
        let j0 = (m % 4) * 4;
        while i < nb {
            let blk = dof + i * 5;
            let mut p = 0i32;
            let mut wq = 0u32;
            if sel == 0u32 { wq = ar[(blk + 1 + j0 / 4) as usize]; } else { wq = ar1[(blk + 1 + j0 / 4) as usize]; }
            for jj in 0..4u32 {
                let j = j0 + jj;
                let qv = (wq >> (8 * (j % 4))) & 255u32;
                let xw = hq[(e * nb * 16 + i * 16 + j) as usize];
                let pk = (((qv >> (2 * k0)) & 3u32) << (8 * k0)) | (((qv >> (2 * k0 + 4)) & 3u32) << (8 * k0 + 16));
                p += dp4a(i32::cast_from(pk), i32::cast_from(xw), 0i32);
            }
            pi[(wy * 512 + i * 8 + m) as usize] = p;
            i += 4u32;
        }
    }
    sync_cube();
    if act && ux < 9 {
        let mut acc = 0.0f32;
        for i in 0..nb {
            let mut w0 = 0.0f32;
            if sel == 0u32 { w0 = f32::reinterpret(ar[(dof + i * 5) as usize]); } else { w0 = f32::reinterpret(ar1[(dof + i * 5) as usize]); }
            let dd = w0 * hd[(e * nb + i) as usize];
            if ux < 8 {
                acc = fma(f32::cast_from(pi[(wy * 512 + i * 8 + ux) as usize]), dd, acc);
            } else {
                let pr = fma(dd, f32::cast_from(i32::cast_from(hs[(e * nb + i) as usize])), 0.0f32);
                acc = acc - pr;
            }
        }
        fa[(wy * 9 + ux) as usize] = acc;
    }
    sync_cube();
    if act && ux == 0 {
        let f = wy * 9;
        let hsum = ((fa[(f) as usize] + fa[(f + 4) as usize]) + (fa[(f + 2) as usize] + fa[(f + 6) as usize])) + ((fa[(f + 1) as usize] + fa[(f + 5) as usize]) + (fa[(f + 3) as usize] + fa[(f + 7) as usize]));
        let v = fa[(f + 8) as usize] + hsum;
        let wt = f32::reinterpret(meta[(e * 4 + 3) as usize]);
        y[(e * rows + row) as usize] = fma(wt, v, 0.0f32);
    }
}
/// PYRE_QN_XGR: routing on the GPU right after the router GEMV. One cube (unit = expert, <= 1024), tokens in turn:
/// rank(e) = #{j: r_j > r_e} + #{j < e: r_j == r_e}; selected iff rank < k. That is the host top_k's set whenever the
/// k-th and (k+1)-th logits differ (the host checks this per token and falls back otherwise). The selected experts that
/// are resident (rt[e*4+3] != 0, rt = the host's xmeta words of the layer's cache table) become hit entries in ascending
/// expert order (weight word 1.0; the routing weight is applied later by k_xgr_scale with the host's softmax), oc =
/// cumulative hit counts per token (padded to 4), dbg = the selected ids in ascending order (PYRE_QN_XGR_CHK).
#[cube(launch_unchecked)]
fn k_xgr_topk(rd: &[f32], rt: &[u32], mh: &mut [u32], oc: &mut [u32], dbg: &mut [u32], rn: u32, d: u32, ne: u32, kk: u32, t: u32) {
    let mut lg = Shared::<[f32]>::new_slice(1024usize);
    let mut hf = Shared::<[u32]>::new_slice(1024usize);
    let mut sf = Shared::<[u32]>::new_slice(1024usize);
    let e = UNIT_POS;
    let mut base = 0u32;
    let mut ti = 0u32;
    while ti < t {
        if e < ne { lg[e as usize] = rd[(ti * rn + d + e) as usize]; }
        sync_cube();
        let mut sel = 0u32;
        let mut hit = 0u32;
        if e < ne {
            let v = lg[e as usize];
            let mut rank = 0u32;
            for j in 0..ne { let u = lg[j as usize]; if u > v { rank += 1u32; } else if u == v && j < e { rank += 1u32; } }
            if rank < kk { sel = 1u32; if rt[(e * 4 + 3) as usize] != 0u32 { hit = 1u32; } }
            hf[e as usize] = hit;
            sf[e as usize] = sel;
        }
        sync_cube();
        let mut tot = 0u32;
        if e < ne {
            let mut hp = 0u32;
            let mut sp = 0u32;
            for j in 0..ne { if j < e { hp += hf[j as usize]; sp += sf[j as usize]; } tot += hf[j as usize]; }
            if hit == 1u32 { let o = (base + hp) * 4; mh[o as usize] = rt[(e * 4) as usize]; mh[(o + 1) as usize] = rt[(e * 4 + 1) as usize]; mh[(o + 2) as usize] = rt[(e * 4 + 2) as usize]; mh[(o + 3) as usize] = 1065353216u32; }
            if sel == 1u32 { dbg[(ti * kk + sp) as usize] = e; }
        }
        base += tot;
        if e == 0u32 { oc[ti as usize] = base; }
        sync_cube();
        ti += 1u32;
    }
    if e == 0u32 { let mut j = t; while j < 4u32 { oc[j as usize] = base; j += 1u32; } }
}
/// PYRE_QN_XGR: the hit rows (written by k_xdown_ex16 with weight 1.0, i.e. fma(1, v, 0)) scaled by the host's routing
/// weights with the same single fma as the weighted down kernel: y = fma(w, v, 0).
#[cube(launch_unchecked)]
fn k_xgr_scale(y: &mut [f32], w: &[u32], n: u32, d: u32) {
    let i = ABSOLUTE_POS_X;
    if i < n { let wt = f32::reinterpret(w[(i / d) as usize]); y[i as usize] = fma(wt, y[i as usize], 0.0f32); }
}
/// PYRE_QN_XEXACT: per token, its routed experts' weighted rows summed in routing order from 0 (as the CPU's
/// per-token accumulation), each row from the GPU entry rows (gy) or the CPU rows (cy, code bit 31).
#[cube(launch_unchecked)]
fn k_rsum_ex(gy: &[f32], cy: &[f32], tab: &[u32], y: &mut [f32], d: u32, kk: u32) {
    let r = ABSOLUTE_POS_X;
    let ti = CUBE_POS_Y;
    if r < d {
        let b = ti * (kk + 1);
        let n = tab[b as usize];
        let mut acc = 0.0f32;
        for k in 0..n {
            let cd = tab[(b + 1 + k) as usize];
            let ix = cd & 2147483647u32;
            if (cd >> 31) == 1u32 { acc = acc + cy[(ix * d + r) as usize]; } else { acc = acc + gy[(ix * d + r) as usize]; }
        }
        y[(ti * d + r) as usize] = acc;
    }
}
/// PYRE_QN_XFUSE: k_rsum_ex + k_comb3_m (gate on, b = the routed sum, no c) in one launch: each element recomputes its
/// token's routed sum (same rows, same order from 0) and applies k_comb3_m's arithmetic, so the result is bit-identical.
#[cube(launch_unchecked)]
fn k_rsc_m(x0: &mut [f32], x1: &mut [f32], x2: &mut [f32], x3: &mut [f32], gy: &[f32], cy: &[f32], tab: &[u32], a: &[f32], gb: &[f32], dn: &[f32],
    d: u32, kk: u32, gs: u32, goff: u32, ds: u32, lr: u32, #[comptime] hc: u32) {
    let k = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let ti = CUBE_POS_Y;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let b0 = ti * (kk + 1);
        let n = tab[b0 as usize];
        let mut acc = 0.0f32;
        for j in 0..n {
            let cd = tab[(b0 + 1 + j) as usize];
            let ix = cd & 2147483647u32;
            if (cd >> 31) == 1u32 { acc = acc + cy[(ix * d + i) as usize]; } else { acc = acc + gy[(ix * d + i) as usize]; }
        }
        let mut o = a[(ti * d + i) as usize];
        o = o / (1.0f32 + f32::exp(-gb[(ti * gs + goff) as usize]));
        o += acc;
        let inj = dn[(ti * ds + lr + st) as usize] / f32::cast_from(hc);
        let v = o * 2.0f32 / (1.0f32 + f32::exp(-inj));
        if ti == 0u32 { x0[k as usize] += v; } else if ti == 1u32 { x1[k as usize] += v; } else if ti == 2u32 { x2[k as usize] += v; } else { x3[k as usize] += v; }
    }
}
/// PYRE_QN_MOEGRAPH: k_xgu_st / k_xdown_qt with the cumulative entry counts o1..o4 read from `oc` (device) instead of
/// launch scalars and a grid sized for the window's maximum entries; entries e >= o4 are padding and skipped, so one
/// captured launch serves every routing. Same per-entry/per-row arithmetic as the originals.
#[cube(launch_unchecked)]
fn k_xgu_stg(xq: &[u32], xs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], gsrc: &[u32], h: &mut [f32], rows: u32, rbg: u32, rbu: u32, oc: &[u32], dq: u32,
    #[comptime] ng: u32, #[comptime] gn: u32, #[comptime] ty: u32) {
    let mut grid = Shared::<[u32]>::new_slice(comptime!(gn as usize));
    let mut i = UNIT_POS;
    while i < gn { grid[i as usize] = gsrc[i as usize]; i += CUBE_DIM; }
    sync_cube();
    let e = CUBE_POS_Y;
    let lane = UNIT_POS_X;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let o1 = oc[0];
    let o2 = oc[1];
    let o3 = oc[2];
    let o4 = oc[3];
    let mut ti = 0u32;
    if e >= o1 { ti += 1u32; }
    if e >= o2 { ti += 1u32; }
    if e >= o3 { ti += 1u32; }
    let xo = ti * dq;
    let so = ti * (dq / 8);
    if row < rows && e < o4 {
        let m0 = meta[(e * 4) as usize];
        let sel = m0 >> 31;
        let go = (m0 & 2147483647u32) + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        #[unroll]
        for k in 0..ng {
            let q = lane + 32 * k;
            let x0 = i32::cast_from(xq[(xo + q * 2) as usize]);
            let x1 = i32::cast_from(xq[(xo + q * 2 + 1) as usize]);
            let xsc = xs[(so + q / 4) as usize];
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

#[cube(launch_unchecked)]
fn k_xdown_qtg(hq: &[u32], hs: &[f32], ar: &[u32], ar1: &[u32], meta: &[u32], q2l: &[u32], y: &mut [f32], rows: u32, cols: u32, rbd: u32, oc: &[u32]) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let ti = CUBE_POS_Y;
    let o1 = oc[0];
    let o2 = oc[1];
    let o3 = oc[2];
    let o4 = oc[3];
    let mut e0 = 0u32;
    let mut e1 = o1;
    if ti == 1u32 { e0 = o1; e1 = o2; } else if ti == 2u32 { e0 = o2; e1 = o3; } else if ti == 3u32 { e0 = o3; e1 = o4; }
    if row < rows {
        let nsb = cols / 8;
        let tot = (e1 - e0) * nsb;
        let mut s = 0.0f32;
        let mut j = UNIT_POS_X;
        while j < tot {
            let el = j / nsb;
            let q = j - el * nsb;
            let e = e0 + el;
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
            y[(ti * rows + row) as usize] = t;
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
/// PYRE_QN_ATTN2 (default on under the NATIVE=2 / XEXACT profile, =0 off; the Q8 default path keeps k_attn): k_attn with
/// 8 warps per head instead of one. Scores: one position per warp at a time, lanes over the head dims (coalesced K
/// rows, plane_sum); softmax max/sum over the cube; P.V: warps over positions, lanes over dims, partials summed in
/// shared memory. k_attn ran 24 one-warp cubes for ~84 us per token at T=3 (nsys e24). hd <= 512.
#[cube(launch_unchecked)]
fn k_attn2(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n: u32, cap: u32, scale: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let wy = UNIT_POS_Y;
    let nw = CUBE_DIM_Y;
    let tid = wy * 32u32 + lane;
    let kvh = h / (nh / nkv);
    let mut red = Shared::<[f32]>::new_slice(32usize);
    let mut part = Shared::<[f32]>::new_slice(4096usize);
    let mut mx = scale * -3.0e30f32;
    let mut pp = wy;
    while pp < n {
        let kb = (pp * nkv + kvh) * hd;
        let mut dd = 0.0f32;
        let mut i = lane;
        while i < hd { dd += q[(h * hd + i) as usize] * kc[(kb + i) as usize]; i += 32u32; }
        let v = plane_sum(dd) * scale;
        if lane == 0u32 { sc[(h * cap + pp) as usize] = v; }
        mx = f32::max(mx, v);
        pp += nw;
    }
    if lane == 0u32 { red[wy as usize] = mx; }
    sync_cube();
    let mut m = red[0];
    for w in 1..nw { m = f32::max(m, red[w as usize]); }
    let mut sum = 0.0f32;
    pp = tid;
    while pp < n {
        let e = f32::exp(sc[(h * cap + pp) as usize] - m);
        sc[(h * cap + pp) as usize] = e;
        sum += e;
        pp += nw * 32u32;
    }
    let ws = plane_sum(sum);
    if lane == 0u32 { red[(16u32 + wy) as usize] = ws; }
    sync_cube();
    let mut tot = 0.0f32;
    for w in 0..nw { tot += red[(16u32 + w) as usize]; }
    let inv = 1.0f32 / tot;
    let mut i = lane;
    while i < hd {
        let mut acc = 0.0f32;
        let mut t = wy;
        while t < n { acc += sc[(h * cap + t) as usize] * vc[((t * nkv + kvh) * hd + i) as usize]; t += nw; }
        part[(wy * hd + i) as usize] = acc;
        i += 32u32;
    }
    sync_cube();
    let mut j = tid;
    while j < hd {
        let mut a = 0.0f32;
        for w in 0..nw { a += part[(w * hd + j) as usize]; }
        let g = p[(h * 2 * hd + hd + j) as usize];
        o[(h * hd + j) as usize] = a * inv * (1.0f32 / (1.0f32 + f32::exp(-g)));
        j += nw * 32u32;
    }
}
/// PYRE_QN_ATTN3 (default on wherever k_attn2 runs, =0 back to k_attn2; t4: C T=3 mixer 121 -> 78 us, bit-identical): k_attn2 with more independent work in flight per warp, same bits: the score loop takes
/// two positions per step (independent dots, each still lanes-over-dims + plane_sum) and P.V keeps hd/32 accumulators
/// (one per lane dim) with positions in the outer loop, so each (warp, dim) partial is the same ascending-t chain
/// t = wy, wy+8, ... as k_attn2; exp/sum/merge code unchanged. Launch shape must stay (32, 8). nd = hd/32 (comptime).
#[cube(launch_unchecked)]
fn k_attn3(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n: u32, cap: u32, scale: f32, #[comptime] nd: u32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let wy = UNIT_POS_Y;
    let nw = CUBE_DIM_Y;
    let tid = wy * 32u32 + lane;
    let kvh = h / (nh / nkv);
    let mut red = Shared::<[f32]>::new_slice(32usize);
    let mut part = Shared::<[f32]>::new_slice(4096usize);
    let mut qr = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { qr[d as usize] = q[(h * hd + d * 32u32 + lane) as usize]; }
    let mut mx = scale * -3.0e30f32;
    let mut pp = wy;
    while pp + nw < n {
        let kb = (pp * nkv + kvh) * hd; let kb2 = ((pp + nw) * nkv + kvh) * hd;
        let mut dd = 0.0f32; let mut d2 = 0.0f32;
        #[unroll]
        for d in 0..nd { dd += qr[d as usize] * kc[(kb + d * 32u32 + lane) as usize]; d2 += qr[d as usize] * kc[(kb2 + d * 32u32 + lane) as usize]; }
        let v = plane_sum(dd) * scale; let v2 = plane_sum(d2) * scale;
        if lane == 0u32 { sc[(h * cap + pp) as usize] = v; sc[(h * cap + pp + nw) as usize] = v2; }
        mx = f32::max(mx, v); mx = f32::max(mx, v2);
        pp += 2u32 * nw;
    }
    if pp < n {
        let kb = (pp * nkv + kvh) * hd;
        let mut dd = 0.0f32;
        #[unroll]
        for d in 0..nd { dd += qr[d as usize] * kc[(kb + d * 32u32 + lane) as usize]; }
        let v = plane_sum(dd) * scale;
        if lane == 0u32 { sc[(h * cap + pp) as usize] = v; }
        mx = f32::max(mx, v);
    }
    if lane == 0u32 { red[wy as usize] = mx; }
    sync_cube();
    let mut m = red[0];
    for w in 1..nw { m = f32::max(m, red[w as usize]); }
    let mut sum = 0.0f32;
    pp = tid;
    while pp < n {
        let e = f32::exp(sc[(h * cap + pp) as usize] - m);
        sc[(h * cap + pp) as usize] = e;
        sum += e;
        pp += nw * 32u32;
    }
    let ws = plane_sum(sum);
    if lane == 0u32 { red[(16u32 + wy) as usize] = ws; }
    sync_cube();
    let mut tot = 0.0f32;
    for w in 0..nw { tot += red[(16u32 + w) as usize]; }
    let inv = 1.0f32 / tot;
    let mut acc = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { acc[d as usize] = 0.0f32; }
    let mut t = wy;
    while t < n {
        let e = sc[(h * cap + t) as usize]; let vb = (t * nkv + kvh) * hd;
        #[unroll]
        for d in 0..nd { acc[d as usize] += e * vc[(vb + d * 32u32 + lane) as usize]; }
        t += nw;
    }
    #[unroll]
    for d in 0..nd { part[(wy * hd + d * 32u32 + lane) as usize] = acc[d as usize]; }
    sync_cube();
    let mut j = tid;
    while j < hd {
        let mut a = 0.0f32;
        for w in 0..nw { a += part[(w * hd + j) as usize]; }
        let g = p[(h * 2 * hd + hd + j) as usize];
        o[(h * hd + j) as usize] = a * inv * (1.0f32 / (1.0f32 + f32::exp(-g)));
        j += nw * 32u32;
    }
}
/// QSA sparse decode attention on the GPU (PYRE_QN_QSA_GPU, default on): k_attn3 with the positions of unselected
/// index blocks (block = pos / ir < nbs with sel[block] == 0) scored -inf, i.e. softmax over the selected blocks plus the tail.
#[cube(launch_unchecked)]
fn k_attn3m(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n: u32, cap: u32, scale: f32, sel: &[f32], ir: u32, nbs: u32, #[comptime] nd: u32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let wy = UNIT_POS_Y;
    let nw = CUBE_DIM_Y;
    let tid = wy * 32u32 + lane;
    let kvh = h / (nh / nkv);
    let mut red = Shared::<[f32]>::new_slice(32usize);
    let mut part = Shared::<[f32]>::new_slice(4096usize);
    let mut qr = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { qr[d as usize] = q[(h * hd + d * 32u32 + lane) as usize]; }
    // block-structured: warps take whole index blocks (tail blocks >= nbs are always active); only the selected blocks'
    // positions are scored, exponentiated and summed into P.V, so the cost follows the QSA budget, not the context
    let nbt = (n + ir - 1u32) / ir;
    let mut mx = scale * -3.0e30f32;
    let mut b = wy;
    while b < nbt {
        let mut on = 1.0f32;
        if b < nbs { on = sel[b as usize]; }
        if on != 0.0f32 {
            let mut pe = (b + 1u32) * ir; if pe > n { pe = n; }
            let mut pp = b * ir;
            while pp < pe {
                let kb = (pp * nkv + kvh) * hd;
                let mut dd = 0.0f32;
                #[unroll]
                for d in 0..nd { dd += qr[d as usize] * kc[(kb + d * 32u32 + lane) as usize]; }
                let v = plane_sum(dd) * scale;
                if lane == 0u32 { sc[(h * cap + pp) as usize] = v; }
                mx = f32::max(mx, v);
                pp += 1u32;
            }
        }
        b += nw;
    }
    if lane == 0u32 { red[wy as usize] = mx; }
    sync_cube();
    let mut m = red[0];
    for w in 1..nw { m = f32::max(m, red[w as usize]); }
    let mut sum = 0.0f32;
    b = wy;
    while b < nbt {
        let mut on = 1.0f32;
        if b < nbs { on = sel[b as usize]; }
        if on != 0.0f32 {
            let mut pe = (b + 1u32) * ir; if pe > n { pe = n; }
            let mut pp = b * ir + lane;
            while pp < pe {
                let e = f32::exp(sc[(h * cap + pp) as usize] - m);
                sc[(h * cap + pp) as usize] = e;
                sum += e;
                pp += 32u32;
            }
        }
        b += nw;
    }
    let ws = plane_sum(sum);
    if lane == 0u32 { red[(16u32 + wy) as usize] = ws; }
    sync_cube();
    let mut tot = 0.0f32;
    for w in 0..nw { tot += red[(16u32 + w) as usize]; }
    let inv = 1.0f32 / tot;
    let mut acc = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { acc[d as usize] = 0.0f32; }
    b = wy;
    while b < nbt {
        let mut on = 1.0f32;
        if b < nbs { on = sel[b as usize]; }
        if on != 0.0f32 {
            let mut pe = (b + 1u32) * ir; if pe > n { pe = n; }
            let mut t = b * ir;
            while t < pe {
                let e = sc[(h * cap + t) as usize]; let vb = (t * nkv + kvh) * hd;
                #[unroll]
                for d in 0..nd { acc[d as usize] += e * vc[(vb + d * 32u32 + lane) as usize]; }
                t += 1u32;
            }
        }
        b += nw;
    }
    #[unroll]
    for d in 0..nd { part[(wy * hd + d * 32u32 + lane) as usize] = acc[d as usize]; }
    sync_cube();
    let mut j = tid;
    while j < hd {
        let mut a = 0.0f32;
        for w in 0..nw { a += part[(w * hd + j) as usize]; }
        let g = p[(h * 2 * hd + hd + j) as usize];
        o[(h * hd + j) as usize] = a * inv * (1.0f32 / (1.0f32 + f32::exp(-g)));
        j += nw * 32u32;
    }
}
/// QSA: pooled block key b = rope(rms(mean of raw index keys b*ir..(b+1)*ir) * ikn) at position b*ir (attn_core's pooling).
#[cube(launch_unchecked)]
fn k_qsa_pool(ik: &[f32], ikn: &[f32], cs: &[f32], ib: &mut [f32], b: u32, ir: u32, ihd: u32, half: u32, eps: f32) {
    let lane = UNIT_POS_X;
    let mut pk = Shared::<[f32]>::new_slice(512usize);
    let mut ss = 0.0f32;
    let mut i = lane;
    while i < ihd {
        let mut a = 0.0f32;
        let mut p = b * ir;
        while p < (b + 1u32) * ir { a += ik[(p * ihd + i) as usize]; p += 1u32; }
        let mv = a / f32::cast_from(ir);
        pk[i as usize] = mv; ss += mv * mv;
        i += 32u32;
    }
    sync_cube();
    let s = 1.0f32 / f32::sqrt(plane_sum(ss) / f32::cast_from(ihd) + eps);
    i = lane;
    while i < ihd {
        let mut j = i;
        if i < half { j = i + half; } else if i < 2 * half { j = i - half; }
        let ni = pk[i as usize] * (s * ikn[i as usize]);
        let nj = pk[j as usize] * (s * ikn[j as usize]);
        let mut o = ni;
        if i < half { o = ni * cs[i as usize] - nj * cs[(half + i) as usize]; } else if i < 2 * half { o = ni * cs[j as usize] + nj * cs[(half + j) as usize]; }
        ib[(b * ihd + i) as usize] = o;
        i += 32u32;
    }
}
/// QSA index query head h (cube): rope(rms(p[qo + h*ihd ..]) * iqn) at the token's position.
#[cube(launch_unchecked)]
fn k_qsa_q(p: &[f32], iqn: &[f32], cs: &[f32], iq: &mut [f32], qo: u32, ihd: u32, half: u32, eps: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let base = qo + h * ihd;
    let mut ss = 0.0f32;
    let mut i = lane;
    while i < ihd { let v = p[(base + i) as usize]; ss += v * v; i += 32u32; }
    let s = 1.0f32 / f32::sqrt(plane_sum(ss) / f32::cast_from(ihd) + eps);
    i = lane;
    while i < ihd {
        let mut j = i;
        if i < half { j = i + half; } else if i < 2 * half { j = i - half; }
        let ni = p[(base + i) as usize] * (s * iqn[i as usize]);
        let nj = p[(base + j) as usize] * (s * iqn[j as usize]);
        let mut o = ni;
        if i < half { o = ni * cs[i as usize] - nj * cs[(half + i) as usize]; } else if i < 2 * half { o = ni * cs[j as usize] + nj * cs[(half + j) as usize]; }
        iq[(h * ihd + i) as usize] = o;
        i += 32u32;
    }
}
/// QSA block score (cube = block): iscale * sum over index heads of relu(q_h . kb).
#[cube(launch_unchecked)]
fn k_qsa_score(iq: &[f32], ib: &[f32], sc: &mut [f32], inh: u32, ihd: u32, iscale: f32) {
    let b = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let mut s = 0.0f32;
    for h in 0..inh {
        let mut d = 0.0f32;
        let mut i = lane;
        while i < ihd { d += iq[(h * ihd + i) as usize] * ib[(b * ihd + i) as usize]; i += 32u32; }
        s += f32::max(plane_sum(d), 0.0f32);
    }
    if lane == 0u32 { sc[b as usize] = s * iscale; }
}
/// QSA top-kk blocks: rank = #{s_j > s_b} + #{j < b, s_j == s_b}; sel[b] = rank < kk.
#[cube(launch_unchecked)]
fn k_qsa_sel(sc: &[f32], sel: &mut [f32], nb: u32, kk: u32) {
    let b = ABSOLUTE_POS_X;
    if b < nb {
        let sb = sc[b as usize];
        let mut r = 0u32;
        for j in 0..nb {
            let sj = sc[j as usize];
            if sj > sb { r += 1u32; } else if sj == sb && j < b { r += 1u32; }
        }
        let mut v = 0.0f32;
        if r < kk { v = 1.0f32; }
        sel[b as usize] = v;
    }
}
/// Batched prefill copies of k_attn_prep / k_attn3: token tq = CUBE_POS_Y at position pos0 + tq (per-token offsets pst/qst/cst/ost/sst); per token the same arithmetic as the one-token kernels.
#[cube(launch_unchecked)]
fn k_attn_prep_b(p: &[f32], qn: &[f32], kn: &[f32], cs: &[f32], q: &mut [f32], kc: &mut [f32], vc: &mut [f32], ko: u32, vo: u32, nh: u32, nkv: u32, hd: u32, half: u32, pos0: u32, eps: f32, pst: u32, qst: u32, cst: u32) {
    let h = CUBE_POS_X;
    let tq = CUBE_POS_Y;
    let pos = pos0 + tq;
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
        let v = p[(tq * pst + base + i) as usize];
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
        let ni = p[(tq * pst + base + i) as usize] * (s * wi);
        let nj = p[(tq * pst + base + j) as usize] * (s * wj);
        let mut o = ni;
        if i < half {
            o = ni * cs[(tq * cst + i) as usize] - nj * cs[(tq * cst + half + i) as usize];
        } else if i < 2 * half {
            o = ni * cs[(tq * cst + j) as usize] + nj * cs[(tq * cst + half + j) as usize];
        }
        if h < nh {
            q[(tq * qst + dst + i) as usize] = o;
        } else {
            kc[(dst + i) as usize] = o;
            vc[(dst + i) as usize] = p[(tq * pst + vo + (h - nh) * hd + i) as usize];
        }
        i += 32u32;
    }
}
#[cube(launch_unchecked)]
fn k_attn3_b(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n0: u32, cap: u32, scale: f32, #[comptime] nd: u32, pst: u32, qst: u32, ost: u32, sst: u32) {
    let h = CUBE_POS_X;
    let tq = CUBE_POS_Y;
    let n = n0 + tq;
    let lane = UNIT_POS_X;
    let wy = UNIT_POS_Y;
    let nw = CUBE_DIM_Y;
    let tid = wy * 32u32 + lane;
    let kvh = h / (nh / nkv);
    let mut red = Shared::<[f32]>::new_slice(32usize);
    let mut part = Shared::<[f32]>::new_slice(4096usize);
    let mut qr = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { qr[d as usize] = q[(tq * qst + h * hd + d * 32u32 + lane) as usize]; }
    let mut mx = scale * -3.0e30f32;
    let mut pp = wy;
    while pp + nw < n {
        let kb = (pp * nkv + kvh) * hd; let kb2 = ((pp + nw) * nkv + kvh) * hd;
        let mut dd = 0.0f32; let mut d2 = 0.0f32;
        #[unroll]
        for d in 0..nd { dd += qr[d as usize] * kc[(kb + d * 32u32 + lane) as usize]; d2 += qr[d as usize] * kc[(kb2 + d * 32u32 + lane) as usize]; }
        let v = plane_sum(dd) * scale; let v2 = plane_sum(d2) * scale;
        if lane == 0u32 { sc[(tq * sst + h * cap + pp) as usize] = v; sc[(tq * sst + h * cap + pp + nw) as usize] = v2; }
        mx = f32::max(mx, v); mx = f32::max(mx, v2);
        pp += 2u32 * nw;
    }
    if pp < n {
        let kb = (pp * nkv + kvh) * hd;
        let mut dd = 0.0f32;
        #[unroll]
        for d in 0..nd { dd += qr[d as usize] * kc[(kb + d * 32u32 + lane) as usize]; }
        let v = plane_sum(dd) * scale;
        if lane == 0u32 { sc[(tq * sst + h * cap + pp) as usize] = v; }
        mx = f32::max(mx, v);
    }
    if lane == 0u32 { red[wy as usize] = mx; }
    sync_cube();
    let mut m = red[0];
    for w in 1..nw { m = f32::max(m, red[w as usize]); }
    let mut sum = 0.0f32;
    pp = tid;
    while pp < n {
        let e = f32::exp(sc[(tq * sst + h * cap + pp) as usize] - m);
        sc[(tq * sst + h * cap + pp) as usize] = e;
        sum += e;
        pp += nw * 32u32;
    }
    let ws = plane_sum(sum);
    if lane == 0u32 { red[(16u32 + wy) as usize] = ws; }
    sync_cube();
    let mut tot = 0.0f32;
    for w in 0..nw { tot += red[(16u32 + w) as usize]; }
    let inv = 1.0f32 / tot;
    let mut acc = Array::<f32>::new(nd as usize);
    #[unroll]
    for d in 0..nd { acc[d as usize] = 0.0f32; }
    let mut t = wy;
    while t < n {
        let e = sc[(tq * sst + h * cap + t) as usize]; let vb = (t * nkv + kvh) * hd;
        #[unroll]
        for d in 0..nd { acc[d as usize] += e * vc[(vb + d * 32u32 + lane) as usize]; }
        t += nw;
    }
    #[unroll]
    for d in 0..nd { part[(wy * hd + d * 32u32 + lane) as usize] = acc[d as usize]; }
    sync_cube();
    let mut j = tid;
    while j < hd {
        let mut a = 0.0f32;
        for w in 0..nw { a += part[(w * hd + j) as usize]; }
        let g = p[(tq * pst + h * 2 * hd + hd + j) as usize];
        o[(tq * ost + h * hd + j) as usize] = a * inv * (1.0f32 / (1.0f32 + f32::exp(-g)));
        j += nw * 32u32;
    }
}
fn attn3() -> bool { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *V.get_or_init(|| std::env::var("PYRE_QN_ATTN3").map(|v| v != "0").unwrap_or(true)) }
fn attn2() -> bool { static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *V.get_or_init(|| std::env::var("PYRE_QN_ATTN2").map(|v| v != "0").unwrap_or(native2())) }
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
        crate::limits::vram_watch();
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
    pub fn empty(&self, n: usize) -> Handle { crate::limits::vram_check((n * 4) as u64); self.client.empty(n * 4) }
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
        crate::limits::vram_check(b.len() as u64);
        if b.len() >= (16 << 20) && xpersist() { return self.client.memory_persistent_allocation((), |_| self.client.create_from_slice(b)); } // PYRE_QN_XARENA_PERSIST: big uploads (weights) persistent, see qn_gpu::upload
        if hpin() && b.len() <= (1 << 20) { let mut by = cubecl::bytes::Bytes::from_bytes_vec(b.to_vec()); self.client.staging(std::iter::once(&mut by), false); return self.client.create(by); }
        self.client.create_from_slice(b)
    }
    /// Start a readback now (its fence follows the work enqueued so far) and finish it later with `read_finish`,
    /// so work enqueued in between does not delay it (the verify wavefront).
    pub fn read_start(&self, h: &Handle) -> impl std::future::Future<Output = Result<Vec<burn_cubecl::cubecl::bytes::Bytes>, burn_cubecl::cubecl::server::ServerError>> + Send + '_ { let f: std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<burn_cubecl::cubecl::bytes::Bytes>, burn_cubecl::cubecl::server::ServerError>> + Send + '_>> = if readnb() { self.client.read_async_nb(vec![h.clone()]) } else { Box::pin(self.client.read_async(vec![h.clone()])) }; f }
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
    /// lossy draft-head GEMV (k_gemv_q8d): plain Q8 CMat only (no native/qseg segments); false = not eligible
    pub fn gemv_q8d(&self, m: &CMat, x: &Handle, out: &Handle) -> bool {
        if !m.nat.is_empty() || !m.qseg.is_empty() || m.cols % 32 != 0 { return false; }
        let nb = m.cols / 32; let xq = self.empty(m.cols / 4); let xs = self.empty(nb);
        unsafe { k_q8pack::launch_unchecked(&self.client, CubeCount::Static((nb as u32).div_ceil(64), 1, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(x.clone(), m.cols), BufferArg::from_raw_parts(xq.clone(), m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nb), 0u32, nb as u32) }
        unsafe { k_gemv_q8d::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
            BufferArg::from_raw_parts(xq, m.cols / 4), BufferArg::from_raw_parts(xs, nb), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
            BufferArg::from_raw_parts(out.clone(), m.rows), m.rows as u32, m.cols as u32) }
        true
    }    /// k_best into o[0..2] (no readback; PYRE_MTP_GX chained drafts)
    pub fn best_into(&self, lg: &Handle, n: usize, o: &Handle) { unsafe { k_best::launch_unchecked(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1024), BufferArg::from_raw_parts(lg.clone(), n), BufferArg::from_raw_parts(o.clone(), 2), n as u32) } }
    /// mtp::best on the GPU: (argmax, 1/sum exp(v-max)) of lg[0..n] with an 8-byte readback
    pub fn best(&self, lg: &Handle, n: usize) -> (usize, f32) {
        let o = self.empty(2);
        unsafe { k_best::launch_unchecked(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1024), BufferArg::from_raw_parts(lg.clone(), n), BufferArg::from_raw_parts(o.clone(), 2), n as u32) }
        let r = self.read(&o); (r[0] as usize, r[1])
    }    /// out[ooff..ooff+rows] = W x[xoff..xoff+cols]
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
            if natx() { for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4s_m::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, 0u32, 1u32) } } } else {
            for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4d::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32) } } }
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
        if !m.nat.is_empty() && native_mode() == 2 && ((2..=4).contains(&nt) && natm() || natx() && (1..=4).contains(&nt)) {
            let nb = m.cols / 32; let xq = self.empty(nt * m.cols / 4); let xs = self.empty(nt * nb);
            if natx() { unsafe { k_q8pack_mx::launch_unchecked(&self.client, CubeCount::Static((nb as u32).div_ceil(64), nt as u32, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), xoff as u32, xst as u32, nb as u32) } } else { unsafe { k_q8pack_m::launch_unchecked(&self.client, CubeCount::Static((nb as u32).div_ceil(64), nt as u32, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), xoff as u32, xst as u32, nb as u32) } }
            if natx() { for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4s_mx::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32, natx_fma()) } } } else if nats_on() || natx() { for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4s_m::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } } } else {
            for (r0, n, w) in &m.nat { if natr() && NATS_OV.load(std::sync::atomic::Ordering::Relaxed) != 1 && *n >= natr_min() { if natr3() > 0 { let rr = natr3(); unsafe { k_gemv_iq4d_r3::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(natr_y() * rr), 1, 1), CubeDim::new_2d(32, natr_y()),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32, rr, natr3_pl() || natr_prmt()) } continue; }
                if natr_u() > 1 { unsafe { k_gemv_iq4d_r2::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(natr_y() * 4), 1, 1), CubeDim::new_2d(32, natr_y()),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32, natr_u(), natr_prmt()) } continue; }
                unsafe { k_gemv_iq4d_r::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(natr_y() * 4), 1, 1), CubeDim::new_2d(32, natr_y()),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } continue; }
                unsafe { k_gemv_iq4d_m::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(xq.clone(), nt * m.cols / 4), BufferArg::from_raw_parts(xs.clone(), nt * nb), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 256),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, (ooff + r0) as u32, ost as u32, nt as u32) } } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); let rpc = gemv_rpc(*n);
                unsafe { k_gemv4r::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, xst as u32, (ooff + r0) as u32, ost as u32, nt as u32, nt > 1, nt > 2, nt > 3) } }
            if natx() { natchk(self, m, x, xoff, xst, &xq, &xs, out, ooff, ost, nt); }
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
        // PYRE_QN_GEMV4S=1: k_gemv4r with the activation columns staged in shared memory (same lanes, same words, same
        // per-lane and plane_sum order: bit-identical; only where x is read from changes)
        if gemv4s() && nt <= 4 && nt * m.cols <= GEMV4S_MAX as usize {
            let rpc = gemv_rpc(m.rows); unsafe { k_gemv4s::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, xst as u32, ooff as u32, ost as u32, nt as u32, nt > 1, nt > 2, nt > 3) }
            return;
        }
        if gemv8() && (5..=8).contains(&nt) {
            let rpc = gemv_rpc(m.rows); unsafe { k_gemv8r::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, xst as u32, ooff as u32, ost as u32, nt as u32, true, true, true, true, nt > 5, nt > 6, nt > 7) }
            return;
        }
        if std::env::var("PYRE_QN_GEMV4R").map(|v| v != "0").unwrap_or(true) {
            while c0 < nt { let n = (nt - c0).min(4);
                if gemv4rp() && m.rows < 1024 && (m.cols / 4) % 128 == 0 {
                    let rpc = gemv_rpc(m.rows); unsafe { k_gemv4rp::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(rpc * 4), 1, 1), CubeDim::new_2d(32, rpc),
                        BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                        BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32) }
                    c0 += n; continue;
                }
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
    /// Batched-prefill GEMM (k_gemm_pf) of `t` tokens: x [t][cols] -> out [t][rows], every native / Q8 segment of m,
    /// in launches of at most 128 tokens (each a few ms at most).
    pub fn gemm_pf(&self, m: &CMat, x: &Handle, xn: usize, t: usize, out: &Handle, on: usize) { self.gemm_pfx(m, x, xn, m.cols, t, out, on, 0, m.rows, false) }
    /// gemm_pf with token strides: x row j at j*xst, out row j at j*ost + obase
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_pfx(&self, m: &CMat, x: &Handle, xn: usize, xst: usize, t: usize, out: &Handle, on: usize, obase: usize, ostr: usize, f32only: bool) {
        let cols = m.cols; let kv: Vec<f32> = super::iq_tables::KVALUES_IQ4NL.iter().map(|&v| v as f32).collect(); let kvb = self.buf(&kv);
        let mut c0 = 0usize; let _ = (xn, on);
        // PYRE_QN_BPF_TC (default on; 0 = f32 k_gemm_pf): tensor-core f16 tiles
        let tc = !f32only && std::env::var("PYRE_QN_BPF_TC").map(|v| v != "0").unwrap_or(true);
        if std::env::var("PYRE_QN_BPF_BENCH").is_ok() && t >= 128 { eprintln!("gemm_pf {}x{} t={t} nat {:?} qseg {:?}", m.rows, cols, m.nat.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>(), m.qseg.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>()); }
        while c0 < t { let nt = (t - c0).min(256);
            let xo = x.clone().offset_start((c0 * xst * 4) as u64); let xl = (nt - 1) * xst + cols; let oo = out.clone().offset_start((c0 * ostr * 4) as u64); let ol = (nt - 1) * ostr + obase + m.rows;
            // split-K (per segment) so that small-row matrices (HC down: 324 rows, 64-row Q8 segments) still fill the SMs;
            // K ranges in whole 256 blocks; no split below 2048 columns. A split segment writes ks partial planes of its own
            // (ks x nt x n floats, small) that k_gemm_sum adds in plane order into out.
            let ksplit = |n: usize| -> (usize, usize) { if cols < 2048 { return (1, cols); } let cubes = n.div_ceil(if tc { 64 } else { 128 }) * nt.div_ceil(64); let ks0 = (112usize.div_ceil(cubes)).clamp(1, 8).min(cols / 256).max(1);
                let kch = (cols / 256).div_ceil(ks0) * 256; (cols.div_ceil(kch), kch) };
            let launch = |w: Handle, wl: usize, sh: Handle, sl: usize, r0: usize, n: usize, nat: bool| { let (ks, kch) = ksplit(n);
                let part = if ks > 1 { Some(self.empty(ks * nt * n)) } else { None };
                let (dst, dl, ooff, ost, pst) = match &part { Some(p) => (p.clone(), ks * nt * n, 0usize, n, nt * n), None => (oo.clone(), ol, obase + r0, ostr, 0) };
                if tc { unsafe { k_gemm_tc::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(64), (nt as u32).div_ceil(64), ks as u32), CubeDim::new_1d(128),
                BufferArg::from_raw_parts(xo.clone(), xl), BufferArg::from_raw_parts(w, wl), BufferArg::from_raw_parts(sh, sl), BufferArg::from_raw_parts(kvb.clone(), 16),
                BufferArg::from_raw_parts(dst, dl), n as u32, cols as u32, nt as u32, kch as u32, ooff as u32, ost as u32, pst as u32, xst as u32, nat) } } else {
                unsafe { k_gemm_pf::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(128), (nt as u32).div_ceil(64), ks as u32), CubeDim::new_1d(256),
                BufferArg::from_raw_parts(xo.clone(), xl), BufferArg::from_raw_parts(w, wl), BufferArg::from_raw_parts(sh, sl), BufferArg::from_raw_parts(kvb.clone(), 16),
                BufferArg::from_raw_parts(dst, dl), n as u32, cols as u32, nt as u32, kch as u32, ooff as u32, ost as u32, pst as u32, xst as u32, nat) } }
                if let Some(p) = &part { unsafe { k_gemm_sum::launch_unchecked(&self.client, CubeCount::Static(((nt * n) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256), BufferArg::from_raw_parts(p.clone(), ks * nt * n), BufferArg::from_raw_parts(oo.clone(), ol), n as u32, (nt * n) as u32, ks as u32, (obase + r0) as u32, ostr as u32) } } };
            if m.nat.is_empty() && m.qseg.is_empty() { launch(m.w.clone(), m.rows * cols / 4, m.s.clone(), m.rows * cols / 32, 0, m.rows, false); }
            for (r0, n, w) in &m.nat { launch(w.clone(), n * cols / 256 * 35, kvb.clone(), 16, *r0, *n, true); }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * cols) as u64, (qb * cols / 32 * 4) as u64); launch(m.w.clone().offset_start(wo), n * cols / 4, m.s.clone().offset_start(so), n * cols / 32, *r0, *n, false); }
            c0 += nt; }
    }
    /// rms of t tokens (separate x buffers) into xn at token stride hc*d; one launch for up to 4 (PYRE_QN_RMS4=0: per token)
    pub fn rms_t(&self, xs: &[Handle], norm: &Handle, xn: &Handle, hc: usize, d: usize, eps: f32) {
        if xs.len() > 4 && t8() && w8() { let t = xs.len(); let w = hc * d; let g = |i: usize| unsafe { BufferArg::from_raw_parts(xs[i.min(t - 1)].clone(), w) };
            unsafe { k_rms8::launch_unchecked(&self.client, CubeCount::Static(hc as u32, t as u32, 1), CubeDim::new_1d(256), g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7),
                BufferArg::from_raw_parts(norm.clone(), w), BufferArg::from_raw_parts(xn.clone(), t * w), d as u32, w as u32, eps) } return; }
        if xs.len() > 8 && super::qwen4exp_gguf::bpg() { let w = hc * d; for c0 in (0..xs.len()).step_by(4) { let c1 = (c0 + 4).min(xs.len()); self.rms_t(&xs[c0..c1], norm, &xn.clone().offset_start((c0 * w * 4) as u64), hc, d, eps); } return; }
        if xs.len() > 4 && t8() { let w = hc * d; for c0 in (0..xs.len()).step_by(4) { let c1 = (c0 + 4).min(xs.len()); self.rms_t(&xs[c0..c1], norm, &xn.clone().offset_start((c0 * w * 4) as u64), hc, d, eps); } return; }
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
        if t > 8 && super::qwen4exp_gguf::bpg() { for c0 in (0..t).step_by(4) { let n = (t - c0).min(4); if n == 1 { self.gate(&xn.clone().offset_start((c0 * xs * 4) as u64), &dn.clone().offset_start((c0 * ds * 4) as u64), up, out, on, ooff + c0 * os, hc, d); } else { self.gate_m(&xn.clone().offset_start((c0 * xs * 4) as u64), &dn.clone().offset_start((c0 * ds * 4) as u64), up, out, on, ooff + c0 * os, hc, d, n, xs, ds, os); } } return true; }
        if t > 4 && t <= 8 && t8() && !gate8() { for c0 in (0..t).step_by(4) { let n = (t - c0).min(4); if n == 1 { self.gate(&xn.clone().offset_start((c0 * xs * 4) as u64), &dn.clone().offset_start((c0 * ds * 4) as u64), up, out, on, ooff + c0 * os, hc, d); } else { self.gate_m(&xn.clone().offset_start((c0 * xs * 4) as u64), &dn.clone().offset_start((c0 * ds * 4) as u64), up, out, on, ooff + c0 * os, hc, d, n, xs, ds, os); } } return true; }
        if !((2..=4).contains(&t) || t <= 8 && t8() && gate8()) { return false; }
        let lr = up.cols;
        if gatem2() && 8 % hc == 0 && lr % 32 == 0 {
            unsafe { k_gate_m2::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(8 / hc as u32), 1, 1), CubeDim::new_2d(32, 8),
                BufferArg::from_raw_parts(xn.clone(), t * xs), BufferArg::from_raw_parts(dn.clone(), t * ds), BufferArg::from_raw_parts(up.w.clone(), up.rows * lr / 4),
                BufferArg::from_raw_parts(up.s.clone(), up.rows * lr / 32), BufferArg::from_raw_parts(out.clone(), on), d as u32, ooff as u32, xs as u32, ds as u32, os as u32, lr, hc as u32, t as u32) }
            return true;
        }
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
    /// k_comb for xs.len() <= 4 tokens in one launch: a = t*d contiguous, dn = t*ds contiguous
    pub fn comb_m(&self, xs: &[Handle], a: &Handle, dn: &Handle, ds: usize, lr: usize, hc: usize, d: usize) {
        if xs.len() > 4 && t8() && w8() { let t = xs.len(); let x = |i: usize| unsafe { BufferArg::from_raw_parts(xs[i.min(t - 1)].clone(), hc * d) };
            unsafe { k_comb_m8::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256), x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7),
                BufferArg::from_raw_parts(a.clone(), t * d), BufferArg::from_raw_parts(dn.clone(), t * ds), d as u32, ds as u32, lr as u32, hc as u32) } return; }
        if xs.len() > 4 { for c0 in (0..xs.len()).step_by(4) { let c1 = (c0 + 4).min(xs.len()); self.comb_m(&xs[c0..c1], &a.clone().offset_start((c0 * d * 4) as u64), &dn.clone().offset_start((c0 * ds * 4) as u64), ds, lr, hc, d); } return; }
        let t = xs.len(); let x = |i: usize| xs[i.min(t - 1)].clone();
        unsafe { k_comb_m::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x(0), hc * d), BufferArg::from_raw_parts(x(1), hc * d), BufferArg::from_raw_parts(x(2), hc * d), BufferArg::from_raw_parts(x(3), hc * d),
            BufferArg::from_raw_parts(a.clone(), t * d), BufferArg::from_raw_parts(dn.clone(), t * ds), d as u32, ds as u32, lr as u32, hc as u32) }
    }
    /// k_comb3 (gate, b, optional per-token c) for xs.len() <= 4 tokens in one launch
    #[allow(clippy::too_many_arguments)]
    pub fn comb3_m(&self, xs: &[Handle], cs: &[Option<Handle>], a: &Handle, b: &Handle, gb: &Handle, gn: usize, gs: usize, goff: usize, dn: &Handle, ds: usize, lr: usize, hc: usize, d: usize) {
        if xs.len() > 4 && t8() && w8() { let t = xs.len(); let x = |i: usize| unsafe { BufferArg::from_raw_parts(xs[i.min(t - 1)].clone(), hc * d) }; let mut cm = 0u32;
            for i in 0..t { if cs[i].is_some() { cm |= 1 << i; } }
            let c = |i: usize| unsafe { BufferArg::from_raw_parts(match cs.get(i) { Some(Some(h)) => h.clone(), _ => a.clone() }, d) };
            unsafe { k_comb3_m8::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256), x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7),
                c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7),
                BufferArg::from_raw_parts(a.clone(), t * d), BufferArg::from_raw_parts(b.clone(), t * d), BufferArg::from_raw_parts(gb.clone(), gn), BufferArg::from_raw_parts(dn.clone(), t * ds),
                d as u32, gs as u32, goff as u32, ds as u32, lr as u32, cm, hc as u32) } return; }
        if xs.len() > 4 { for c0 in (0..xs.len()).step_by(4) { let c1 = (c0 + 4).min(xs.len()); let o = |h: &Handle, n: usize| h.clone().offset_start((c0 * n * 4) as u64);
            self.comb3_m(&xs[c0..c1], &cs[c0..c1], &o(a, d), &o(b, d), &o(gb, gs), gn - c0 * gs, gs, goff, &o(dn, ds), ds, lr, hc, d); } return; }
        let t = xs.len(); let x = |i: usize| xs[i.min(t - 1)].clone(); let mut cm = 0u32;
        for i in 0..t { if cs[i].is_some() { cm |= 1 << i; } }
        let c = |i: usize| -> Handle { match cs.get(i) { Some(Some(h)) => h.clone(), _ => a.clone() } };
        unsafe { k_comb3_m::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x(0), hc * d), BufferArg::from_raw_parts(x(1), hc * d), BufferArg::from_raw_parts(x(2), hc * d), BufferArg::from_raw_parts(x(3), hc * d),
            BufferArg::from_raw_parts(c(0), d), BufferArg::from_raw_parts(c(1), d), BufferArg::from_raw_parts(c(2), d), BufferArg::from_raw_parts(c(3), d),
            BufferArg::from_raw_parts(a.clone(), t * d), BufferArg::from_raw_parts(b.clone(), t * d), BufferArg::from_raw_parts(gb.clone(), gn), BufferArg::from_raw_parts(dn.clone(), t * ds),
            d as u32, gs as u32, goff as u32, ds as u32, lr as u32, cm, hc as u32) }
    }
    pub fn comb(&self, x: &Handle, a: &Handle, b: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) { self.comb3(x, a, b, None, g, dn, lr, hc, d) }
    #[allow(clippy::too_many_arguments)]
    pub fn comb3(&self, x: &Handle, a: &Handle, b: Option<&Handle>, c3: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) {
        let (gb, gn, goff) = g.map(|(h, n, o)| (h.clone(), n, o)).unwrap_or((a.clone(), d, 0));
        let bb = b.cloned().unwrap_or_else(|| a.clone()); let cc = c3.cloned().unwrap_or_else(|| a.clone());
        unsafe { k_comb::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(a.clone(), d), BufferArg::from_raw_parts(bb, d), BufferArg::from_raw_parts(cc, d), BufferArg::from_raw_parts(gb, gn),
            BufferArg::from_raw_parts(dn.clone(), lr + hc), d as u32, goff as u32, lr as u32, hc as u32, g.is_some(), b.is_some(), c3.is_some()) }
    }
    /// t tokens' convs in one launch; cks[ti] = Some(checkpoint) for ti < 4 (state after token ti)
    pub fn conv_m(&self, proj: &Handle, pn: usize, pr: usize, w: &Handle, cs: &Handle, co: &Handle, ch: usize, kk: usize, t: usize, cks: &[Option<Handle>]) {
        if t > 4 { for c0 in (0..t).step_by(4) { let n = (t - c0).min(4); self.conv_m(&proj.clone().offset_start((c0 * pr * 4) as u64), pn - c0 * pr, pr, w, cs, &co.clone().offset_start((c0 * ch * 4) as u64), ch, kk, n, &cks[c0.min(cks.len())..(c0 + n).min(cks.len())]); } return; }
        let cn = ch * (kk - 1); let mut ckm = 0u32; let dummy = self.empty(cn);
        let ck = |i: usize| -> Handle { match cks.get(i) { Some(Some(h)) => h.clone(), _ => dummy.clone() } };
        for i in 0..t.min(4) { if matches!(cks.get(i), Some(Some(_))) { ckm |= 1 << i; } }
        unsafe { k_conv_m::launch_unchecked(&self.client, CubeCount::Static((ch as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(proj.clone(), pn), BufferArg::from_raw_parts(w.clone(), ch * kk), BufferArg::from_raw_parts(cs.clone(), cn),
            BufferArg::from_raw_parts(co.clone(), t * ch), BufferArg::from_raw_parts(ck(0), cn), BufferArg::from_raw_parts(ck(1), cn), BufferArg::from_raw_parts(ck(2), cn), BufferArg::from_raw_parts(ck(3), cn),
            ch as u32, pr as u32, t as u32, ckm, kk as u32) }
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
        if nt > 4 && nt <= 8 && t8() && w8() { let sn = nv * dk * dk; let dummy = self.empty(1);
            let ck = |i: usize| unsafe { match cks.get(i) { Some(Some(h)) if (ckm >> i) & 1 == 1 => BufferArg::from_raw_parts(h.clone(), sn), _ => BufferArg::from_raw_parts(dummy.clone(), 1) } };
            unsafe { k_delta_m8::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
                BufferArg::from_raw_parts(co.clone(), nt * ch), BufferArg::from_raw_parts(proj.clone(), nt * pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
                BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), sn), BufferArg::from_raw_parts(out.clone(), nt * ostr),
                ck(0), ck(1), ck(2), ck(3), ck(4), ck(5), ck(6), ck(7),
                nk as u32, ch as u32, pn as u32, ostr as u32, ckm, zoff as u32, boff as u32, aoff as u32, eps, dk as u32, nt as u32) } return; }
        if nt > 4 { for c0 in (0..nt).step_by(4) { let n = (nt - c0).min(4); self.delta_m(&co.clone().offset_start((c0 * ch * 4) as u64), ch, &proj.clone().offset_start((c0 * pn * 4) as u64), pn, w, s, &out.clone().offset_start((c0 * ostr * 4) as u64), ostr, &cks[c0.min(cks.len())..(c0 + n).min(cks.len())], (ckm >> c0) & 15, nk, nv, dk, zoff, boff, aoff, eps, n); } return; }
        let sn = nv * dk * dk; let dummy = self.empty(1);
        let ck = |i: usize| match cks.get(i) { Some(Some(h)) if (ckm >> i) & 1 == 1 => (h.clone(), sn), _ => (dummy.clone(), 1) };
        let (k0, k1, k2, k3) = (ck(0), ck(1), ck(2), ck(3));
        if delta2() {
        unsafe { k_delta_m2::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
            BufferArg::from_raw_parts(co.clone(), nt * ch), BufferArg::from_raw_parts(proj.clone(), nt * pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
            BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), sn), BufferArg::from_raw_parts(out.clone(), nt * ostr),
            BufferArg::from_raw_parts(k0.0, k0.1), BufferArg::from_raw_parts(k1.0, k1.1), BufferArg::from_raw_parts(k2.0, k2.1), BufferArg::from_raw_parts(k3.0, k3.1),
            nk as u32, ch as u32, pn as u32, ostr as u32, ckm, zoff as u32, boff as u32, aoff as u32, eps, dk as u32, nt as u32) }
        } else {
        unsafe { k_delta_m::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
            BufferArg::from_raw_parts(co.clone(), nt * ch), BufferArg::from_raw_parts(proj.clone(), nt * pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
            BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), sn), BufferArg::from_raw_parts(out.clone(), nt * ostr),
            BufferArg::from_raw_parts(k0.0, k0.1), BufferArg::from_raw_parts(k1.0, k1.1), BufferArg::from_raw_parts(k2.0, k2.1), BufferArg::from_raw_parts(k3.0, k3.1),
            nk as u32, ch as u32, pn as u32, ostr as u32, ckm, zoff as u32, boff as u32, aoff as u32, eps, dk as u32, nt as u32) }
        }
    }
    /// k_attn_prep_b + k_attn3_b for t tokens at positions pos0.. (K/V of all t written first; token j attends 0..=pos0+j).
    /// cs = t rope tables (2*half each); p = t projections (stride pst); out stride ost. false = not eligible (caller loops attn()).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_b(&self, p: &Handle, pn: usize, pst: usize, ko: usize, vo: usize, qn: &Handle, kn: &Handle, cs: &Handle, kc: &Handle, vc: &Handle, cap: usize, out: &Handle, ost: usize, nh: usize, nkv: usize, hd: usize, half: usize, pos0: usize, t: usize, eps: f32) -> bool {
        if !(attn2() && attn3() && hd <= 512 && hd % 32 == 0) || pos0 + t > cap { return false; }
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        let q = self.empty(t * nh * hd); let sc = self.empty(t * nh * cap);
        unsafe {
            k_attn_prep_b::launch_unchecked(&self.client, CubeCount::Static((nh + nkv) as u32, t as u32, 1), CubeDim::new_1d(32),
                a(p, pn), a(qn, hd), a(kn, hd), a(cs, t * 2 * half), a(&q, t * nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), ko as u32, vo as u32, nh as u32, nkv as u32, hd as u32, half as u32, pos0 as u32, eps, pst as u32, (nh * hd) as u32, (2 * half) as u32);
            k_attn3_b::launch_unchecked(&self.client, CubeCount::Static(nh as u32, t as u32, 1), CubeDim::new_2d(32, 8),
                a(&q, t * nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, t * nh * cap), a(out, t * ost), nh as u32, nkv as u32, hd as u32, (pos0 + 1) as u32, cap as u32, (hd as f32).sqrt().recip(), (hd / 32) as u32, pst as u32, (nh * hd) as u32, ost as u32, (nh * cap) as u32);
        }
        true
    }
    /// Decode attention for one token (see k_attn_prep / k_attn); `p` = the fused q|k|v projection.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(&self, p: &Handle, pn: usize, ko: usize, vo: usize, qn: &Handle, kn: &Handle, cs: &Handle, kc: &Handle, vc: &Handle, cap: usize, out: &Handle, nh: usize, nkv: usize, hd: usize, half: usize, pos: usize, eps: f32) {
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        let q = self.empty(nh * hd); let sc = self.empty(nh * cap);
        unsafe {
            k_attn_prep::launch_unchecked(&self.client, CubeCount::Static((nh + nkv) as u32, 1, 1), CubeDim::new_1d(32),
                a(p, pn), a(qn, hd), a(kn, hd), a(cs, 2 * half), a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), ko as u32, vo as u32, nh as u32, nkv as u32, hd as u32, half as u32, pos as u32, eps);
            if attn2() && attn3() && hd <= 512 && hd % 32 == 0 { k_attn3::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_2d(32, 8),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip(), (hd / 32) as u32); } else
            if attn2() && hd <= 512 { k_attn2::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_2d(32, 8),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip()); } else {
            k_attn::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(32),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip()); }
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
    /// QSA: pool index block b of `ik` into `ib` (see k_qsa_pool); cs = rope table at position b*ir.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_pool(&self, ik: &Handle, ikn_: usize, ikn: &Handle, cs: &Handle, ib: &Handle, ibn: usize, b: usize, ir: usize, ihd: usize, half: usize, eps: f32) {
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        unsafe { k_qsa_pool::launch_unchecked(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), a(ik, ikn_), a(ikn, ihd), a(cs, 2 * half), a(ib, ibn), b as u32, ir as u32, ihd as u32, half as u32, eps); }
    }
    /// QSA sparse decode attention for one token: as `attn`, but only keys in the top-kk index blocks (of nb complete
    /// blocks pooled in `ib`) plus the incomplete tail. iqo = offset of the index query in `p`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_qsa(&self, p: &Handle, pn: usize, ko: usize, vo: usize, iqo: usize, qn: &Handle, kn: &Handle, iqn: &Handle, cs: &Handle, kc: &Handle, vc: &Handle, cap: usize, ib: &Handle, ibn: usize, nb: usize,
                    out: &Handle, nh: usize, nkv: usize, hd: usize, half: usize, pos: usize, eps: f32, inh: usize, ihd: usize, ir: usize, kk: usize) {
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        let q = self.empty(nh * hd); let sc = self.empty(nh * cap); let iq = self.empty(inh * ihd); let bs = self.empty(nb.max(1)); let sel = self.empty(nb.max(1));
        unsafe {
            k_attn_prep::launch_unchecked(&self.client, CubeCount::Static((nh + nkv) as u32, 1, 1), CubeDim::new_1d(32),
                a(p, pn), a(qn, hd), a(kn, hd), a(cs, 2 * half), a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), ko as u32, vo as u32, nh as u32, nkv as u32, hd as u32, half as u32, pos as u32, eps);
            k_qsa_q::launch_unchecked(&self.client, CubeCount::Static(inh as u32, 1, 1), CubeDim::new_1d(32), a(p, pn), a(iqn, ihd), a(cs, 2 * half), a(&iq, inh * ihd), iqo as u32, ihd as u32, half as u32, eps);
            k_qsa_score::launch_unchecked(&self.client, CubeCount::Static(nb as u32, 1, 1), CubeDim::new_1d(32), a(&iq, inh * ihd), a(ib, ibn), a(&bs, nb), inh as u32, ihd as u32, (ihd as f32).sqrt().recip());
            k_qsa_sel::launch_unchecked(&self.client, CubeCount::Static(nb.div_ceil(256) as u32, 1, 1), CubeDim::new_1d(256), a(&bs, nb), a(&sel, nb), nb as u32, kk as u32);
            k_attn3m::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_2d(32, 8),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip(), a(&sel, nb), ir as u32, nb as u32, (hd / 32) as u32);
        }
    }
    pub fn xarena(&self, bytes: usize) -> Handle { crate::limits::vram_check(bytes as u64); self.client.empty(bytes) }
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
        XEnt { off: [b, b + uo as u32, b + dof as u32], arena: 0, slot: base / slot_bytes.max(1), bytes: slot_bytes }
    }
    /// Upload one pre-packed expert (`xpacked`: gate words, then up at `uo`, down at `dof`) into slot `slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn xupload_pk(&self, ar: &Handle, ar_bytes: usize, slot: usize, slot_bytes: usize, v: &cubecl::bytes::Bytes, uo: usize, dof: usize) -> XEnt {
        self.xupload_at(ar, ar_bytes, slot * slot_bytes, slot_bytes, v, uo, dof)
    }
    /// xupload_pk at a byte offset with a given capacity (PYRE_QN_XCLASS per-class slots)
    pub fn xupload_at(&self, ar: &Handle, ar_bytes: usize, base: usize, slot_bytes: usize, v: &cubecl::bytes::Bytes, uo: usize, dof: usize) -> XEnt {
        let t0 = std::time::Instant::now();
        let len = v.len(); assert!(len <= slot_bytes && base + len <= ar_bytes);
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
        XEnt { off: [b, b + uo as u32, b + dof as u32], arena: 0, slot: base / slot_bytes.max(1), bytes: slot_bytes }
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
    /// xrun_fast_to_m for the entries of up to 4 window tokens in 4 launches (instead of 4 per token).
    /// cnt[ti] = entries of token ti (in `mh` in token order); src = t rows of stride sn (the readback rows);
    /// y = t * d outputs (token ti at ti * d; a token without entries gets zeros, which callers ignore).
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_fast_multi(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, cnt: &[usize], mh: &Handle, src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), y: &Handle) -> bool {
        if cnt.len() > 4 && cnt.len() <= 8 && t8() && x8() { return self.xrun_fast_multi8(tb, ars, ar_bytes, cnt, mh, src, sn, d, ff, tys, y); }
        if cnt.len() > 4 && cnt.len() <= 8 && t8() { let t = cnt.len(); let rs = sn / t; let mut e0 = 0usize; let mut ok = true;
            for c0 in (0..t).step_by(4) { let c1 = (c0 + 4).min(t); let ne: usize = cnt[c0..c1].iter().sum();
                if ne > 0 { ok &= self.xrun_fast_multi(tb, ars, ar_bytes, &cnt[c0..c1], &mh.clone().offset_start((e0 * 4 * 4) as u64), &src.clone().offset_start((c0 * rs * 4) as u64), (c1 - c0) * rs, d, ff, tys, &y.clone().offset_start((c0 * d * 4) as u64)); }
                e0 += ne; }
            return ok; }
        let t = cnt.len(); let ne: usize = cnt.iter().sum();
        if t == 0 || t > 4 || ne == 0 || tys.0 != tys.1 || tys.0 > 1 || tys.2 != 3 || d % 256 != 0 || ff % 32 != 0 || ars.is_empty() || ars.len() > 2 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let aw = ar_bytes / 4; let rpc = 8u32; let ml = ne * 4;
        let mut o = [0u32; 5]; for i in 0..4 { o[i + 1] = o[i] + cnt.get(i).copied().unwrap_or(0) as u32; }
        let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) }; let rp = xshr();
        let (ar, ar1) = (&ars[0], ars.last().unwrap());
        let (xq, xs) = (self.empty(t * d / 4), self.empty(t * d / 32)); let h = self.empty(ne * ff); let (hq, hs) = (self.empty(ne * ff / 4), self.empty(ne * ff / 32));
        let nb = (d / 32) as u32;
        unsafe {
            k_q8pack_m::launch_unchecked(&self.client, CubeCount::Static(nb.div_ceil(64), t as u32, 1), CubeDim::new_1d(64), a(src, sn), a(&xq, t * d / 4), a(&xs, t * d / 32), 0, sn as u32 / t as u32, nb);
            k_xgu_st::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rp), ne as u32, 1), CubeDim::new_2d(32, rp),
                a(&xq, t * d / 4), a(&xs, t * d / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, o[1], o[2], o[3], (d / 4) as u32, (d / 256) as u32, gn, tys.0);
            k_q8pack::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(&h, ne * ff), a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), 0, (ne * ff / 32) as u32);
            k_xdown_qt::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), t as u32, 1), CubeDim::new_2d(32, rpc),
                a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(&tb.q2l, 256), a(y, t * d), d as u32, ff as u32, rbs.2 as u32, o[1], o[2], o[3], o[4]);
        }
        true
    }
    /// xrun_fast_multi for 5-8 tokens in one set of launches (PYRE_QN_X8, default on under T8)
    pub fn xrun_fast_multi8(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, cnt: &[usize], mh: &Handle, src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), y: &Handle) -> bool {
        let t = cnt.len(); let ne: usize = cnt.iter().sum();
        if t == 0 || t > 8 || ne == 0 || tys.0 != tys.1 || tys.0 > 1 || tys.2 != 3 || d % 256 != 0 || ff % 32 != 0 || ars.is_empty() || ars.len() > 2 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let aw = ar_bytes / 4; let rpc = 8u32; let ml = ne * 4;
        let mut o = [0u32; 9]; for i in 0..8 { o[i + 1] = o[i] + cnt.get(i).copied().unwrap_or(0) as u32; }
        let eo = self.hup(u32::as_bytes(&o));
        let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) }; let rp = xshr();
        let (ar, ar1) = (&ars[0], ars.last().unwrap());
        let (xq, xs) = (self.empty(t * d / 4), self.empty(t * d / 32)); let h = self.empty(ne * ff); let (hq, hs) = (self.empty(ne * ff / 4), self.empty(ne * ff / 32));
        let nb = (d / 32) as u32;
        unsafe {
            k_q8pack_m::launch_unchecked(&self.client, CubeCount::Static(nb.div_ceil(64), t as u32, 1), CubeDim::new_1d(64), a(src, sn), a(&xq, t * d / 4), a(&xs, t * d / 32), 0, sn as u32 / t as u32, nb);
            k_xgu_st8::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rp), ne as u32, 1), CubeDim::new_2d(32, rp),
                a(&xq, t * d / 4), a(&xs, t * d / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, a(&eo, 9), t as u32, (d / 4) as u32, (d / 256) as u32, gn, tys.0);
            k_q8pack::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(&h, ne * ff), a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), 0, (ne * ff / 32) as u32);
            k_xdown_qt8::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), t as u32, 1), CubeDim::new_2d(32, rpc),
                a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(&tb.q2l, 256), a(y, t * d), d as u32, ff as u32, rbs.2 as u32, a(&eo, 9));
        }
        true
    }
    /// xrun_fast_multi with a fixed shape (PYRE_QN_MOEGRAPH): `mh` holds max_ne entries (padding allowed past the
    /// real ones), `oc` the 4 cumulative per-token entry counts, both written by the host before the launch, so the
    /// launches (grid max_ne, buffer sizes) do not depend on the routing and can be captured once and replayed.
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_graph_multi(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, t: usize, max_ne: usize, mh: &Handle, oc: &Handle, src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), y: &Handle) -> bool {
        let ne = max_ne;
        if t == 0 || t > 4 || ne == 0 || tys.0 != tys.1 || tys.0 > 1 || tys.2 != 3 || d % 256 != 0 || ff % 32 != 0 || ars.is_empty() || ars.len() > 2 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let aw = ar_bytes / 4; let rpc = 8u32; let ml = ne * 4;
        let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) }; let rp = xshr();
        let (ar, ar1) = (&ars[0], ars.last().unwrap());
        let (xq, xs) = (self.empty(t * d / 4), self.empty(t * d / 32)); let h = self.empty(ne * ff); let (hq, hs) = (self.empty(ne * ff / 4), self.empty(ne * ff / 32));
        let nb = (d / 32) as u32;
        unsafe {
            k_q8pack_m::launch_unchecked(&self.client, CubeCount::Static(nb.div_ceil(64), t as u32, 1), CubeDim::new_1d(64), a(src, sn), a(&xq, t * d / 4), a(&xs, t * d / 32), 0, sn as u32 / t as u32, nb);
            k_xgu_stg::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rp), ne as u32, 1), CubeDim::new_2d(32, rp),
                a(&xq, t * d / 4), a(&xs, t * d / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, a(oc, 4), (d / 4) as u32, (d / 256) as u32, gn, tys.0);
            k_q8pack::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(&h, ne * ff), a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), 0, (ne * ff / 32) as u32);
            k_xdown_qtg::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), t as u32, 1), CubeDim::new_2d(32, rpc),
                a(&hq, ne * ff / 4), a(&hs, ne * ff / 32), a(ar, aw), a(ar1, aw), a(mh, ml), a(&tb.q2l, 256), a(y, t * d), d as u32, ff as u32, rbs.2 as u32, a(oc, 4));
        }
        true
    }
    /// PYRE_QN_XEXACT: the cached experts of up to 4 tokens with the CPU's arithmetic (see k_xgu_ex / k_xdown_ex);
    /// `y` gets one row per entry (entry e at e * d, weighted, not summed). `mh` holds ne entries, `oc` the cumulative
    /// per-token counts. Only IQ2_S/IQ2_XXS gate+up with Q2_0 down.
    #[allow(clippy::too_many_arguments)]
    /// PYRE_QN_XEX8=0: the one-row-per-cube exact kernels (reference for the 8-row ones)
    /// PYRE_QN_XFUSE: rsum_ex(gy, cy, tab) then comb3_m(xs, no c, a, sum, gb, ..) as one k_rsc_m launch (bit-identical)
    #[allow(clippy::too_many_arguments)]
    pub fn rsum_comb3_m(&self, gy: &Handle, cy: &Handle, tab: &Handle, kk: usize, xs: &[Handle], a: &Handle, gb: &Handle, gn: usize, gs: usize, goff: usize, dn: &Handle, ds: usize, lr: usize, hc: usize, d: usize) {
        let t = xs.len(); let x = |i: usize| xs[i.min(t - 1)].clone();
        let sz = |hd: &Handle| unsafe { BufferArg::from_raw_parts(hd.clone(), (hd.size_in_used() / 4) as usize) };
        unsafe { k_rsc_m::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x(0), hc * d), BufferArg::from_raw_parts(x(1), hc * d), BufferArg::from_raw_parts(x(2), hc * d), BufferArg::from_raw_parts(x(3), hc * d),
            sz(gy), sz(cy), BufferArg::from_raw_parts(tab.clone(), t * (kk + 1)), BufferArg::from_raw_parts(a.clone(), t * d), BufferArg::from_raw_parts(gb.clone(), gn), BufferArg::from_raw_parts(dn.clone(), t * ds),
            d as u32, kk as u32, gs as u32, goff as u32, ds as u32, lr as u32, hc as u32) }
    }
    /// PYRE_QN_XGR: k_xgr_topk over the t router rows of `rd` (stride rn, logits at d..d+ne) against the layer table `rt`
    pub fn xgr_topk(&self, rd: &Handle, rdn: usize, rt: &Handle, ne: usize, mh: &Handle, kk: usize, oc: &Handle, dbg: &Handle, rn: usize, d: usize, t: usize) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let cd = (ne.div_ceil(32) * 32).min(1024) as u32;
        unsafe { k_xgr_topk::launch_unchecked(&self.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(cd), a(rd, rdn), a(rt, ne * 4), a(mh, t * kk * 4), a(oc, 4), a(dbg, t * kk), rn as u32, d as u32, ne as u32, kk as u32, t as u32) }
    }
    /// PYRE_QN_XGR: y[i] = fma(w[i / d], y[i], 0) for the first n entries' rows
    pub fn xgr_scale(&self, y: &Handle, w: &Handle, n: usize, d: usize) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        unsafe { k_xgr_scale::launch_unchecked(&self.client, CubeCount::Static(((n * d) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256), a(y, n * d), a(w, n), (n * d) as u32, d as u32) }
    }
    pub fn rsum_ex(&self, gy: &Handle, cy: &Handle, tab: &Handle, y: &Handle, d: usize, kk: usize, t: usize) {
        let a = |hd: &Handle| unsafe { BufferArg::from_raw_parts(hd.clone(), (hd.size_in_used() / 4) as usize) };
        unsafe { k_rsum_ex::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256), a(gy), a(cy), BufferArg::from_raw_parts(tab.clone(), t * (kk + 1)), BufferArg::from_raw_parts(y.clone(), t * d), d as u32, kk as u32) }
    }
    pub fn xrun_exact(&self, tb: &XTabs, ars: &[Handle], ar_bytes: usize, t: usize, ne: usize, mh: &Handle, oc: &Handle, src: &Handle, sn: usize, d: usize, ff: usize, tys: (u32, u32, u32), y: &Handle, scr: Option<&XScr>) -> bool {
        if t == 0 || t > 4 || ne == 0 || tys.0 != tys.1 || tys.0 > 1 || tys.2 != 3 || d % 256 != 0 || ff % 64 != 0 || d / 256 > 16 || ff / 64 > 64 || ars.is_empty() || ars.len() > 2 { return false; }
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let aw = ar_bytes / 4; let ml = ne * 4;
        let (gsrc, gn) = if tys.0 == 0 { (&tb.g2s, 2048u32) } else { (&tb.g2x, 512u32) };
        let (ar, ar1) = (&ars[0], ars.last().unwrap());
        let nsb = d / 256; let nb = ff / 64;
        // PYRE_QN_XEXACT decode: the persistent XScr buffers (sizes checked: xs d/32 >= nsb, hs holds hd and hs) instead of six pool allocations
        let (xq, xd, h, hq, hd, hs) = match scr { Some(sc) if t == 1 && ne <= sc.max_ne => (sc.xq.clone(), sc.xs.clone(), sc.h.clone(), sc.hq.clone(), sc.hs.clone(), sc.hs.clone().offset_start((ne * nb * 4) as u64)),
            _ => (self.empty(t * d / 4), self.empty(t * nsb), self.empty(ne * ff), self.empty(ne * ff / 4), self.empty(ne * nb), self.empty(ne * nb)) };
        unsafe {
            let stg = XEX_STG.load(std::sync::atomic::Ordering::Relaxed); let gxm = xgx_mode(); let gx = gxm & 1 != 0 && nsb <= 16; let gxd = gxm & 2 != 0 && nb <= 64;
            if stg & 1 != 0 { k_q8k_m::launch_unchecked(&self.client, CubeCount::Static(1, t as u32, 1), CubeDim::new_1d(16), a(src, sn), a(&xq, t * d / 4), a(&xd, t * nsb), sn as u32 / t as u32, nsb as u32); }
            let alt = || XGU2_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 0;
            let (g64, g2) = match xgu2() { 0 => (false, false), 2 => (false, alt()), 3 => (ff % 64 == 0, true), 4 => (ff % 64 == 0 && alt(), true), _ => (false, true) };
            let fits = { let bw = if tys.0 == 0 { 21 } else { 17 }; 32 * nsb * bw <= 6720 && t * d / 4 <= 2560 };
            let g16s = fits && match xgu2() { 5 => true, 6 => alt(), _ => false };
            if stg & 2 == 0 {} else if gx { let lp = xgx_lp(); let gr = xgx_r().min(64).min(512 / (16 / lp)); k_xgu_gx::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(gr), ne as u32, 1), CubeDim::new_1d(gr * 16 / lp),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, gn, tys.0, lp); } else
            if xex8() && g16s { k_xgu_ex16s::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(16), ne as u32, 1), CubeDim::new_2d(32, 8),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, (t * d / 4) as u32, gn, tys.0); } else
            if xex8() && g64 { k_xgu_ex64::launch_unchecked(&self.client, CubeCount::Static((ff / 64) as u32, ne as u32, 1), CubeDim::new_2d(32, 32),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, gn, tys.0); } else
            if xex8() && g2 { k_xgu_ex16::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(16), ne as u32, 1), CubeDim::new_2d(32, 8),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, gn, tys.0); } else
            if xex8() { let ry = xgu_rows(); k_xgu_ex8::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(ry), ne as u32, 1), CubeDim::new_2d(32, ry),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, gn, tys.0); } else {
            k_xgu_ex::launch_unchecked(&self.client, CubeCount::Static(ff as u32, ne as u32, 1), CubeDim::new_1d(32),
                a(&xq, t * d / 4), a(&xd, t * nsb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(gsrc, gn as usize), a(&h, ne * ff), ff as u32, rbs.0 as u32, rbs.1 as u32, nsb as u32, gn, tys.0); }
            if stg & 4 != 0 && (gx || !(xex8() && g64)) { k_q8d::launch_unchecked(&self.client, CubeCount::Static(((ne * nb) as u32).div_ceil(64), 1, 1), CubeDim::new_1d(64), a(&h, ne * ff), a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), (ne * nb) as u32); }
            let alt2 = || XDN2_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 0;
            let (d2s, d2) = match xdn2() { 0 => (false, false), 2 => (false, alt2()), 3 => (nb <= 64, true), 4 => (nb <= 64 && alt2(), true), _ => (false, true) };
            if stg & 8 == 0 {} else if gxd { let dr = xgx_dr(); k_xdown_gx::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(dr), ne as u32, 1), CubeDim::new_1d(dr),
                a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(y, ne * d), d as u32, rbs.2 as u32, nb as u32); } else
            if xex8() && d2s { k_xdown_ex16s::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(16), ne as u32, 1), CubeDim::new_2d(32, 8),
                a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(y, ne * d), d as u32, nb as u32, rbs.2 as u32); } else
            if xex8() && d2 { k_xdown_ex16::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(16), ne as u32, 1), CubeDim::new_2d(32, 8),
                a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(y, ne * d), d as u32, nb as u32, rbs.2 as u32); } else
            if xex8() { k_xdown_ex8::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(8), ne as u32, 1), CubeDim::new_2d(32, 8),
                a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(y, ne * d), d as u32, nb as u32, rbs.2 as u32); } else {
            k_xdown_ex::launch_unchecked(&self.client, CubeCount::Static(d as u32, ne as u32, 1), CubeDim::new_1d(32),
                a(&hq, ne * ff / 4), a(&hd, ne * nb), a(&hs, ne * nb), a(ar, aw), a(ar1, aw), a(mh, ml), a(oc, 4), a(y, ne * d), d as u32, nb as u32, rbs.2 as u32); }
        }
        true
    }
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
    pub fn silu_mul_m(&self, f: &Handle, fnn: usize, fs: usize, og: usize, ou: usize, h: &Handle, n: usize, t: usize) {
        unsafe { k_silu_mul_m::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), t as u32, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(f.clone(), fnn), BufferArg::from_raw_parts(h.clone(), t * n), og as u32, ou as u32, n as u32, fs as u32) }
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
    if std::env::var("PYRE_QN_XBENCH").as_deref() == Ok("ex") { return xbench_ex(); }
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

/// PYRE_QN_XBENCH=ex: xrun_exact on random experts (d 2560, ff 640, both gate types), old ex16 kernels vs grouped
/// (PYRE_QN_XGX=3): bitwise output compare and wall time per stage (q8k, +gate/up, +q8d, +down) over PYRE_QN_XBENCH_REPS.
pub fn xbench_ex() -> anyhow::Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let cb = Cube::new()?; let tb = cb.xtabs();
    let (d, ff) = (2560usize, 640usize);
    let reps: usize = std::env::var("PYRE_QN_XBENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    for tys in [(1u32, 1u32, 3u32), (0u32, 0u32, 3u32)] {
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut rb = |n: usize, blk: usize| -> Vec<u8> { (0..n).map(|i| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; if i % blk < 2 { if i % blk == 0 { 0x00 } else { 0x20 } } else { rng as u8 } }).collect() };
        let nexp = 40usize;
        let (gbk, gw) = if tys.0 == 0 { (82usize, 21usize) } else { (66, 17) }; let gb = ff * d / 256 * gbk; let db = d * ff / 64 * 18;
        let ebytes = (2 * ff * d / 256 * gw + d * ff / 64 * 5) * 4;
        let slot = ebytes.next_multiple_of(256);
        let ar_bytes = slot * nexp; let ar = cb.xarena(ar_bytes); let ars = vec![ar.clone()];
        // xupload_pk (word offsets, byte payload) like the real cache fill
        let ents: Vec<XEnt> = (0..nexp).map(|i| { let mut v = repack(tys.0, &rb(gb, gbk)); let uo = v.len(); v.extend(repack(tys.1, &rb(gb, gbk))); let dof = v.len(); v.extend(repack(tys.2, &rb(db, 18)));
            cb.xupload_pk(&ar, ar_bytes, i, slot, &cubecl::bytes::Bytes::from_elems(v), uo, dof) }).collect();
        for &(t, hpt, sh) in &[(1usize, 4usize, 0usize), (1, 8, 0), (2, 5, 1), (3, 5, 1), (3, 5, 3), (4, 5, 2), (4, 8, 4)] {
            let mut sel: Vec<(&XEnt, f32)> = vec![]; let mut oc = [0u32; 4]; let mut used = std::collections::BTreeSet::new();
            for ti in 0..t { for k in 0..hpt { let ei = if k < sh { k } else { (sh + ti * (hpt - sh) + k - sh) % nexp }; used.insert(ei); sel.push((&ents[ei], 0.05 + 0.01 * (ti * hpt + k) as f32)); } }
            let mut acc = 0u32; for (ti, o) in oc.iter_mut().enumerate() { if ti < t { acc += hpt as u32; } *o = acc; }
            let ne = sel.len();
            let mh = cb.hup(u32::as_bytes(&Cube::xmeta(&ars, &sel))); let och = cb.hup(u32::as_bytes(&oc));
            let xs: Vec<f32> = (0..t * d).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0 * (1.0 + (i % 7) as f32 * 0.1)).collect(); let src = cb.buf(&xs);
            let mut outs = vec![];
            for (mode, lp) in [(0u32, 0u32), (3, 8), (3, 4), (3, 2)] {
                XEX_OVR.store(mode, Relaxed); XEX_STG.store(15, Relaxed); XGX_LP_OVR.store(lp, Relaxed);
                let y = cb.empty(ne * d);
                assert!(cb.xrun_exact(&tb, &ars, ar_bytes, t, ne, &mh, &och, &src, t * d, d, ff, tys, &y, None));
                outs.push(cb.read(&y));
                let mut st = vec![];
                for stg in [1u32, 3, 7, 15] {
                    XEX_STG.store(stg, Relaxed); cb.xrun_exact(&tb, &ars, ar_bytes, t, ne, &mh, &och, &src, t * d, d, ff, tys, &y, None); cb.read(&y);
                    let t0 = std::time::Instant::now(); for _ in 0..reps { cb.xrun_exact(&tb, &ars, ar_bytes, t, ne, &mh, &och, &src, t * d, d, ff, tys, &y, None); } cb.read(&y);
                    st.push(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
                }
                eprintln!("xbex ty={} t={t} ne={ne} distinct={} mode={mode} lp={lp}: wall us q8k {:.1} +gu {:.1} +q8d {:.1} +down {:.1} = {:.1} ({:.2} us/pair, {:.0} GB/s unique weights)", tys.0, used.len(), st[0], st[1] - st[0], st[2] - st[1], st[3] - st[2], st[3], st[3] / ne as f64, (used.len() * ebytes) as f64 / st[3] / 1e3);
            }
            XEX_STG.store(15, Relaxed); XEX_OVR.store(u32::MAX, Relaxed);
            XGX_LP_OVR.store(0, Relaxed);
            let a = &outs[0]; let nz = a.iter().filter(|x| **x != 0.0).count();
            for (k, b) in outs.iter().enumerate().skip(1) { let nd = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                eprintln!("xbex ty={} t={t} ne={ne} variant {k}: {nd}/{} outputs differ bitwise old vs grouped ({nz} nonzero)", tys.0, a.len()); }
        }
    }
    Ok(())
}

/// xupload time split: [repack on the host, client.write] in ns.
pub static XUP_NS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub fn xpersist() -> bool { static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XARENA_PERSIST").map(|v| v != "0").unwrap_or(true)) }

fn gemv4s() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_GEMV4S").map(|v| v == "1").unwrap_or(false)) }

/// PYRE_QN_NATCHK=N: for the first N native-IQ4 GEMV calls under PYRE_QN_NATX, recompute on the CPU in ggml/Strata
/// native_iq4_xs_mmvq order (q8_1 quantize roundf(x/d); 128 threads per row, thread tid takes blocks tid/8 + 16k,
/// sub-block tid%8; warps 1-3 added into warp 0 in order; xor butterfly 16..1) and compare bits.
fn natchk(cb: &Cube, m: &CMat, x: &Handle, xoff: usize, xst: usize, xq: &Handle, xs: &Handle, out: &Handle, ooff: usize, ost: usize, nt: usize) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let lim: usize = std::env::var("PYRE_QN_NATCHK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let call = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed); if call >= lim { return; }
    const LV: [i32; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    let (cols, nb32, nb) = (m.cols, m.cols / 32, m.cols / 256);
    let xv = cb.read(x); let gq: Vec<u32> = cb.read(xq).iter().map(|v| v.to_bits()).collect(); let gs = cb.read(xs); let ov = cb.read(out);
    let (mut qbad, mut sbad) = (0usize, 0usize);
    let mut q8 = vec![0i8; nt * cols]; let mut sd = vec![0f32; nt * nb32];
    for c in 0..nt { for b in 0..nb32 { let src = &xv[xoff + c * xst + b * 32..xoff + c * xst + b * 32 + 32];
        let amax = src.iter().fold(0f32, |a, v| a.max(v.abs())); let d = amax / 127.0; sd[c * nb32 + b] = d; if gs[c * nb32 + b].to_bits() != d.to_bits() { sbad += 1; }
        for i in 0..32 { let q = if amax == 0.0 { 0 } else { (src[i] / d).round() as i32 } as i8; q8[c * cols + b * 32 + i] = q;
            let gb = (gq[c * cols / 4 + b * 8 + i / 4] >> (8 * (i % 4))) as u8 as i8; if gb != q { qbad += 1; } } } }
    let (mut tot, mut bad, mut maxu) = (0usize, 0usize, 0u32); let mut rows_nat = 0;
    for (r0, n, w) in &m.nat { rows_nat += n; let wv: Vec<u32> = cb.read(w).iter().map(|v| v.to_bits()).collect();
        for r in 0..*n { for c in 0..nt {
            let blk = |kb: usize, ib: usize| -> (f32, f32) { let b0 = r * nb * 35 + kb * 35; let d = f32::from_bits(wv[b0]); let (sh, sl) = (wv[b0 + 1], wv[b0 + 2]);
                let ls = (((sl >> (4 * ib)) & 15) | (((sh >> (2 * ib)) & 3) << 4)) as i32 - 32; let mut s = 0i32;
                for j in 0..4 { let q = wv[b0 + 3 + ib * 4 + j]; for t in 0..4 { let by = (q >> (8 * t)) & 255; let xo = c * cols + kb * 256 + ib * 32;
                    s += LV[(by & 15) as usize] * q8[xo + 4 * j + t] as i32 + LV[(by >> 4) as usize] * q8[xo + 16 + 4 * j + t] as i32; } }
                let dx = half::f16::from_f32(sd[c * nb32 + kb * 8 + ib]).to_f32(); ((d * dx), ((s * ls) as f32)) };
            let fma = std::env::var("PYRE_QN_NATCHK_FMA").map(|v| v != "0").unwrap_or(true); let mut tmp = [0f32; 128];
            for tid in 0..128 { let mut kb = tid / 8; while kb < nb { let (dd, si) = blk(kb, tid % 8); tmp[tid] = if fma { dd.mul_add(si, tmp[tid]) } else { tmp[tid] + dd * si }; kb += 16; } }
            let mut v = [0f32; 32]; for l in 0..32 { let mut a = tmp[l]; for wp in 1..4 { a += tmp[32 * wp + l]; } v[l] = a; }
            for sft in [16usize, 8, 4, 2, 1] { let o = v; for l in 0..32 { v[l] = o[l] + o[l ^ sft]; } }
            let g = ov[ooff + c * ost + r0 + r]; tot += 1; if g.to_bits() != v[0].to_bits() { bad += 1; maxu = maxu.max((g.to_bits() as i64 - v[0].to_bits() as i64).unsigned_abs() as u32); } } } }
    eprintln!("natchk call {call}: {}x{} nt={nt} nat rows {rows_nat} q8 rows {} | q8_1 codes bad {qbad} scales bad {sbad} | out {bad}/{tot} differ (max {maxu} ulp)", m.rows, cols, m.rows - rows_nat);
}

/// PYRE_QN_NATX_FMA=1: NATX in-lane accumulation as fma(d, sumi, acc) (nvcc contracts ggml's tmp += d*sumi)
fn natx_fma() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_NATX_FMA").map(|v| v == "1").unwrap_or(false)) }

fn gemv8() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_GEMV8").map(|v| v == "1").unwrap_or(false)) }

/// PYRE_QN_T8 (or PYRE_QN_GEMV8=1): verify windows of 5-8 tokens keep the multi-token kernels (split 4 + rest launches)
pub fn t8() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_T8").map(|v| v == "1").unwrap_or(false) || gemv8()) }

/// PYRE_QN_GATE8 (default on under T8): 5-8 token gate as one k_gate_m launch (t is comptime there; per-token arithmetic unchanged)
/// PYRE_QN_W8 (default on; only under T8): single-launch 8-buffer rms / comb / comb3 / delta for 5-8 tokens
fn w8() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_W8").map(|v| v != "0").unwrap_or(true)) }
/// PYRE_QN_X8 (default on; only under T8): single-set xrun_fast_multi for 5-8 tokens
fn x8() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_X8").map(|v| v != "0").unwrap_or(true)) }
fn gate8() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_GATE8").map(|v| v != "0").unwrap_or(true)) }
