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

#[cube(launch_unchecked)]
fn k_gemv(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let words = cols / 4;
        let wb = row * words;
        let sb = row * (cols / 32);
        let mut acc = 0.0f32;
        let mut wi = UNIT_POS_X;
        while wi < words {
            let word = w[(wb + wi) as usize];
            let k = wi * 4;
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..4u32 {
                let raw = (word >> (j * 8)) & 255u32;
                let v = f32::cast_from(raw) - f32::cast_from(((raw >> 7) & 1u32) << 8);
                part += v * x[(xoff + k + j) as usize];
            }
            acc += part * s[(sb + k / 32) as usize];
            wi += CUBE_DIM_X;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}

/// One cube (256 units) per stream: xn = x * rsqrt(mean(x^2) + eps) * norm.
#[cube(launch_unchecked)]
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

/// mixed[i] = mean_s xn[s,i] * sigmoid(up[s*d+i] . silu(dn[0..lr]/hc)).
#[cube(launch_unchecked)]
fn k_gate(xn: &[f32], dn: &[f32], w: &[u32], s: &[f32], out: &mut [f32], d: u32, ooff: u32, #[comptime] lr: usize, #[comptime] hc: u32) {
    let mut lo = Shared::<[f32]>::new_slice(lr);
    let hcf = f32::cast_from(hc);
    let mut j = UNIT_POS_X;
    while (j as usize) < lr {
        let v = dn[j as usize] / hcf;
        lo[j as usize] = v / (1.0f32 + f32::exp(-v));
        j += CUBE_DIM_X;
    }
    sync_cube();
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < d {
        let words = comptime!(lr as u32 / 4);
        let nb = comptime!(lr as u32 / 32);
        let mut acc = 0.0f32;
        #[unroll]
        for st in 0..hc {
            let row = st * d + i;
            let mut g = 0.0f32;
            for b in 0..nb {
                let mut part = 0.0f32;
                #[unroll]
                for q in 0..8u32 {
                    let word = w[(row * words + b * 8 + q) as usize];
                    #[unroll]
                    for jj in 0..4u32 {
                        let raw = (word >> (jj * 8)) & 255u32;
                        let v = f32::cast_from(raw) - f32::cast_from(((raw >> 7) & 1u32) << 8);
                        part += v * lo[(b * 32 + q * 4 + jj) as usize];
                    }
                }
                g += part * s[(row * nb + b) as usize];
            }
            acc += xn[row as usize] / (1.0f32 + f32::exp(-g));
        }
        out[(ooff + i) as usize] = acc / hcf;
    }
}

/// x[s,i] += (a[i] * (gate? sigmoid(gb[goff]) : 1) + (b? b[i] : 0)) * 2 sigmoid(dn[lr+s]/hc)
#[cube(launch_unchecked)]
fn k_comb(x: &mut [f32], a: &[f32], b: &[f32], gb: &[f32], dn: &[f32], d: u32, goff: u32, lr: u32, #[comptime] hc: u32, #[comptime] gate: bool, #[comptime] addb: bool) {
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
        let inj = dn[(lr + st) as usize] / f32::cast_from(hc);
        x[k as usize] += o * 2.0f32 / (1.0f32 + f32::exp(-inj));
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

pub struct Cube { pub client: Client, pub budget: std::sync::atomic::AtomicI64 }

pub struct CMat { w: Handle, s: Handle, pub rows: usize, pub cols: usize, pub splits: Vec<usize> }

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
        Some(CMat { w: self.client.create_from_slice(u32::as_bytes(&words)), s: self.client.create_from_slice(f32::as_bytes(&sc)), rows, cols, splits })
    }
    pub fn buf(&self, v: &[f32]) -> Handle { self.client.create_from_slice(f32::as_bytes(v)) }
    pub fn empty(&self, n: usize) -> Handle { self.client.empty(n * 4) }
    pub fn read(&self, h: &Handle) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let b = self.client.read_one(h.clone()).expect("gpu readback");
        let v = f32::from_bytes(&b).to_vec();
        super::qn_gpu::GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); super::qn_gpu::GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
    /// out[ooff..ooff+rows] = W x[xoff..xoff+cols]
    pub fn gemv(&self, m: &CMat, x: &Handle, xn: usize, xoff: usize, out: &Handle, on: usize, ooff: usize) {
        unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
            BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
            BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, ooff as u32) }
    }
    pub fn rms(&self, x: &Handle, norm: &Handle, xn: &Handle, hc: usize, d: usize, eps: f32) {
        unsafe { k_rms::launch_unchecked(&self.client, CubeCount::Static(hc as u32, 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(norm.clone(), hc * d), BufferArg::from_raw_parts(xn.clone(), hc * d), d as u32, eps) }
    }
    pub fn gate(&self, xn: &Handle, dn: &Handle, up: &CMat, out: &Handle, on: usize, ooff: usize, hc: usize, d: usize) {
        let lr = up.cols;
        unsafe { k_gate::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(128), 1, 1), CubeDim::new_1d(128),
            BufferArg::from_raw_parts(xn.clone(), hc * d), BufferArg::from_raw_parts(dn.clone(), lr + hc), BufferArg::from_raw_parts(up.w.clone(), up.rows * lr / 4),
            BufferArg::from_raw_parts(up.s.clone(), up.rows * lr / 32), BufferArg::from_raw_parts(out.clone(), on), d as u32, ooff as u32, lr, hc as u32) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn comb(&self, x: &Handle, a: &Handle, b: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) {
        let (gb, gn, goff) = g.map(|(h, n, o)| (h.clone(), n, o)).unwrap_or((a.clone(), d, 0));
        let bb = b.cloned().unwrap_or_else(|| a.clone());
        unsafe { k_comb::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(a.clone(), d), BufferArg::from_raw_parts(bb, d), BufferArg::from_raw_parts(gb, gn),
            BufferArg::from_raw_parts(dn.clone(), lr + hc), d as u32, goff as u32, lr as u32, hc as u32, g.is_some(), b.is_some()) }
    }
    pub fn silu_mul(&self, f: &Handle, fnn: usize, og: usize, ou: usize, h: &Handle, n: usize) {
        unsafe { k_silu_mul::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(f.clone(), fnn), BufferArg::from_raw_parts(h.clone(), n), og as u32, ou as u32, n as u32) }
    }
}

/// A hyper-connection mixer: norm weights, down(+inject) rows, up rows.
pub struct CHc { pub norm: Handle, pub di: CMat, pub up: CMat }
/// One layer's dense half (see `Model::layer_cube`). `fin` = router | shared gate | shared up | shared-gate row.
pub struct CLayer { pub hca: CHc, pub min: CMat, pub mout: CMat, pub hcf: CHc, pub fin: CMat, pub sdown: CMat }
