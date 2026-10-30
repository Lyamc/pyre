//! GPU side of the qwen4exp GGUF path (Strata's split): the dense, non-routed projections live on
//! the GPU as Burn Q8F block-32 tensors (i-quants / K-quants / BF16 are dequantized and requantized to
//! Q8_0 once at load); routed experts stay on the CPU. Several projections that read the same input
//! are concatenated into one weight so a decode step needs one upload + one readback per group.
use super::gguf_q::QMat;
use anyhow::{anyhow, Result};
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorCreationOptions, TensorData};
use burn::tensor::activation::{sigmoid, silu};
use half::f16;

pub struct Gpu { pub dev: Device, pub dt: FloatDType, pub budget: std::sync::atomic::AtomicI64 }

pub struct GMat { w: Tensor<2>, pub rows: usize, pub cols: usize, pub splits: Vec<usize> }

/// ggml Q8_0 bytes (f16 scale + 32 x i8 per block) of the rows of `ms`, concatenated.
pub fn to_q8_0(ms: &[&QMat]) -> Vec<u8> {
    use rayon::prelude::*;
    let cols = ms[0].cols; let nb = cols / 32; let rb = nb * 34;
    let rows: Vec<(&QMat, usize)> = ms.iter().flat_map(|m| (0..m.rows).map(move |r| (*m, r))).collect();
    let mut out = vec![0u8; rows.len() * rb];
    out.par_chunks_mut(rb).zip(rows.par_iter()).for_each_init(|| vec![0f32; cols], |buf, (o, &(m, r))| {
        m.row(r, buf);
        for b in 0..nb {
            let x = &buf[b * 32..(b + 1) * 32];
            let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let ob = &mut o[b * 34..(b + 1) * 34];
            ob[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for j in 0..32 { ob[2 + j] = ((x[j] * id).round().clamp(-127.0, 127.0) as i8) as u8; }
        }
    });
    out
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let dt = match std::env::var("PYRE_QN_GPU_DTYPE").as_deref() { Ok("bf16") => FloatDType::BF16, Ok("f16") => FloatDType::F16, _ => FloatDType::F32 };
        let (dev, _, _) = crate::backend::select("cuda", "f32", None)?;
        crate::limits::vram_watch();
        // what may be uploaded: free VRAM above pyre's reserve, less 1.5 GiB for activations/kernels
        let v = crate::limits::vram().ok_or_else(|| anyhow!("no NVML VRAM reading"))?;
        let budget = v.free as i64 - crate::limits::reserve() as i64 - (1536i64 << 20);
        Ok(Self { dev, dt, budget: budget.into() })
    }
    pub fn budget_left(&self) -> i64 { self.budget.load(std::sync::atomic::Ordering::Relaxed) }
    /// Upload `ms` (same `cols`) as one [sum rows, cols] Q8 weight, or None when the budget is spent.
    pub fn upload(&self, ms: &[&QMat]) -> Result<Option<GMat>> {
        let cols = ms[0].cols; anyhow::ensure!(ms.iter().all(|m| m.cols == cols) && cols % 32 == 0);
        let rows: usize = ms.iter().map(|m| m.rows).sum();
        let need = (rows * cols / 32 * 34) as i64;
        if self.budget.fetch_sub(need, std::sync::atomic::Ordering::Relaxed) < need { self.budget.fetch_add(need, std::sync::atomic::Ordering::Relaxed); return Ok(None); }
        crate::limits::vram_check(need as u64);
        let q = to_q8_0(ms);
        let ty = burn_gguf::GgmlType::Q8_0; let scheme = ty.to_quant_scheme().unwrap();
        let packed = burn_gguf::repack_to_burn(ty, &q, scheme, rows * cols).map_err(|e| anyhow!("{e}"))?;
        let mk = |packed: Vec<u8>| Tensor::<2>::from_data(TensorData::from_bytes_vec(packed, vec![rows, cols], DType::QFloat(scheme)), TensorCreationOptions::new(self.dev.clone()));
        // PYRE_QN_XARENA_PERSIST: all burn weights (incl. the ~0.7 GB Q8 lm head) in CubeCL's persistent pool: a weight left in a dynamic pool that later goes outdated (page size grew) is copied by the relocation in graph_prepare (+2.4 GB transient on Q8). Big ones in CubeCL's persistent pool, so they
        // do not grow the dynamic pool's page size to their size (every later dynamic page would then be ~700 MB, mostly empty)
        let w = if std::env::var("PYRE_QN_XARENA_PERSIST").map(|v| v != "0").unwrap_or(true) {
            match burn_cubecl::cubecl::Device::cuda(0) { Ok(d) => { let cl = d.client(); let w = cl.memory_persistent_allocation(packed, |p| mk(p)); let _ = cl.flush(); w } Err(_) => mk(packed) }
        } else { mk(packed) };
        let mut splits = vec![0]; for m in ms { splits.push(splits.last().unwrap() + m.rows); }
        Ok(Some(GMat { w: w.transpose(), rows, cols, splits }))
    }
}

impl GMat {
    /// y[t][rows] = x[t][cols] @ W^T (all groups concatenated along rows).
    pub fn matmul(&self, g: &Gpu, x: &[f32], t: usize, y: &mut [f32]) {
        let t0 = std::time::Instant::now();
        let xt = Tensor::<2>::from_data(TensorData::new(x[..t * self.cols].to_vec(), [t, self.cols]), TensorCreationOptions::new(g.dev.clone()).with_dtype(g.dt.into()));
        let o = xt.matmul(self.w.clone());
        let v = o.into_data().convert::<f32>().into_vec::<f32>().expect("gpu readback");
        y[..t * self.rows].copy_from_slice(&v);
        GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub static GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static GPU_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// (calls, seconds) spent in GPU projections since the last call.
pub fn take_stats() -> (u64, f64) { use std::sync::atomic::Ordering::Relaxed; (GPU_N.swap(0, Relaxed), GPU_NS.swap(0, Relaxed) as f64 / 1e9) }

impl Gpu {
    pub fn opts(&self) -> TensorCreationOptions { TensorCreationOptions::new(self.dev.clone()).with_dtype(self.dt.into()) }
    pub fn up(&self, x: &[f32], shape: [usize; 2]) -> Tensor<2> { Tensor::from_data(TensorData::new(x[..shape[0] * shape[1]].to_vec(), shape), self.opts()) }
    pub fn up3(&self, x: &[f32], shape: [usize; 3]) -> Tensor<3> { Tensor::from_data(TensorData::new(x.to_vec(), shape), self.opts()) }
    /// One device->host copy (the sync point of a step).
    pub fn read(&self, t: Tensor<2>) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let v = t.into_data().convert::<f32>().into_vec::<f32>().expect("gpu readback");
        GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
}

impl GMat {
    pub fn fwd(&self, x: Tensor<2>) -> Tensor<2> { x.matmul(self.w.clone()) }
    /// Host copy [t][rows] -> one [t][rows_i] vector per member.
    pub fn split_host(&self, y: &[f32], t: usize) -> Vec<Vec<f32>> {
        (0..self.splits.len() - 1).map(|i| { let (a, b) = (self.splits[i], self.splits[i + 1]); let mut o = Vec::with_capacity(t * (b - a));
            for ti in 0..t { o.extend_from_slice(&y[ti * self.rows + a..ti * self.rows + b]); } o }).collect()
    }
}

/// A hyper-connection mixer resident on the GPU: grouped RMSNorm, low-rank gate, stream mean.
pub struct GHc { pub norm: Tensor<3>, pub di: GMat, pub up: GMat, pub inj: bool }
impl GHc {
    /// x [t, hc*d] -> (mixed [t, d], inject/hc [t, hc]).
    pub fn mix(&self, x: Tensor<2>, t: usize, hc: usize, d: usize, eps: f32) -> (Tensor<2>, Option<Tensor<2>>) {
        let x3 = x.reshape([t, hc, d]);
        let ms = x3.clone().powi_scalar(2).mean_dim(2);
        let xn = x3 * ms.add_scalar(eps).sqrt().recip() * self.norm.clone();
        let di = self.di.fwd(xn.clone().reshape([t, hc * d]));
        let lr = self.di.splits[1];
        let lo = silu(di.clone().slice([0..t, 0..lr]).div_scalar(hc as f32));
        let inj = if self.inj { Some(di.slice([0..t, lr..lr + hc]).div_scalar(hc as f32)) } else { None };
        let gate = self.up.fwd(lo).reshape([t, hc, d]);
        ((xn * sigmoid(gate)).mean_dim(1).reshape([t, d]), inj)
    }
}

/// x [t, hc*d] += out [t, d] * 2 sigmoid(inj [t, hc]) per stream.
pub fn combine(x: Tensor<2>, out: Tensor<2>, inj: Tensor<2>, t: usize, hc: usize, d: usize) -> Tensor<2> {
    (x.reshape([t, hc, d]) + out.reshape([t, 1, d]) * sigmoid(inj).mul_scalar(2.0).reshape([t, hc, 1])).reshape([t, hc * d])
}

/// One decoder layer's dense half on the GPU (Strata's split): HC mixers, mixer in/out projections,
/// router + shared expert. Routed experts and the token-serial DeltaNet/attention cores stay on the host.
pub struct GLayer { pub hca: GHc, pub min: GMat, pub mout: GMat, pub hcf: GHc, pub fin: GMat, pub sdown: GMat, pub sgw: Tensor<2> }
