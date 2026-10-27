//! GGUF K-quants kept packed on the GPU as Burn quantized tensors (Burn has no K-quant scheme).
//!
//! * `Q4_K` (w = s*q - m, q in 0..16, 32-value sub-blocks) becomes Burn `Q4F` block-32 with the
//!   exact f32 sub-block scale and q' = q - 8, plus a per-block offset `c = 8s - m`. A linear then is
//!   `x @ W'^T + blocksum(x) @ C^T`, where C is `[out, in/32]` (1/32 of the weight count).
//! * `Q6_K` (w = d*sc*(q - 32), 16-value sub-blocks, symmetric) becomes Burn `Q8F` block-16 with
//!   f32 scales, which is exact.
use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

pub fn q4k_scheme() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q4F).with_store(QuantStore::PackedU32(0)).per_block([32], ScaleDtype::F32)
}
pub fn q6k_scheme() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q8F).with_store(QuantStore::Native).per_block([16], ScaleDtype::F32)
}

fn f16(b: &[u8]) -> f32 { half::f16::from_le_bytes([b[0], b[1]]).to_f32() }

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) } else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

/// Q4_K -> (Burn packed bytes `[u32 values][f32 scales]`, per-32 offsets in value order).
pub fn repack_q4k(data: &[u8], n: usize) -> (Vec<u8>, Vec<f32>) {
    let blocks = n / 256;
    let mut vals = Vec::with_capacity(n / 2);
    let mut scales = Vec::with_capacity(n / 32 * 4);
    let mut offs = Vec::with_capacity(n / 32);
    for b in 0..blocks {
        let blk = &data[b * 144..b * 144 + 144];
        let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
        let sc = &blk[4..16];
        let qs = &blk[16..144];
        for c in 0..4 {
            for half in 0..2 {
                let (s6, m6) = scale_min_k4(2 * c + half, sc);
                let s = d * s6 as f32;
                let m = dmin * m6 as f32;
                scales.extend_from_slice(&s.to_le_bytes());
                offs.push(8.0 * s - m);
                let mut logical = [0u8; 32];
                for l in 0..32 {
                    let byte = qs[32 * c + l];
                    let q = if half == 0 { byte & 0xF } else { byte >> 4 };
                    logical[l] = q ^ 8; // (q - 8) in 4-bit two's complement
                }
                for g in logical.chunks(8) {
                    let mut w = 0u32;
                    for (i, &v) in g.iter().enumerate() { w |= (v as u32) << (4 * i); }
                    vals.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
    }
    vals.extend_from_slice(&scales);
    (vals, offs)
}

/// Q6_K -> Burn bytes `[i8 values][f32 scales]` (block 16).
pub fn repack_q6k(data: &[u8], n: usize) -> Vec<u8> {
    let blocks = n / 256;
    let mut vals = vec![0u8; n];
    let mut scales = Vec::with_capacity(n / 16 * 4);
    for b in 0..blocks {
        let blk = &data[b * 210..b * 210 + 210];
        let (ql, qh, sc, d) = (&blk[0..128], &blk[128..192], &blk[192..208], f16(&blk[208..210]));
        let y = &mut vals[b * 256..b * 256 + 256];
        for h in 0..2 {
            let (ql, qh, y) = (&ql[64 * h..], &qh[32 * h..], &mut y[128 * h..]);
            for l in 0..32 {
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[l] = q1 as u8; y[l + 32] = q2 as u8; y[l + 64] = q3 as u8; y[l + 96] = q4 as u8;
            }
        }
        for s in sc { scales.extend_from_slice(&(d * (*s as i8) as f32).to_le_bytes()); }
    }
    vals.extend_from_slice(&scales);
    vals
}

/// Q5_0 (w = d*(q - 16), block 32) -> Burn `Q8F` block-32 `[i8 values][f16 scales]`, exact.
pub fn repack_q5_0(data: &[u8], n: usize) -> Vec<u8> {
    let blocks = n / 32;
    let mut vals = Vec::with_capacity(n + blocks * 2);
    let mut scales = Vec::with_capacity(blocks * 2);
    for b in 0..blocks {
        let blk = &data[b * 22..b * 22 + 22];
        scales.extend_from_slice(&blk[0..2]);
        let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
        let qs = &blk[6..22];
        let mut y = [0u8; 32];
        for j in 0..16 {
            let h0 = (((qh >> j) << 4) & 0x10) as u8;
            let h1 = ((qh >> (j + 12)) & 0x10) as u8;
            y[j] = (((qs[j] & 0xF) | h0) as i8 - 16) as u8;
            y[j + 16] = (((qs[j] >> 4) | h1) as i8 - 16) as u8;
        }
        vals.extend_from_slice(&y);
    }
    vals.extend_from_slice(&scales);
    vals
}
pub fn q5_0_scheme() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q8F).with_store(QuantStore::Native).per_block([32], ScaleDtype::F16)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q5_0_matches_reference_dequant() {
        let mut s = 3u64;
        let mut data: Vec<u8> = (0..22 * 4).map(|_| lcg(&mut s)).collect();
        for b in 0..4 { data[b * 22..b * 22 + 2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); }
        let r = burn_gguf::dequantize_checked(burn_gguf::GgmlType::Q5_0, &data, 128).unwrap();
        let p = repack_q5_0(&data, 128);
        for i in 0..128 {
            let s = half::f16::from_le_bytes([p[128 + i / 32 * 2], p[128 + i / 32 * 2 + 1]]).to_f32();
            assert!(((p[i] as i8) as f32 * s - r[i]).abs() < 1e-5, "{i}");
        }
    }
    fn lcg(seed: &mut u64) -> u8 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*seed >> 33) as u8 }

    fn q4_nib(vals: &[u8], i: usize) -> i32 { let w = u32::from_le_bytes(vals[i / 8 * 4..i / 8 * 4 + 4].try_into().unwrap()); let v = ((w >> (4 * (i % 8))) & 0xF) as i32; if v >= 8 { v - 16 } else { v } }

    #[test]
    fn q4k_matches_reference_dequant() {
        let mut s = 7u64;
        let mut data: Vec<u8> = (0..144 * 3).map(|_| lcg(&mut s)).collect();
        for b in 0..3 { data[b * 144..b * 144 + 2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes()); data[b * 144 + 2..b * 144 + 4].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes()); }
        let r = burn_gguf::dequantize_checked(burn_gguf::GgmlType::Q4K, &data, 768).unwrap();
        let (p, offs) = repack_q4k(&data, 768);
        let sc = &p[384..];
        for i in 0..768 {
            let s = f32::from_le_bytes(sc[i / 32 * 4..i / 32 * 4 + 4].try_into().unwrap());
            let w = s * q4_nib(&p, i) as f32 + offs[i / 32];
            assert!((w - r[i]).abs() < 1e-4 + 1e-4 * r[i].abs(), "{i}: {w} vs {}", r[i]);
        }
    }

    #[test]
    fn q6k_matches_reference_dequant() {
        let mut s = 9u64;
        let mut data: Vec<u8> = (0..210 * 2).map(|_| lcg(&mut s)).collect();
        for b in 0..2 { data[b * 210 + 208..b * 210 + 210].copy_from_slice(&half::f16::from_f32(0.003).to_le_bytes()); }
        let r = burn_gguf::dequantize_checked(burn_gguf::GgmlType::Q6K, &data, 512).unwrap();
        let p = repack_q6k(&data, 512);
        let sc = &p[512..];
        for i in 0..512 {
            let s = f32::from_le_bytes(sc[i / 16 * 4..i / 16 * 4 + 4].try_into().unwrap());
            let w = s * (p[i] as i8) as f32;
            assert!((w - r[i]).abs() < 1e-5 + 1e-4 * r[i].abs(), "{i}: {w} vs {}", r[i]);
        }
    }

    /// GPU check of the fused dequant+matmul in the burn fork against the unfused path and a host
    /// reference. `cargo test --release --features cuda kquant::tests::fused -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn fused_qmatmul_matches_and_bench() {
        use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData, FloatDType};
        let backend = std::env::var("PYRE_TEST_BACKEND").unwrap_or_else(|_| "cuda".into());
        let (dev, _, _) = crate::backend::select(&backend, "bf16", None).unwrap();
        let mut seed = 11u64;
        let quick = std::env::var("PYRE_FQ_QUICK").is_ok();
        let shapes: &[(usize, usize, u32, usize)] = if quick { &[(896, 896, 4, 32)] } else { &[(896, 896, 4, 32), (4864, 896, 8, 32), (1536, 8960, 8, 16), (8960, 1536, 4, 32)] };
        for &(n, k, bits, block) in shapes {
            // random packed weight
            let nb = n * k / block;
            let (bytes, scheme) = if bits == 4 {
                let mut v: Vec<u8> = (0..n * k / 2).map(|_| lcg(&mut seed)).collect();
                for b in 0..nb { v.extend_from_slice(&(0.002f32 + (b % 7) as f32 * 0.001).to_le_bytes()); }
                (v, q4k_scheme())
            } else {
                let mut v: Vec<u8> = (0..n * k).map(|_| lcg(&mut seed) / 2).collect();
                for b in 0..nb { v.extend_from_slice(&(0.001f32 + (b % 5) as f32 * 0.0005).to_le_bytes()); }
                (v, if block == 16 { q6k_scheme() } else { QuantScheme::default().with_value(QuantValue::Q8F).with_store(QuantStore::Native).per_block([32], ScaleDtype::F32) })
            };
            let wq: Tensor<2> = Tensor::from_data(TensorData::from_bytes_vec(bytes, [n, k], DType::QFloat(scheme)), TensorCreationOptions::new(dev.clone()));
            let wt = wq.clone().transpose();
            let wf: Tensor<2> = wq.dequantize().cast(FloatDType::BF16).transpose();
            for &m in &[1usize, 4, 8, 37, 300] {
                let xs: Vec<f32> = (0..m * k).map(|_| (lcg(&mut seed) as f32 - 127.5) / 128.0).collect();
                let x: Tensor<2> = Tensor::from_data(TensorData::new(xs, [m, k]), TensorCreationOptions::new(dev.clone()).with_dtype(DType::BF16));
                let reference: Vec<f32> = x.clone().matmul(wf.clone()).cast(FloatDType::F32).into_data().to_vec().unwrap();
                burn_cubecl::kernel::matmul::set_fused_qmatmul(true);
                let l0 = burn_cubecl::kernel::matmul::fused_qmatmul_launches();
                let fused: Vec<f32> = x.clone().matmul(wt.clone()).cast(FloatDType::F32).into_data().to_vec().unwrap();
                assert!(burn_cubecl::kernel::matmul::fused_qmatmul_launches() > l0, "fused path not taken");
                let l1 = burn_cubecl::kernel::matmul::fused_qmatmul_launches();
                burn_cubecl::kernel::matmul::set_fused_qmatmul(false);
                let unfused: Vec<f32> = x.clone().matmul(wt.clone()).cast(FloatDType::F32).into_data().to_vec().unwrap();
                assert_eq!(burn_cubecl::kernel::matmul::fused_qmatmul_launches(), l1, "unfused run used the fused kernel");
                println!("    sample fused {:?} unfused {:?}", &fused[..3], &unfused[..3]);
                let err = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let scale = reference.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
                let (ef, eu) = (err(&fused, &reference) / scale, err(&unfused, &reference) / scale);
                let fu = err(&fused, &unfused) / scale;
                println!("q{bits} b{block} n={n} k={k} m={m}: max rel err vs bf16 ref: fused {ef:.2e}, unfused {eu:.2e}; fused vs unfused {fu:.2e}");
                if ef >= 2e-2 { let bad: Vec<usize> = (0..m * n).filter(|&i| (fused[i] - reference[i]).abs() / scale > 2e-2).collect(); println!("    bad count {} first {:?} rows {:?}", bad.len(), &bad[..bad.len().min(8)], bad.iter().map(|i| i / n).collect::<std::collections::BTreeSet<_>>()); }
                assert!(ef < 2e-2, "fused mismatch");
                // timing (device-synced)
                if m == 1 || m == 300 {
                    let reps = if m == 1 { 200 } else { 20 };
                    let mut t = vec![];
                    for (name, on, w) in [("fused", Some(true), &wt), ("unfused", Some(false), &wt), ("bf16", None, &wf)] {
                        if let Some(on) = on { burn_cubecl::kernel::matmul::set_fused_qmatmul(on); }
                        for _ in 0..3 { let _ = x.clone().matmul(w.clone()); }
                        dev.sync().unwrap();
                        let t0 = std::time::Instant::now();
                        for _ in 0..reps { let _ = x.clone().matmul(w.clone()); }
                        dev.sync().unwrap();
                        t.push(format!("{name} {:.1} us", t0.elapsed().as_secs_f64() * 1e6 / reps as f64));
                    }
                    println!("    timing m={m}: {}", t.join(", "));
                }
            }
        }
        burn_cubecl::kernel::matmul::set_fused_qmatmul(true);
    }
}