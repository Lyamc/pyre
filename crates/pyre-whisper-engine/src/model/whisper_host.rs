//! CPU Whisper encoder on top of `matrixmultiply`.
//!
//! The production pass already caches decoder keys. The remaining cost on CPU is the
//! encoder over the full 30 s mel (absolute positions on the zero tail, so the tail stays).
//! This path runs that same graph with SGEMM. It is used only when the device is the flex
//! CPU backend and `PYRE_STT_HOST_ENCODER` is not `0`. `std::simd` is not on the stable
//! toolchain this repo pins; `matrixmultiply` is the SIMD kernel already used for GGUF.

use std::cell::RefCell;

use burn::tensor::{Tensor, TensorData};

use super::encoder::{AudioEncoder, ResidualEncoderAttentionBlock};

struct HostLin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    inn: usize,
    out: usize,
}

struct HostBlock {
    attn_ln_g: Vec<f32>,
    attn_ln_b: Vec<f32>,
    q: HostLin,
    k: HostLin,
    v: HostLin,
    o: HostLin,
    n_head: usize,
    mlp_ln_g: Vec<f32>,
    mlp_ln_b: Vec<f32>,
    fc1: HostLin,
    fc2: HostLin,
}

struct HostEnc {
    key: u64,
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    n_mels: usize,
    state: usize,
    pos: Vec<f32>,
    blocks: Vec<HostBlock>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
}

thread_local! {
    static CACHE: RefCell<Option<HostEnc>> = const { RefCell::new(None) };
}

pub fn enabled() -> bool {
    // Off unless asked. The Burn encoder stays the default; this path is a parity check.
    matches!(std::env::var("PYRE_STT_HOST_ENCODER").as_deref(), Ok("1" | "true" | "on"))
}

pub fn on_cpu(device: &burn::tensor::Device) -> bool {
    let label = format!("{device:?}").to_ascii_lowercase();
    label.contains("flex") || label.contains("cpu") || label.contains("ndarray")
}

/// Encoder hidden `[1, ctx, state]`, or `None` when this graph is not the Whisper conv encoder.
pub fn try_encode(enc: &AudioEncoder, mel: Tensor<3>) -> Option<Tensor<3>> {
    if !enabled() || !on_cpu(&mel.device()) {
        return None;
    }
    let [batch, n_mels, _time] = mel.dims();
    if batch != 1 || n_mels != enc.n_mels || enc.positional_embedding.is_none() {
        return None;
    }
    let input = mel.clone().into_data().to_vec::<f32>().ok()?;
    let time = input.len() / n_mels.max(1);
    let conv1 = enc.conv1.weight.val().clone().into_data().to_vec::<f32>().ok()?;
    let mut key = fnv(&conv1);
    key ^= (enc.blocks.len() as u64) << 48;
    key ^= enc.n_audio_ctx as u64;
    let (y, ctx, state) = CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().map(|h| h.key) != Some(key) {
            *slot = build(enc, key, conv1);
        }
        let host = slot.as_ref()?;
        let (y, ctx) = encode_host(host, &input, time)?;
        Some((y, ctx, host.state))
    })?;
    let device = mel.device();
    Some(Tensor::<3>::from_data(TensorData::new(y, [1, ctx, state]), &device))
}

fn build(enc: &AudioEncoder, key: u64, conv1_w: Vec<f32>) -> Option<HostEnc> {
    let conv1_b = enc.conv1.bias.as_ref()?.val().clone().into_data().to_vec::<f32>().ok()?;
    let conv2_w = enc.conv2.weight.val().clone().into_data().to_vec::<f32>().ok()?;
    let conv2_b = enc.conv2.bias.as_ref()?.val().clone().into_data().to_vec::<f32>().ok()?;
    let pos_param = enc.positional_embedding.as_ref()?;
    let pos = pos_param.val().clone().into_data().to_vec::<f32>().ok()?;
    let state = enc.ln_post.gamma.dims()[0];
    let mut blocks = Vec::with_capacity(enc.blocks.len());
    for block in &enc.blocks {
        blocks.push(build_block(block)?);
    }
    let ln_g = enc.ln_post.gamma.val().clone().into_data().to_vec::<f32>().ok()?;
    let ln_b = enc.ln_post.beta.as_ref()?.val().clone().into_data().to_vec::<f32>().ok()?;
    Some(HostEnc {
        key,
        conv1_w,
        conv1_b,
        conv2_w,
        conv2_b,
        n_mels: enc.n_mels,
        state,
        pos,
        blocks,
        ln_g,
        ln_b,
    })
}

fn build_block(block: &ResidualEncoderAttentionBlock) -> Option<HostBlock> {
    Some(HostBlock {
        attn_ln_g: vec1(&block.attn_ln.gamma)?,
        attn_ln_b: vec1_beta(&block.attn_ln.beta)?,
        q: lin(&block.attn.query)?,
        k: lin(&block.attn.key)?,
        v: lin(&block.attn.value)?,
        o: lin(&block.attn.out)?,
        n_head: block.attn.n_head,
        mlp_ln_g: vec1(&block.mlp_ln.gamma)?,
        mlp_ln_b: vec1_beta(&block.mlp_ln.beta)?,
        fc1: lin(&block.mlp.lin1)?,
        fc2: lin(&block.mlp.lin2)?,
    })
}

fn vec1(p: &burn::module::Param<Tensor<1>>) -> Option<Vec<f32>> {
    p.val().clone().into_data().to_vec::<f32>().ok()
}

fn vec1_beta(p: &Option<burn::module::Param<Tensor<1>>>) -> Option<Vec<f32>> {
    vec1(p.as_ref()?)
}

fn lin(layer: &burn::nn::Linear) -> Option<HostLin> {
    let w = layer.weight.val().clone().into_data().to_vec::<f32>().ok()?;
    let [inn, out] = layer.weight.dims();
    let b = match &layer.bias {
        Some(b) => Some(b.val().clone().into_data().to_vec::<f32>().ok()?),
        None => None,
    };
    Some(HostLin { w, b, inn, out })
}

fn encode_host(h: &HostEnc, mel: &[f32], time: usize) -> Option<(Vec<f32>, usize)> {
    let (mut hidden, t1) = conv1d(mel, h.n_mels, time, &h.conv1_w, &h.conv1_b, h.state, 3, 1, 1)?;
    gelu_slice(&mut hidden);
    let (mut c2, t2) = conv1d(&hidden, h.state, t1, &h.conv2_w, &h.conv2_b, h.state, 3, 2, 1)?;
    gelu_slice(&mut c2);
    let ctx = t2.min(h.pos.len() / h.state.max(1));
    let mut x = vec![0f32; ctx * h.state];
    for t in 0..ctx {
        for c in 0..h.state {
            x[t * h.state + c] = c2[c * t2 + t] + h.pos[t * h.state + c];
        }
    }
    for block in &h.blocks {
        block_forward(block, &mut x, ctx, h.state);
    }
    layer_norm(&mut x, ctx, h.state, &h.ln_g, &h.ln_b);
    Some((x, ctx))
}

fn gelu_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        let z = *v / std::f32::consts::SQRT_2;
        *v = 0.5 * *v * (1.0 + libm::erff(z));
    }
}

fn block_forward(b: &HostBlock, x: &mut [f32], seq: usize, state: usize) {
    let mut normed = x.to_vec();
    layer_norm(&mut normed, seq, state, &b.attn_ln_g, &b.attn_ln_b);
    let attn = attention(&normed, seq, state, b);
    for (dst, src) in x.iter_mut().zip(attn) {
        *dst += src;
    }
    let mut normed = x.to_vec();
    layer_norm(&mut normed, seq, state, &b.mlp_ln_g, &b.mlp_ln_b);
    let mut mid = gemm(&normed, seq, &b.fc1);
    gelu_slice(&mut mid);
    let mlp = gemm(&mid, seq, &b.fc2);
    for (dst, src) in x.iter_mut().zip(mlp) {
        *dst += src;
    }
}

fn attention(x: &[f32], seq: usize, state: usize, b: &HostBlock) -> Vec<f32> {
    let q = gemm(x, seq, &b.q);
    let k = gemm(x, seq, &b.k);
    let v = gemm(x, seq, &b.v);
    let head = state / b.n_head.max(1);
    let scale = (head as f32).sqrt().recip();
    let mut ctx = vec![0f32; seq * state];
    let mut qh = vec![0f32; seq * head];
    let mut kh = vec![0f32; seq * head];
    let mut vh = vec![0f32; seq * head];
    let mut scores = vec![0f32; seq * seq];
    let mut mixed = vec![0f32; seq * head];
    for h_i in 0..b.n_head {
        for t in 0..seq {
            let src = t * state + h_i * head;
            qh[t * head..(t + 1) * head].copy_from_slice(&q[src..src + head]);
            kh[t * head..(t + 1) * head].copy_from_slice(&k[src..src + head]);
            vh[t * head..(t + 1) * head].copy_from_slice(&v[src..src + head]);
        }
        if seq > 0 && head > 0 {
            unsafe {
                matrixmultiply::sgemm(
                    seq,
                    head,
                    seq,
                    scale,
                    qh.as_ptr(),
                    head as isize,
                    1,
                    kh.as_ptr(),
                    1,
                    head as isize,
                    0.0,
                    scores.as_mut_ptr(),
                    seq as isize,
                    1,
                );
            }
        }
        for i in 0..seq {
            let row = &mut scores[i * seq..(i + 1) * seq];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = 1.0 / sum.max(1e-20);
            for s in row.iter_mut() {
                *s *= inv;
            }
        }
        if seq > 0 && head > 0 {
            unsafe {
                matrixmultiply::sgemm(
                    seq,
                    seq,
                    head,
                    1.0,
                    scores.as_ptr(),
                    seq as isize,
                    1,
                    vh.as_ptr(),
                    head as isize,
                    1,
                    0.0,
                    mixed.as_mut_ptr(),
                    head as isize,
                    1,
                );
            }
        }
        for t in 0..seq {
            let dst = t * state + h_i * head;
            ctx[dst..dst + head].copy_from_slice(&mixed[t * head..(t + 1) * head]);
        }
    }
    gemm(&ctx, seq, &b.o)
}

fn gemm(x: &[f32], seq: usize, lin: &HostLin) -> Vec<f32> {
    let mut y = vec![0f32; seq * lin.out];
    if seq > 0 && lin.inn > 0 && lin.out > 0 {
        unsafe {
            matrixmultiply::sgemm(
                seq,
                lin.inn,
                lin.out,
                1.0,
                x.as_ptr(),
                lin.inn as isize,
                1,
                lin.w.as_ptr(),
                lin.out as isize,
                1,
                0.0,
                y.as_mut_ptr(),
                lin.out as isize,
                1,
            );
        }
    }
    if let Some(b) = &lin.b {
        for t in 0..seq {
            for o in 0..lin.out {
                y[t * lin.out + o] += b[o];
            }
        }
    }
    y
}

fn conv1d(
    x: &[f32],
    cin: usize,
    time: usize,
    w: &[f32],
    b: &[f32],
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Option<(Vec<f32>, usize)> {
    if time + 2 * pad < k {
        return None;
    }
    let out_t = (time + 2 * pad - k) / stride + 1;
    let mut y = vec![0f32; cout * out_t];
    for oc in 0..cout {
        for ot in 0..out_t {
            let mut acc = b.get(oc).copied().unwrap_or(0.0);
            let origin = ot * stride;
            for ic in 0..cin {
                for kk in 0..k {
                    let it = origin as isize + kk as isize - pad as isize;
                    if it < 0 || it >= time as isize {
                        continue;
                    }
                    let xv = x[ic * time + it as usize];
                    let wv = w[(oc * cin + ic) * k + kk];
                    acc += wv * xv;
                }
            }
            y[oc * out_t + ot] = acc;
        }
    }
    Some((y, out_t))
}

fn layer_norm(x: &mut [f32], seq: usize, dim: usize, gamma: &[f32], beta: &[f32]) {
    for t in 0..seq {
        let row = &mut x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| {
            let d = *v - mean;
            d * d
        }).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..dim {
            row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

fn fnv(bytes: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let n = bytes.len().min(64);
    for v in &bytes[..n] {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ bytes.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AudioEncoderConfig;
    use burn::tensor::Distribution;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn host_encoder_matches_burn_on_a_small_graph() {
        let device = burn::tensor::Device::flex();
        let enc = AudioEncoderConfig::new(16, 40, 32, 4, 2).init(&device);
        let mel = Tensor::<3>::random([1, 16, 48], Distribution::Normal(0.0, 0.4), &device);
        let expect = enc.forward(mel.clone()).into_data().to_vec::<f32>().unwrap();
        std::env::set_var("PYRE_STT_HOST_ENCODER", "1");
        let got = try_encode(&enc, mel).expect("host encode").into_data().to_vec::<f32>().unwrap();
        let diff = max_abs(&expect, &got);
        assert!(diff < 2e-3, "host encoder max abs {diff}");
    }
}
