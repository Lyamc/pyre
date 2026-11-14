//! Fluid Speech Pro extras on top of Parakeet TDT v3.
//!
//! The NeMo checkpoint (loaded by `fluid_speech.py`) inserts one residual block,
//! `encoder.fv_insert`, between subsampling and the first Conformer layer:
//! LayerNorm, a linear down-projection, a depthwise Conv1d, GELU, and a linear
//! up-projection, added back onto the frames. Hugging Face Transformers weights
//! omit that block.
//!
//! Clean mode is not a text filter. The same checkpoint stores a rank-32 LoRA
//! (`clean_mode.<layer>.A` / `.B`) plus `decoder_input` and `joint_hidden`.
//! `load(..., clean_mode=True)` adds `scale * (B @ A)` onto each adapted encoder
//! weight, with `scale = alpha / rank`, and adds
//! `weight_ih_l0 @ decoder_input` onto the first LSTM input bias and
//! `joint_hidden` onto `joint.pred.bias`. Those tensors stay beside the base
//! weights here and are applied only when a request asks for clean mode.

use std::cell::Cell;
use std::fs;
use std::path::Path;

use burn::nn::Linear;
use burn::tensor::activation::gelu;
use burn::tensor::Tensor;

use super::parakeet::Parakeet;

thread_local! {
    static CLEAN: Cell<bool> = const { Cell::new(false) };
    static BEAM: Cell<usize> = const { Cell::new(1) };
}

pub fn clean_on() -> bool {
    CLEAN.with(|c| c.get())
}

pub fn tdt_beam() -> usize {
    BEAM.with(|b| b.get())
}

/// Sets the per-request Fluid Speech flags for the current thread and restores them after.
pub struct PassGuard {
    prev_clean: bool,
    prev_beam: usize,
}

impl PassGuard {
    pub fn new(clean: bool, beam: usize) -> Self {
        let prev_clean = CLEAN.with(|c| c.replace(clean));
        let prev_beam = BEAM.with(|b| b.replace(beam));
        Self { prev_clean, prev_beam }
    }
}

impl Drop for PassGuard {
    fn drop(&mut self) {
        CLEAN.with(|c| c.set(self.prev_clean));
        BEAM.with(|b| b.set(self.prev_beam));
    }
}

/// Low-rank update in Burn layout: `y += scale * (x @ a) @ b`, with `a` `[in, rank]` and `b` `[rank, out]`.
#[derive(Clone, Debug)]
pub struct LoraAB {
    pub a: Tensor<2>,
    pub b: Tensor<2>,
    pub scale: f32,
}

pub fn add_lora(y: Tensor<3>, x: Tensor<3>, lora: Option<&LoraAB>) -> Tensor<3> {
    if !clean_on() {
        return y;
    }
    let Some(lora) = lora else {
        return y;
    };
    let hidden = x.matmul(lora.a.clone().unsqueeze());
    let delta = hidden.matmul(lora.b.clone().unsqueeze()).mul_scalar(lora.scale);
    y + delta
}

/// `scale * (B @ A)` for a PyTorch Linear/Conv weight, then viewed as `weight_shape`.
pub fn lora_delta(b: &[f32], a: &[f32], b_shape: &[usize], a_shape: &[usize], scale: f32) -> Result<Vec<f32>, String> {
    if b_shape.len() != 2 || a_shape.len() != 2 {
        return Err(format!("LoRA factors must be matrices, got {b_shape:?} and {a_shape:?}"));
    }
    let (b_rows, b_cols) = (b_shape[0], b_shape[1]);
    let (a_rows, a_cols) = (a_shape[0], a_shape[1]);
    if b_cols != a_rows {
        return Err(format!(
            "LoRA matmul {b_rows}x{b_cols} @ {a_rows}x{a_cols} is not defined"
        ));
    }
    if b.len() != b_rows * b_cols || a.len() != a_rows * a_cols {
        return Err("LoRA factor length does not match its shape".into());
    }
    let mut out = vec![0.0f32; b_rows * a_cols];
    for i in 0..b_rows {
        for k in 0..b_cols {
            let bv = b[i * b_cols + k];
            let row = &a[k * a_cols..(k + 1) * a_cols];
            let dest = &mut out[i * a_cols..(i + 1) * a_cols];
            for j in 0..a_cols {
                dest[j] += bv * row[j];
            }
        }
    }
    for v in &mut out {
        *v *= scale;
    }
    Ok(out)
}

/// `weight @ vector` for a row-major `[rows, cols]` matrix and a length-`cols` vector.
pub fn matvec(weight: &[f32], rows: usize, cols: usize, vector: &[f32]) -> Result<Vec<f32>, String> {
    if weight.len() != rows * cols || vector.len() != cols {
        return Err(format!(
            "matvec shape [{rows}, {cols}] x [{}] does not match the buffers",
            vector.len()
        ));
    }
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * vector[c];
        }
        out[r] = acc;
    }
    Ok(out)
}

/// PyTorch `[out, in]` LoRA (`B @ A`) stored for a Burn linear `[in, out]`.
pub fn lora_burn_factors(
    b: &[f32],
    a: &[f32],
    b_shape: &[usize],
    a_shape: &[usize],
    scale: f32,
    device: &burn::tensor::Device,
) -> Result<LoraAB, String> {
    let (out, rank_b) = (b_shape[0], b_shape[1]);
    let (rank_a, inp) = (a_shape[0], a_shape[1]);
    if rank_a != rank_b {
        return Err(format!("LoRA ranks differ: A {rank_a}, B {rank_b}"));
    }
    if a.len() != rank_a * inp || b.len() != out * rank_b {
        return Err("LoRA buffer length does not match the factor shapes".into());
    }
    let mut a_t = vec![0.0f32; inp * rank_a];
    for r in 0..rank_a {
        for c in 0..inp {
            a_t[c * rank_a + r] = a[r * inp + c];
        }
    }
    let mut b_t = vec![0.0f32; rank_b * out];
    for r in 0..out {
        for c in 0..rank_b {
            b_t[c * out + r] = b[r * rank_b + c];
        }
    }
    Ok(LoraAB {
        a: Tensor::<1>::from_floats(a_t.as_slice(), device).reshape([inp, rank_a]),
        b: Tensor::<1>::from_floats(b_t.as_slice(), device).reshape([rank_b, out]),
        scale,
    })
}

#[derive(Clone, Debug)]
pub struct FrameInsert {
    ln_weight: Tensor<1>,
    ln_bias: Tensor<1>,
    down_weight: Tensor<2>,
    down_bias: Tensor<1>,
    mix_weight: Tensor<3>,
    mix_bias: Tensor<1>,
    up_weight: Tensor<2>,
    up_bias: Tensor<1>,
    eps: f32,
}

impl FrameInsert {
    /// Residual block. `x` is `[batch, time, width]`, the subsampling output.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h = layer_norm(x.clone(), &self.ln_weight, &self.ln_bias, self.eps);
        let h = h.matmul(self.down_weight.clone().unsqueeze()) + self.down_bias.clone().unsqueeze();
        let h = depthwise_same(h.swap_dims(1, 2), &self.mix_weight, &self.mix_bias);
        let h = gelu(h).swap_dims(1, 2);
        let h = h.matmul(self.up_weight.clone().unsqueeze()) + self.up_bias.clone().unsqueeze();
        x + h
    }
}

fn layer_norm(x: Tensor<3>, weight: &Tensor<1>, bias: &Tensor<1>, eps: f32) -> Tensor<3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean.clone();
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let h = centered / (var + eps).sqrt();
    h * weight.clone().unsqueeze() + bias.clone().unsqueeze()
}

fn depthwise_same(x: Tensor<3>, weight: &Tensor<3>, bias: &Tensor<1>) -> Tensor<3> {
    let [b, c, t] = x.dims();
    let k = weight.dims()[2];
    let pad = k / 2;
    let padded = if pad == 0 {
        x
    } else {
        let z = Tensor::<3>::zeros([b, c, pad], &x.device());
        Tensor::cat(vec![z.clone(), x, z], 2)
    };
    let mut acc = Tensor::<3>::zeros([b, c, t], &padded.device());
    for i in 0..k {
        let slice = padded.clone().slice([0..b, 0..c, i..i + t]);
        let tap = weight.clone().slice([0..c, 0..1, i..i + 1]).reshape([1, c, 1]);
        acc = acc + slice * tap;
    }
    // Channel axis is the middle dim: [batch, channels, time].
    let bias = bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);
    acc + bias
}

#[derive(Clone, Debug)]
struct FluidSpec {
    scale: f32,
}

fn read_spec(dir: &Path) -> FluidSpec {
    let path = dir.join("fluid_speech.json");
    let text = fs::read_to_string(path).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    let alpha = v.get("alpha").and_then(|x| x.as_f64()).unwrap_or(64.0) as f32;
    let rank = v.get("rank").and_then(|x| x.as_f64()).unwrap_or(32.0) as f32;
    let scale = if v.get("scale").and_then(|x| x.as_f64()).is_some() {
        v["scale"].as_f64().unwrap_or(2.0) as f32
    } else if rank > 0.0 {
        alpha / rank
    } else {
        1.0
    };
    FluidSpec { scale }
}

fn shapes_of(dir: &Path) -> std::collections::HashMap<String, Vec<usize>> {
    let path = dir.join("fluid").join("shapes.json");
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn load_f32(dir: &Path, rel: &str, shapes: &std::collections::HashMap<String, Vec<usize>>) -> Result<(Vec<f32>, Vec<usize>), String> {
    let path = dir.join(rel);
    let buf = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let data = super::load::helpers::decode_npy_f32(&buf).map_err(|e| format!("{rel}: {e}"))?;
    let shape = shapes
        .iter()
        .find(|(k, _)| rel.replace('\\', "/").ends_with(k.as_str()) || k.ends_with(&rel.replace('\\', "/")))
        .map(|(_, v)| v.clone())
        .ok_or_else(|| format!("no shape for {rel}"))?;
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!(
            "{rel} has {} values, shape {shape:?} wants {}",
            data.len(),
            shape.iter().product::<usize>()
        ));
    }
    Ok((data, shape))
}

fn tensor2(data: &[f32], shape: &[usize], device: &burn::tensor::Device) -> Result<Tensor<2>, String> {
    if shape.len() != 2 {
        return Err(format!("expected rank 2, got {shape:?}"));
    }
    Ok(Tensor::<1>::from_floats(data, device).reshape([shape[0], shape[1]]))
}

fn tensor1(data: &[f32], device: &burn::tensor::Device) -> Tensor<1> {
    Tensor::<1>::from_floats(data, device)
}

/// Load `fluid/` sidecars written by the Rust converter. Missing files leave the base Parakeet path unchanged.
pub fn attach_sidecars(model: &mut Parakeet, dir: &Path) -> Result<(), String> {
    let fluid = dir.join("fluid");
    if !fluid.exists() {
        return Ok(());
    }
    let shapes = shapes_of(dir);
    let device = model.pre_encode_out.weight.device();
    if fluid.join("fv_insert/down/weight.npy").exists() {
        model.frame_insert = Some(load_frame_insert(dir, &shapes, &device)?);
    }
    if fluid.join("clean/decoder_input.npy").exists() {
        attach_clean(model, dir, &shapes, &device)?;
    }
    Ok(())
}

fn load_frame_insert(
    dir: &Path,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    device: &burn::tensor::Device,
) -> Result<FrameInsert, String> {
    let (ln_w, _) = load_f32(dir, "fluid/fv_insert/ln/gamma.npy", shapes)?;
    let (ln_b, _) = load_f32(dir, "fluid/fv_insert/ln/beta.npy", shapes)?;
    let (down_w, down_s) = load_f32(dir, "fluid/fv_insert/down/weight.npy", shapes)?;
    let (down_b, _) = load_f32(dir, "fluid/fv_insert/down/bias.npy", shapes)?;
    let (up_w, up_s) = load_f32(dir, "fluid/fv_insert/up/weight.npy", shapes)?;
    let (up_b, _) = load_f32(dir, "fluid/fv_insert/up/bias.npy", shapes)?;
    let (mix_w, mix_s) = load_f32(dir, "fluid/fv_insert/mix/weight.npy", shapes)?;
    let (mix_b, _) = load_f32(dir, "fluid/fv_insert/mix/bias.npy", shapes)?;
    if down_s.len() != 2 || up_s.len() != 2 || mix_s.len() != 3 {
        return Err(format!("frame insert shapes down {down_s:?} up {up_s:?} mix {mix_s:?}"));
    }
    // Converter stores Linear weights already transposed to Burn `[in, out]`.
    Ok(FrameInsert {
        ln_weight: tensor1(&ln_w, device),
        ln_bias: tensor1(&ln_b, device),
        down_weight: tensor2(&down_w, &down_s, device)?,
        down_bias: tensor1(&down_b, device),
        up_weight: tensor2(&up_w, &up_s, device)?,
        up_bias: tensor1(&up_b, device),
        mix_weight: Tensor::<1>::from_floats(mix_w.as_slice(), device).reshape([mix_s[0], mix_s[1], mix_s[2]]),
        mix_bias: tensor1(&mix_b, device),
        eps: 1e-5,
    })
}

fn slot_lora(
    dir: &Path,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    slot: &str,
    scale: f32,
    device: &burn::tensor::Device,
) -> Result<Option<LoraAB>, String> {
    let a_rel = format!("fluid/clean/{slot}/A.npy");
    let b_rel = format!("fluid/clean/{slot}/B.npy");
    if !dir.join(&a_rel).exists() {
        return Ok(None);
    }
    let (a, a_s) = load_f32(dir, &a_rel, shapes)?;
    let (b, b_s) = load_f32(dir, &b_rel, shapes)?;
    Ok(Some(lora_burn_factors(&b, &a, &b_s, &a_s, scale, device)?))
}

fn attach_clean(
    model: &mut Parakeet,
    dir: &Path,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    device: &burn::tensor::Device,
) -> Result<(), String> {
    let spec = read_spec(dir);
    let (dec_in, dec_shape) = load_f32(dir, "fluid/clean/decoder_input.npy", shapes)?;
    let vector = flatten_vector(&dec_in, &dec_shape)?;
    let (joint, _) = load_f32(dir, "fluid/clean/joint_hidden.npy", shapes)?;
    let Some(decoder) = model.decoder.as_mut() else {
        return Err("clean mode needs the TDT decoder".into());
    };
    let layer = decoder.lstm.layers.first().ok_or("decoder LSTM is empty")?;
    let w = layer.weight_ih.val().into_data().to_vec::<f32>().map_err(|e| e.to_string())?;
    let dims = layer.weight_ih.dims();
    let delta = matvec(&w, dims[0], dims[1], &vector)?;
    decoder.lstm.layers[0].clean_bias_ih = Some(tensor1(&delta, device));
    decoder.clean_joint = Some(tensor1(&joint, device));

    let n = model.encoder.layers.len();
    for i in 0..n {
        let block = &mut model.encoder.layers[i];
        block.ff1.lora_lin1 = slot_lora(dir, shapes, &format!("block_{i}/ff1_lin1"), spec.scale, device)?;
        block.ff1.lora_lin2 = slot_lora(dir, shapes, &format!("block_{i}/ff1_lin2"), spec.scale, device)?;
        block.ff2.lora_lin1 = slot_lora(dir, shapes, &format!("block_{i}/ff2_lin1"), spec.scale, device)?;
        block.ff2.lora_lin2 = slot_lora(dir, shapes, &format!("block_{i}/ff2_lin2"), spec.scale, device)?;
        block.conv.lora_pw1 = slot_lora(dir, shapes, &format!("block_{i}/conv_pw1"), spec.scale, device)?;
        block.conv.lora_pw2 = slot_lora(dir, shapes, &format!("block_{i}/conv_pw2"), spec.scale, device)?;
        block.attn.lora_q = slot_lora(dir, shapes, &format!("block_{i}/attn_q"), spec.scale, device)?;
        block.attn.lora_k = slot_lora(dir, shapes, &format!("block_{i}/attn_k"), spec.scale, device)?;
        block.attn.lora_v = slot_lora(dir, shapes, &format!("block_{i}/attn_v"), spec.scale, device)?;
        block.attn.lora_o = slot_lora(dir, shapes, &format!("block_{i}/attn_o"), spec.scale, device)?;
        block.attn.lora_pos = slot_lora(dir, shapes, &format!("block_{i}/attn_pos"), spec.scale, device)?;
    }
    Ok(())
}

fn flatten_vector(data: &[f32], shape: &[usize]) -> Result<Vec<f32>, String> {
    let n: usize = shape.iter().product();
    if n != data.len() {
        return Err(format!("decoder_input has {} values, shape {shape:?} wants {n}", data.len()));
    }
    if shape.len() > 2 || (shape.len() == 2 && shape[0] != 1 && shape[1] != 1) {
        return Err(format!("decoder_input shape {shape:?} is not a vector"));
    }
    Ok(data.to_vec())
}

pub fn frame_seconds() -> f64 {
    0.08
}

/// Used by attention/conv call sites that already hold a `Linear`.
pub fn linear_with_lora(lin: &Linear, x: Tensor<3>, lora: Option<&LoraAB>) -> Tensor<3> {
    let y = lin.forward(x.clone());
    add_lora(y, x, lora)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_delta_matches_scale_b_at_a() {
        // B is [2, 2], A is [2, 3], scale 2.
        let b = [1.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let delta = lora_delta(&b, &a, &[2, 2], &[2, 3], 2.0).unwrap();
        assert_eq!(delta, vec![2.0, 4.0, 6.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn matvec_is_the_lstm_bias_update() {
        // weight_ih @ decoder_input
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = matvec(&w, 2, 3, &[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(v, vec![4.0, 10.0]);
    }

    #[test]
    fn frame_insert_is_a_residual_around_the_depthwise_mix() {
        let device = burn::tensor::Device::flex();
        let eye = [1.0f32, 0.0, 0.0, 1.0];
        let block = FrameInsert {
            ln_weight: Tensor::<1>::from_floats([1.0, 1.0], &device),
            ln_bias: Tensor::<1>::from_floats([0.0, 0.0], &device),
            down_weight: Tensor::<1>::from_floats(eye.as_slice(), &device).reshape([2, 2]),
            down_bias: Tensor::<1>::from_floats([0.0, 0.0], &device),
            mix_weight: Tensor::<1>::from_floats([1.0, 1.0], &device).reshape([2, 1, 1]),
            mix_bias: Tensor::<1>::from_floats([0.0, 0.0], &device),
            up_weight: Tensor::<1>::from_floats(eye.as_slice(), &device).reshape([2, 2]),
            up_bias: Tensor::<1>::from_floats([0.0, 0.0], &device),
            eps: 1e-5,
        };
        let x = Tensor::<1>::from_floats([1.0, 3.0], &device).reshape([1, 1, 2]);
        let y = block.forward(x).into_data().to_vec::<f32>().unwrap();
        // LayerNorm([1, 3]) = [-1, 1]. GELU keeps the residual off the input.
        assert!((y[0] - (1.0 + gelu_ref(-1.0))).abs() < 1e-5, "{y:?}");
        assert!((y[1] - (3.0 + gelu_ref(1.0))).abs() < 1e-5, "{y:?}");
    }

    fn gelu_ref(x: f32) -> f32 {
        // Burn's gelu is the erf form, the same one PyTorch uses by default.
        0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
    }
}
