//! Chatterbox `VoiceEncoder`: a 3-layer LSTM d-vector.
//!
//! Weights are `ResembleAI/chatterbox-nano` `ve.safetensors` (the same file the
//! Turbo and Multilingual checkpoints use). Layout is PyTorch's LSTM, batch first,
//! gate order input / forget / cell / output:
//!
//! ```text
//! i, f, g, o = split(W_ih x + b_ih + W_hh h + b_hh)
//! c = sigmoid(f) * c + sigmoid(i) * tanh(g)
//! h = sigmoid(o) * tanh(c)
//! ```
//!
//! The embedding is `L2(relu(proj(h_T of the last layer)))` on each 160-frame
//! partial, then the mean of those vectors, L2-normalized again. That is
//! `VoiceEncoder.inference` followed by the utterance reduction in
//! `embeds_from_wavs` (one clip, so the utterance embedding is the speaker
//! embedding).

use anyhow::{anyhow, bail, Context, Result};
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::Path;

use crate::mel::{self, N_MELS, PARTIAL_FRAMES};
use crate::verify::{self, SpeakerMatch};

pub const EMBED: usize = 256;
pub const HIDDEN: usize = 256;
pub const LAYERS: usize = 3;

struct LstmLayer {
    w_ih: Tensor<2>,
    w_hh: Tensor<2>,
    b_ih: Tensor<1>,
    b_hh: Tensor<1>,
    input: usize,
    hidden: usize,
}

impl LstmLayer {
    fn step(&self, x: Tensor<2>, h: Tensor<2>, c: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let gates = x
            .matmul(self.w_ih.clone().transpose())
            .add(self.b_ih.clone().unsqueeze())
            .add(h.matmul(self.w_hh.clone().transpose()))
            .add(self.b_hh.clone().unsqueeze());
        let b = gates.dims()[0];
        let hdim = self.hidden;
        let i = sigmoid(gates.clone().slice([0..b, 0..hdim]));
        let f = sigmoid(gates.clone().slice([0..b, hdim..2 * hdim]));
        let g = tanh(gates.clone().slice([0..b, 2 * hdim..3 * hdim]));
        let o = sigmoid(gates.slice([0..b, 3 * hdim..4 * hdim]));
        let c = f.mul(c).add(i.mul(g));
        let h = o.mul(tanh(c.clone()));
        (h, c)
    }

    fn sequence(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, input] = x.dims();
        assert_eq!(input, self.input);
        let dev = x.device();
        let mut h = Tensor::<2>::zeros([b, self.hidden], &dev);
        let mut c = Tensor::<2>::zeros([b, self.hidden], &dev);
        let mut steps = Vec::with_capacity(t);
        for i in 0..t {
            let xt = x
                .clone()
                .slice([0..b, i..i + 1, 0..input])
                .reshape([b, input]);
            let (nh, nc) = self.step(xt, h, c);
            h = nh;
            c = nc;
            steps.push(h.clone().reshape([b, 1, self.hidden]));
        }
        Tensor::cat(steps, 1)
    }
}

/// One LSTM layer, row-major PyTorch layout: `w_ih` is `[4 * hidden, input]`.
#[derive(Clone)]
pub struct LstmParams {
    pub w_ih: Vec<f32>,
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
    pub input: usize,
    pub hidden: usize,
}

pub struct VoiceEncoder {
    layers: Vec<LstmLayer>,
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
    final_relu: bool,
    /// Loaded from the checkpoint. GE2E training scale. Not applied to embeddings.
    pub similarity_weight: f32,
    pub similarity_bias: f32,
    device: Device,
}

impl VoiceEncoder {
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// CPU device. This slice is intentionally not placed on the GPU Qwen-Next is using.
    pub fn cpu() -> Device {
        Device::flex()
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let device = Self::cpu();
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let input = if i == 0 { N_MELS } else { HIDDEN };
            layers.push(LstmLayer {
                w_ih: tensor2(
                    &st,
                    &format!("lstm.weight_ih_l{i}"),
                    4 * HIDDEN,
                    input,
                    &device,
                )?,
                w_hh: tensor2(
                    &st,
                    &format!("lstm.weight_hh_l{i}"),
                    4 * HIDDEN,
                    HIDDEN,
                    &device,
                )?,
                b_ih: tensor1(&st, &format!("lstm.bias_ih_l{i}"), 4 * HIDDEN, &device)?,
                b_hh: tensor1(&st, &format!("lstm.bias_hh_l{i}"), 4 * HIDDEN, &device)?,
                input,
                hidden: HIDDEN,
            });
        }
        Ok(Self {
            layers,
            proj_w: tensor2(&st, "proj.weight", EMBED, HIDDEN, &device)?,
            proj_b: tensor1(&st, "proj.bias", EMBED, &device)?,
            final_relu: true,
            similarity_weight: scalar(&st, "similarity_weight")?,
            similarity_bias: scalar(&st, "similarity_bias")?,
            device,
        })
    }

    pub fn from_params(
        layers: Vec<LstmParams>,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
        final_relu: bool,
    ) -> Result<Self> {
        if layers.is_empty() {
            bail!("voice encoder needs at least one LSTM layer");
        }
        let device = Self::cpu();
        let hidden = layers.last().unwrap().hidden;
        let embed = proj_b.len();
        if proj_w.len() != embed * hidden {
            bail!("proj weight {} != {embed} x {hidden}", proj_w.len());
        }
        let mut built = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let expect_in = if i == 0 {
                layer.input
            } else {
                layers[i - 1].hidden
            };
            if layer.input != expect_in {
                bail!("layer {i} input {} != {expect_in}", layer.input);
            }
            let gates = 4 * layer.hidden;
            if layer.w_ih.len() != gates * layer.input || layer.w_hh.len() != gates * layer.hidden {
                bail!("layer {i} weight shape");
            }
            if layer.b_ih.len() != gates || layer.b_hh.len() != gates {
                bail!("layer {i} bias shape");
            }
            built.push(LstmLayer {
                w_ih: from2(&layer.w_ih, gates, layer.input, &device),
                w_hh: from2(&layer.w_hh, gates, layer.hidden, &device),
                b_ih: from1(&layer.b_ih, &device),
                b_hh: from1(&layer.b_hh, &device),
                input: layer.input,
                hidden: layer.hidden,
            });
        }
        Ok(Self {
            layers: built,
            proj_w: from2(&proj_w, embed, hidden, &device),
            proj_b: from1(&proj_b, &device),
            final_relu,
            similarity_weight: 10.0,
            similarity_bias: -5.0,
            device,
        })
    }

    /// One utterance at `sample_rate` (linear-resampled to 16 kHz when it is not).
    /// Returns a 256-d unit vector for the real checkpoint, or `embed`-d for [`Self::from_params`].
    pub fn embed_wav(&self, wav: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let partials = mel::prepare_wav(wav, sample_rate);
        if partials.is_empty() {
            bail!("reference audio is empty after the 20 dB trim");
        }
        let n_mels = partials[0].len() / PARTIAL_FRAMES;
        self.embed_partials(&partials, n_mels)
    }

    /// `partials[i]` is one partial, time-major, length `frames * n_mels`.
    pub fn embed_partials(&self, partials: &[Vec<f32>], n_mels: usize) -> Result<Vec<f32>> {
        if partials.is_empty() {
            bail!("no mel partials");
        }
        let frames = partials[0].len() / n_mels;
        if frames == 0 || partials.iter().any(|p| p.len() != frames * n_mels) {
            bail!("partials are not a rectangular [n, {frames}, {n_mels}] batch");
        }
        let mut flat = Vec::with_capacity(partials.len() * frames * n_mels);
        for p in partials {
            flat.extend_from_slice(p);
        }
        let batch = from3(&flat, partials.len(), frames, n_mels, &self.device);
        let emb = self.forward_partials(batch);
        let mean = emb.mean_dim(0).squeeze_dim::<1>(0);
        to_vec1(l2_1(mean))
    }

    fn forward_partials(&self, mut x: Tensor<3>) -> Tensor<2> {
        for layer in &self.layers {
            x = layer.sequence(x);
        }
        let [b, t, h] = x.dims();
        let last = x.slice([0..b, t - 1..t, 0..h]).reshape([b, h]);
        let mut raw = last
            .matmul(self.proj_w.clone().transpose())
            .add(self.proj_b.clone().unsqueeze());
        if self.final_relu {
            raw = relu(raw);
        }
        l2_rows(raw)
    }

    pub fn compare(
        &self,
        reference: &[f32],
        sample_rate: u32,
        other: &[f32],
        other_rate: u32,
    ) -> Result<SpeakerMatch> {
        let a = self.embed_wav(reference, sample_rate)?;
        let b = self.embed_wav(other, other_rate)?;
        Ok(verify::match_speakers(&a, &b))
    }
}

fn from1(data: &[f32], device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

fn from2(data: &[f32], rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

fn from3(data: &[f32], b: usize, t: usize, c: usize, device: &Device) -> Tensor<3> {
    Tensor::<3>::from_data(TensorData::new(data.to_vec(), [b, t, c]), device)
}

fn l2_rows(x: Tensor<2>) -> Tensor<2> {
    let n = x
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1.0e-12);
    x.div(n)
}

fn l2_1(x: Tensor<1>) -> Tensor<1> {
    let n = x
        .clone()
        .powi_scalar(2)
        .sum_dim(0)
        .sqrt()
        .clamp_min(1.0e-12);
    x.div(n)
}

fn to_vec1(t: Tensor<1>) -> Result<Vec<f32>> {
    t.into_data()
        .try_to_vec::<f32>()
        .map_err(|e| anyhow!("tensor to vec: {e:?}"))
}

fn f32s(view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != safetensors::Dtype::F32 {
        bail!("expected f32");
    }
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn tensor2(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<Tensor<2>> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let shape = v.shape();
    if shape != [rows, cols] {
        bail!("{name}: shape {shape:?} != [{rows}, {cols}]");
    }
    Ok(from2(&f32s(v)?, rows, cols, device))
}

fn tensor1(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    n: usize,
    device: &Device,
) -> Result<Tensor<1>> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let shape = v.shape();
    if shape != [n] {
        bail!("{name}: shape {shape:?} != [{n}]");
    }
    Ok(from1(&f32s(v)?, device))
}

fn scalar(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<f32> {
    let v = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    let data = f32s(v)?;
    data.first().copied().context(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_forward(
        layers: &[LstmParams],
        proj_w: &[f32],
        proj_b: &[f32],
        relu_out: bool,
        x: &[f32],
        b: usize,
        t: usize,
        input: usize,
    ) -> Vec<f32> {
        let mut seq = x.to_vec();
        let mut width = input;
        for layer in layers {
            let mut next = vec![0f32; b * t * layer.hidden];
            for bi in 0..b {
                let mut h = vec![0f32; layer.hidden];
                let mut c = vec![0f32; layer.hidden];
                for ti in 0..t {
                    let xt = &seq[bi * t * width + ti * width..bi * t * width + (ti + 1) * width];
                    let mut gates = vec![0f32; 4 * layer.hidden];
                    for o in 0..4 * layer.hidden {
                        let mut s = layer.b_ih[o] + layer.b_hh[o];
                        for i in 0..layer.input {
                            s += layer.w_ih[o * layer.input + i] * xt[i];
                        }
                        for i in 0..layer.hidden {
                            s += layer.w_hh[o * layer.hidden + i] * h[i];
                        }
                        gates[o] = s;
                    }
                    let mut nh = vec![0f32; layer.hidden];
                    let mut nc = vec![0f32; layer.hidden];
                    for k in 0..layer.hidden {
                        let i = 1.0 / (1.0 + (-gates[k]).exp());
                        let f = 1.0 / (1.0 + (-gates[layer.hidden + k]).exp());
                        let g = gates[2 * layer.hidden + k].tanh();
                        let o = 1.0 / (1.0 + (-gates[3 * layer.hidden + k]).exp());
                        nc[k] = f * c[k] + i * g;
                        nh[k] = o * nc[k].tanh();
                    }
                    h = nh;
                    c = nc;
                    let dest = &mut next[bi * t * layer.hidden + ti * layer.hidden..];
                    dest[..layer.hidden].copy_from_slice(&h);
                }
            }
            seq = next;
            width = layer.hidden;
        }
        let hidden = layers.last().unwrap().hidden;
        let embed = proj_b.len();
        let mut raws = Vec::with_capacity(b * embed);
        for bi in 0..b {
            let h = &seq[bi * t * hidden + (t - 1) * hidden..bi * t * hidden + t * hidden];
            let mut row = vec![0f32; embed];
            for o in 0..embed {
                let mut s = proj_b[o];
                for i in 0..hidden {
                    s += proj_w[o * hidden + i] * h[i];
                }
                if relu_out {
                    s = s.max(0.0);
                }
                row[o] = s;
            }
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in &mut row {
                *v /= n;
            }
            raws.extend(row);
        }
        let mut mean = vec![0f32; embed];
        for bi in 0..b {
            for o in 0..embed {
                mean[o] += raws[bi * embed + o];
            }
        }
        for v in &mut mean {
            *v /= b as f32;
        }
        let n = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut mean {
            *v /= n;
        }
        mean
    }

    #[test]
    fn burn_lstm_matches_scalar_reference() {
        let layer = LstmParams {
            input: 3,
            hidden: 2,
            w_ih: vec![
                0.2, -0.1, 0.05, 0.1, 0.2, -0.3, -0.4, 0.3, 0.1, 0.05, -0.2, 0.4, 0.3, 0.1, -0.1,
                0.2, -0.05, 0.15, -0.2, 0.25, 0.1, 0.05, -0.3, 0.2,
            ],
            w_hh: vec![
                0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.1, 0.4, -0.3, 0.2, 0.1, -0.1, 0.05, -0.4, 0.2,
                0.1,
            ],
            b_ih: vec![0.01, -0.02, 0.03, -0.04, 0.05, -0.01, 0.02, -0.03],
            b_hh: vec![0.0, 0.01, -0.01, 0.02, 0.0, -0.02, 0.01, 0.0],
        };
        let proj_w = vec![0.4, -0.2, -0.5, 0.3, 0.1, 0.6];
        let proj_b = vec![0.05, -0.1, 0.0];
        let x = vec![
            0.5, -0.25, 0.1, 1.0, 0.0, -0.4, -0.2, 0.3, 0.8, 0.1, -0.1, 0.2,
        ];
        let enc =
            VoiceEncoder::from_params(vec![layer.clone()], proj_w.clone(), proj_b.clone(), true)
                .unwrap();
        let got = enc.embed_partials(&[x.clone()], 3).unwrap();
        let expect = scalar_forward(
            std::slice::from_ref(&layer),
            &proj_w,
            &proj_b,
            true,
            &x,
            1,
            4,
            3,
        );
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-5, "max abs {max}: burn {got:?} scalar {expect:?}");
    }

    fn ve_path() -> Option<std::path::PathBuf> {
        std::env::var_os("PYRE_CHATTERBOX_VE").map(std::path::PathBuf::from)
    }

    /// Real `ve.safetensors`. Ignored unless you pass `--ignored` and set
    /// `PYRE_CHATTERBOX_VE` to that file (MIT, `ResembleAI/chatterbox-nano`).
    #[test]
    #[ignore = "set PYRE_CHATTERBOX_VE to ve.safetensors and pass --ignored"]
    fn real_weights_separate_a_harmonic_from_noise() {
        let Some(path) = ve_path() else {
            eprintln!("PYRE_CHATTERBOX_VE unset; skipping real-weight voice encoder");
            return;
        };
        let enc = VoiceEncoder::load(&path).unwrap();
        assert!(enc.similarity_weight.is_finite() && enc.similarity_bias.is_finite());
        let vowel = mel::synth_harmonic(140.0, 3.0);
        let vowel_again = mel::synth_harmonic(140.0, 3.0);
        let other = mel::synth_harmonic(280.0, 3.0);
        let mut noise = vec![0f32; vowel.len()];
        let mut state = 0x1234_5678_u32;
        for s in &mut noise {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *s = (state >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1;
        }
        let a = enc.embed_wav(&vowel, 16_000).unwrap();
        let a2 = enc.embed_wav(&vowel_again, 16_000).unwrap();
        let b = enc.embed_wav(&other, 16_000).unwrap();
        let n = enc.embed_wav(&noise, 16_000).unwrap();
        assert_eq!(a.len(), EMBED);
        let norm = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
        let same = verify::cosine(&a, &a2);
        let cross = verify::cosine(&a, &b);
        let vs_noise = verify::cosine(&a, &n);
        assert!(same > 0.999, "identical clip cosine {same}");
        assert!(
            same > vs_noise,
            "noise should be farther than a repeated clip ({same} vs {vs_noise})"
        );
        assert!(cross.is_finite() && vs_noise.is_finite());
        let repeated = enc.compare(&vowel, 16_000, &vowel_again, 16_000).unwrap();
        assert!(repeated.accepts_clone, "cosine {}", repeated.cosine);
        eprintln!("real ve cosine same={same:.4} f0_280={cross:.4} noise={vs_noise:.4}");
    }

    /// Byte-level check against `tools/ve_reference.py` (librosa mel + NumPy LSTM).
    #[test]
    #[ignore = "set PYRE_CHATTERBOX_VE and PYRE_CHATTERBOX_VE_GOLDEN, pass --ignored"]
    fn real_weights_match_dev_reference() {
        let Some(path) = ve_path() else {
            eprintln!("PYRE_CHATTERBOX_VE unset; skipping reference compare");
            return;
        };
        let Some(golden) = std::env::var_os("PYRE_CHATTERBOX_VE_GOLDEN") else {
            eprintln!("PYRE_CHATTERBOX_VE_GOLDEN unset; skipping reference compare");
            return;
        };
        let raw = std::fs::read_to_string(golden).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let expect: Vec<f32> = v["vowel_140"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let enc = VoiceEncoder::load(&path).unwrap();
        let got = enc
            .embed_wav(&mel::synth_harmonic(140.0, 3.0), 16_000)
            .unwrap();
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("max abs vs librosa/numpy reference {max:.3e}");
        assert!(max < 1e-4, "max abs vs librosa/numpy reference {max}");
    }
}
