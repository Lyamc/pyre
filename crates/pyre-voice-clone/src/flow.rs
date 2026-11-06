//! Mean-flow token-to-mel (Nano `S3Gen(meanflow=True)`).
//!
//! Upstream is `CausalMaskedDiffWithXvec` plus `ConditionalDecoder` in
//! chatterbox `s3gen/flow.py`, `decoder.py`, and `flow_matching.py`
//! (CosyVoice, Apache-2.0). `generate()` asks for `n_cfm_timesteps=2`.
//! Distilled mean-flow uses `basic_euler`: `t` goes `0 → 0.5 → 1` in a
//! straight line, and classifier-free guidance is not applied. The 10-step
//! cosine schedule and `inference_cfg_rate` belong to the non-meanflow
//! checkpoint, which Nano does not load.
//!
//! The noise is a normal sample. The prompt frames and the generated frames
//! are drawn separately (`torch.randn_like(mu)` then the tail is replaced).
//! Callers pass both, so a test can replay PyTorch's draw. [`SplitMix`] is
//! available when the caller wants a repeatable draw of its own. It is not
//! PyTorch's RNG.
//!
//! Layout: conformer activations are `[time, 512]`. Conformer layer-norm eps
//! is `1e-12`; the input embed and `after_norm` use `1e-5`. Estimator convs
//! are `[channel, time]`. The UNet is one resolution (`channels = [256]`):
//! causal convs pad on the left only, 4 transformer blocks on the way down,
//! 12 in the middle, 4 on the way up. Attention scale is `64^-0.5`. The FF
//! GELU is the erf one, not GPT-2 `gelu_new`.
//!
//! CAMPPlus and the 24 kHz prompt mel are computed by the caller
//! ([`crate::camp`], [`crate::frontend`]) and passed in. HiFT turns the
//! returned mel into samples. [`crate::text_to_cloned_speech_ready`] stays
//! false until that waveform is auditioned.

use crate::s3::{discover_s3, S3_FILENAME};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

const D: usize = 512;
const H: usize = 8;
const HD: usize = 64;
const MEL: usize = 80;
const SPK: usize = 192;
const VOCAB: usize = 6561;
const TIME: usize = 1024;

/// Silence token appended three times before the vocoder (`S3GEN_SIL`).
pub const S3GEN_SIL: u32 = 4299;

pub struct Flow {
    tok: Vec<f32>,
    spk_w: Vec<f32>,
    spk_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    enc: Encoder,
    est: Estimator,
}

struct Encoder {
    embed: Embed,
    la1_w: Vec<f32>,
    la1_b: Vec<f32>,
    la2_w: Vec<f32>,
    la2_b: Vec<f32>,
    layers: Vec<Conf>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    up_embed: Embed,
    up_layers: Vec<Conf>,
    after_w: Vec<f32>,
    after_b: Vec<f32>,
}

struct Embed {
    w: Vec<f32>,
    b: Vec<f32>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
}

struct Conf {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos_w: Vec<f32>,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    mha_w: Vec<f32>,
    mha_b: Vec<f32>,
    ff_w: Vec<f32>,
    ff_b: Vec<f32>,
    w1: Linear,
    w2: Linear,
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    din: usize,
    dout: usize,
}

struct Estimator {
    t1: Linear,
    t2: Linear,
    mixer: Vec<f32>,
    down: Res,
    down_tr: Vec<Tr>,
    down_conv: Conv,
    mids: Vec<(Res, Vec<Tr>)>,
    up: Res,
    up_tr: Vec<Tr>,
    up_conv: Conv,
    fin_w: Vec<f32>,
    fin_b: Vec<f32>,
    fin_ln_w: Vec<f32>,
    fin_ln_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
}

struct Res {
    b1_w: Vec<f32>,
    b1_b: Vec<f32>,
    b1_ln_w: Vec<f32>,
    b1_ln_b: Vec<f32>,
    b2_w: Vec<f32>,
    b2_b: Vec<f32>,
    b2_ln_w: Vec<f32>,
    b2_ln_b: Vec<f32>,
    mlp_w: Vec<f32>,
    mlp_b: Vec<f32>,
    res_w: Vec<f32>,
    res_b: Vec<f32>,
    cin: usize,
    cout: usize,
}

struct Tr {
    n1_w: Vec<f32>,
    n1_b: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    n3_w: Vec<f32>,
    n3_b: Vec<f32>,
    p_w: Vec<f32>,
    p_b: Vec<f32>,
    f_w: Vec<f32>,
    f_b: Vec<f32>,
}

struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    k: usize,
}

/// Repeatable normal source. Not PyTorch's generator.
pub struct SplitMix {
    state: u64,
}

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        let u2 = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        let u1 = u1.clamp(1e-7, 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

impl Flow {
    pub fn discover() -> Option<PathBuf> {
        discover_s3()
    }

    pub fn filename() -> &'static str {
        S3_FILENAME
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("safetensors")?;
        let enc = Encoder {
            embed: embed(&st, "flow.encoder.embed")?,
            la1_w: named(
                &st,
                "flow.encoder.pre_lookahead_layer.conv1.weight",
                &[D, D, 4],
            )?,
            la1_b: named(&st, "flow.encoder.pre_lookahead_layer.conv1.bias", &[D])?,
            la2_w: named(
                &st,
                "flow.encoder.pre_lookahead_layer.conv2.weight",
                &[D, D, 3],
            )?,
            la2_b: named(&st, "flow.encoder.pre_lookahead_layer.conv2.bias", &[D])?,
            layers: (0..6)
                .map(|i| conf(&st, &format!("flow.encoder.encoders.{i}")))
                .collect::<Result<_>>()?,
            up_w: named(&st, "flow.encoder.up_layer.conv.weight", &[D, D, 5])?,
            up_b: named(&st, "flow.encoder.up_layer.conv.bias", &[D])?,
            up_embed: embed(&st, "flow.encoder.up_embed")?,
            up_layers: (0..4)
                .map(|i| conf(&st, &format!("flow.encoder.up_encoders.{i}")))
                .collect::<Result<_>>()?,
            after_w: named(&st, "flow.encoder.after_norm.weight", &[D])?,
            after_b: named(&st, "flow.encoder.after_norm.bias", &[D])?,
        };
        let est = Estimator {
            t1: linear(&st, "flow.decoder.estimator.time_mlp.linear_1", 320, TIME)?,
            t2: linear(&st, "flow.decoder.estimator.time_mlp.linear_2", TIME, TIME)?,
            mixer: named(
                &st,
                "flow.decoder.estimator.time_embed_mixer.weight",
                &[TIME, TIME * 2],
            )?,
            down: res(&st, "flow.decoder.estimator.down_blocks.0.0", 320, 256)?,
            down_tr: (0..4)
                .map(|i| tr(&st, &format!("flow.decoder.estimator.down_blocks.0.1.{i}")))
                .collect::<Result<_>>()?,
            down_conv: conv_named(&st, "flow.decoder.estimator.down_blocks.0.2", 256, 256, 3)?,
            mids: (0..12)
                .map(|i| {
                    let p = format!("flow.decoder.estimator.mid_blocks.{i}");
                    Ok((
                        res(&st, &format!("{p}.0"), 256, 256)?,
                        (0..4)
                            .map(|j| tr(&st, &format!("{p}.1.{j}")))
                            .collect::<Result<_>>()?,
                    ))
                })
                .collect::<Result<_>>()?,
            up: res(&st, "flow.decoder.estimator.up_blocks.0.0", 512, 256)?,
            up_tr: (0..4)
                .map(|i| tr(&st, &format!("flow.decoder.estimator.up_blocks.0.1.{i}")))
                .collect::<Result<_>>()?,
            up_conv: conv_named(&st, "flow.decoder.estimator.up_blocks.0.2", 256, 256, 3)?,
            fin_w: named(
                &st,
                "flow.decoder.estimator.final_block.block.0.weight",
                &[256, 256, 3],
            )?,
            fin_b: named(
                &st,
                "flow.decoder.estimator.final_block.block.0.bias",
                &[256],
            )?,
            fin_ln_w: named(
                &st,
                "flow.decoder.estimator.final_block.block.2.weight",
                &[256],
            )?,
            fin_ln_b: named(
                &st,
                "flow.decoder.estimator.final_block.block.2.bias",
                &[256],
            )?,
            proj_w: named(
                &st,
                "flow.decoder.estimator.final_proj.weight",
                &[MEL, 256, 1],
            )?,
            proj_b: named(&st, "flow.decoder.estimator.final_proj.bias", &[MEL])?,
        };
        Ok(Self {
            tok: named(&st, "flow.input_embedding.weight", &[VOCAB, D])?,
            spk_w: named(&st, "flow.spk_embed_affine_layer.weight", &[MEL, SPK])?,
            spk_b: named(&st, "flow.spk_embed_affine_layer.bias", &[MEL])?,
            proj_w: named(&st, "flow.encoder_proj.weight", &[MEL, D])?,
            proj_b: named(&st, "flow.encoder_proj.bias", &[MEL])?,
            enc,
            est,
        })
    }

    /// Prompt mel is `[prompt_frames, 80]`, time-major, and `prompt_frames` must
    /// be twice the prompt-token count. `speech` is the new tokens only.
    /// Noises are `[80, time]`. The returned mel is the generated tail, `[80, 2 * speech]`.
    pub fn mel_from_tokens(
        &self,
        prompt_tokens: &[u32],
        speech_tokens: &[u32],
        prompt_mel: &[f32],
        speaker: &[f32],
        noise_full: &[f32],
        noise_tail: &[f32],
    ) -> Result<FlowStep> {
        if speaker.len() != SPK {
            bail!(
                "speaker embedding is {}, CAMPPlus emits {SPK}",
                speaker.len()
            );
        }
        let n_prompt = prompt_tokens.len();
        let n_speech = speech_tokens.len();
        if prompt_mel.len() != n_prompt * 2 * MEL {
            bail!(
                "prompt mel {} is not 2 * {} tokens * {MEL}",
                prompt_mel.len() / MEL.max(1),
                n_prompt
            );
        }
        let mut ids = Vec::with_capacity(n_prompt + n_speech);
        ids.extend_from_slice(prompt_tokens);
        ids.extend_from_slice(speech_tokens);
        for id in &ids {
            if *id >= VOCAB as u32 {
                bail!("flow token {id} is outside 0..{VOCAB}");
            }
        }
        let total = ids.len() * 2;
        let gen = n_speech * 2;
        if noise_full.len() != MEL * total || noise_tail.len() != MEL * gen {
            bail!("noise lengths do not match {total} and {gen} frames");
        }
        let mut x = Vec::with_capacity(ids.len() * D);
        for id in &ids {
            let s = *id as usize * D;
            x.extend_from_slice(&self.tok[s..s + D]);
        }
        let h = self.enc.forward(&x, ids.len());
        let mut mu_tc = linear_tc(&self.proj_w, &self.proj_b, &h, total, D, MEL);
        let mu = transpose(&mu_tc, total, MEL);
        let _ = &mut mu_tc;
        let mut cond = vec![0.0; MEL * total];
        // prompt_mel is [frames, 80]. cond is [80, time].
        for t in 0..n_prompt * 2 {
            for c in 0..MEL {
                cond[c * total + t] = prompt_mel[t * MEL + c];
            }
        }
        let mut spk = speaker.to_vec();
        l2(&mut spk);
        let spk = linear_tc(&self.spk_w, &self.spk_b, &spk, 1, SPK, MEL);
        let mut z = noise_full.to_vec();
        let prompt_frames = n_prompt * 2;
        for c in 0..MEL {
            for t in 0..gen {
                z[c * total + t + prompt_frames] = noise_tail[c * gen + t];
            }
        }
        let mut mel_ct = z;
        let mut dx0 = None;
        for step in 0..2 {
            let t0 = step as f32 * 0.5;
            let r0 = t0 + 0.5;
            let dx = self.est.forward(&mel_ct, &mu, &spk, &cond, total, t0, r0);
            if step == 0 {
                dx0 = Some(dx.clone());
            }
            for (a, b) in mel_ct.iter_mut().zip(&dx) {
                *a += 0.5 * *b;
            }
        }
        let mut out = vec![0.0; MEL * gen];
        for c in 0..MEL {
            out[c * gen..(c + 1) * gen]
                .copy_from_slice(&mel_ct[c * total + prompt_frames..(c + 1) * total]);
        }
        Ok(FlowStep {
            mu,
            dx0: dx0.unwrap_or_default(),
            mel: out,
        })
    }
}

pub struct FlowStep {
    pub mu: Vec<f32>,
    pub dx0: Vec<f32>,
    pub mel: Vec<f32>,
}

impl Encoder {
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (mut h, pos) = self.embed.forward(x, t);
        h = lookahead(&h, t, &self.la1_w, &self.la1_b, &self.la2_w, &self.la2_b);
        for layer in &self.layers {
            h = layer.forward(&h, t, &pos);
        }
        h = upsample(&h, t, &self.up_w, &self.up_b);
        let t2 = t * 2;
        let (mut h, pos) = self.up_embed.forward(&h, t2);
        for layer in &self.up_layers {
            h = layer.forward(&h, t2, &pos);
        }
        layernorm(&h, t2, D, &self.after_w, &self.after_b, 1e-5)
    }
}

impl Embed {
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let y = linear_tc(&self.w, &self.b, x, t, D, D);
        let y = layernorm(&y, t, D, &self.ln_w, &self.ln_b, 1e-5);
        let scale = (D as f32).sqrt();
        let y: Vec<f32> = y.into_iter().map(|v| v * scale).collect();
        (y, espnet_pos(t))
    }
}

impl Conf {
    fn forward(&self, x: &[f32], t: usize, pos: &[f32]) -> Vec<f32> {
        let h = layernorm(x, t, D, &self.mha_w, &self.mha_b, 1e-12);
        let att = rel_attn(
            &self.q,
            &self.k,
            &self.v,
            &self.o,
            &self.pos_w,
            &self.bias_u,
            &self.bias_v,
            &h,
            t,
            pos,
        );
        let mut x = x.to_vec();
        for (a, b) in x.iter_mut().zip(&att) {
            *a += *b;
        }
        let h = layernorm(&x, t, D, &self.ff_w, &self.ff_b, 1e-12);
        let mid = self.w1.apply(&h, t);
        let mid: Vec<f32> = mid.into_iter().map(silu).collect();
        let ff = self.w2.apply(&mid, t);
        for (a, b) in x.iter_mut().zip(&ff) {
            *a += *b;
        }
        x
    }
}

impl Linear {
    fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        linear_tc(&self.w, &self.b, x, t, self.din, self.dout)
    }
}

impl Estimator {
    fn temb(&self, t: f32, r: f32) -> Vec<f32> {
        let a = self.t2.apply(&silu_vec(&self.t1.apply(&sinusoid(t), 1)), 1);
        let b = self.t2.apply(&silu_vec(&self.t1.apply(&sinusoid(r), 1)), 1);
        let mut cat = a;
        cat.extend(b);
        let mut y = vec![0.0; TIME];
        for o in 0..TIME {
            let mut acc = 0.0;
            for i in 0..TIME * 2 {
                acc += cat[i] * self.mixer[o * TIME * 2 + i];
            }
            y[o] = acc;
        }
        y
    }

    fn forward(
        &self,
        x: &[f32],
        mu: &[f32],
        spk: &[f32],
        cond: &[f32],
        t: usize,
        t0: f32,
        r0: f32,
    ) -> Vec<f32> {
        let temb = self.temb(t0, r0);
        let mut h = Vec::with_capacity(320 * t);
        for src in [x, mu] {
            h.extend_from_slice(src);
        }
        for _ in 0..t {
            h.extend_from_slice(spk);
        }
        // spk was appended as [80] repeated wrong: I extended t copies of the
        // whole vector, which is [80*t] but the layout must be [80, t] i.e.
        // channel-major with each channel constant. Fix below if the order is
        // channel-major already: spk is [80], repeating the vector t times is
        // [c0,c1,..][c0,c1,..] which is time-major. Rebuild.
        let _ = &mut h;
        h.truncate(160 * t);
        let mut spk_ct = vec![0.0; MEL * t];
        for c in 0..MEL {
            for ti in 0..t {
                spk_ct[c * t + ti] = spk[c];
            }
        }
        h.extend_from_slice(&spk_ct);
        h.extend_from_slice(cond);
        h = self.down.forward(&h, t, &temb);
        h = transformers(&h, 256, t, &self.down_tr);
        let skip = h.clone();
        h = causal(&self.down_conv, &h, t);
        for (res, trs) in &self.mids {
            h = res.forward(&h, t, &temb);
            h = transformers(&h, 256, t, trs);
        }
        let mut cat = Vec::with_capacity(512 * t);
        cat.extend_from_slice(&h);
        cat.extend_from_slice(&skip);
        h = self.up.forward(&cat, t, &temb);
        h = transformers(&h, 256, t, &self.up_tr);
        h = causal(&self.up_conv, &h, t);
        h = causal(
            &Conv {
                w: self.fin_w.clone(),
                b: self.fin_b.clone(),
                cin: 256,
                cout: 256,
                k: 3,
            },
            &h,
            t,
        );
        h = ln_ct(&h, 256, t, &self.fin_ln_w, &self.fin_ln_b);
        mish_inplace(&mut h);
        conv1d(&self.proj_w, &self.proj_b, &h, 256, MEL, t, 1, 0, 1)
    }
}

impl Res {
    fn forward(&self, x: &[f32], t: usize, temb: &[f32]) -> Vec<f32> {
        let mut h = causal(
            &Conv {
                w: self.b1_w.clone(),
                b: self.b1_b.clone(),
                cin: self.cin,
                cout: self.cout,
                k: 3,
            },
            x,
            t,
        );
        h = ln_ct(&h, self.cout, t, &self.b1_ln_w, &self.b1_ln_b);
        mish_inplace(&mut h);
        let bias = linear_tc(
            &self.mlp_w,
            &self.mlp_b,
            &mish_vec(temb),
            1,
            TIME,
            self.cout,
        );
        for c in 0..self.cout {
            for ti in 0..t {
                h[c * t + ti] += bias[c];
            }
        }
        h = causal(
            &Conv {
                w: self.b2_w.clone(),
                b: self.b2_b.clone(),
                cin: self.cout,
                cout: self.cout,
                k: 3,
            },
            &h,
            t,
        );
        h = ln_ct(&h, self.cout, t, &self.b2_ln_w, &self.b2_ln_b);
        mish_inplace(&mut h);
        let skip = conv1d(&self.res_w, &self.res_b, x, self.cin, self.cout, t, 1, 0, 1);
        for (a, b) in h.iter_mut().zip(&skip) {
            *a += *b;
        }
        h
    }
}

fn transformers(x: &[f32], c: usize, t: usize, blocks: &[Tr]) -> Vec<f32> {
    let mut h = transpose(x, c, t);
    for b in blocks {
        h = b.forward(&h, t);
    }
    transpose(&h, t, c)
}

impl Tr {
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let n = layernorm(x, t, 256, &self.n1_w, &self.n1_b, 1e-5);
        let att = attn256(&self.q, &self.k, &self.v, &self.o_w, &self.o_b, &n, t);
        let mut x = x.to_vec();
        for (a, b) in x.iter_mut().zip(&att) {
            *a += *b;
        }
        let n = layernorm(&x, t, 256, &self.n3_w, &self.n3_b, 1e-5);
        let mut mid = linear_tc(&self.p_w, &self.p_b, &n, t, 256, 1024);
        for v in &mut mid {
            *v = gelu_erf(*v);
        }
        let ff = linear_tc(&self.f_w, &self.f_b, &mid, t, 1024, 256);
        for (a, b) in x.iter_mut().zip(&ff) {
            *a += *b;
        }
        x
    }
}

fn attn256(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ow: &[f32],
    ob: &[f32],
    x: &[f32],
    t: usize,
) -> Vec<f32> {
    let qq = linear_tc(q, &vec![0.0; 512], x, t, 256, 512);
    let kk = linear_tc(k, &vec![0.0; 512], x, t, 256, 512);
    let vv = linear_tc(v, &vec![0.0; 512], x, t, 256, 512);
    let scale = (HD as f32).powf(-0.5);
    let mut ctx = vec![0.0; t * 512];
    for head in 0..H {
        let mut scores = vec![0.0; t * t];
        for i in 0..t {
            for j in 0..t {
                let mut dot = 0.0;
                for d in 0..HD {
                    dot += qq[(i * 512) + head * HD + d] * kk[(j * 512) + head * HD + d];
                }
                scores[i * t + j] = dot * scale;
            }
        }
        for i in 0..t {
            softmax_inplace(&mut scores[i * t..(i + 1) * t]);
        }
        for i in 0..t {
            for d in 0..HD {
                let mut acc = 0.0;
                for j in 0..t {
                    acc += scores[i * t + j] * vv[(j * 512) + head * HD + d];
                }
                ctx[i * 512 + head * HD + d] = acc;
            }
        }
    }
    linear_tc(ow, ob, &ctx, t, 512, 256)
}

fn rel_attn(
    q: &Linear,
    k: &Linear,
    v: &Linear,
    o: &Linear,
    pos_w: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    x: &[f32],
    t: usize,
    pos: &[f32],
) -> Vec<f32> {
    let qq = q.apply(x, t);
    let kk = k.apply(x, t);
    let vv = v.apply(x, t);
    let plen = pos.len() / D;
    let pp = linear_tc(pos_w, &vec![0.0; D], pos, plen, D, D);
    let cols = plen;
    let mut bd = vec![0.0; H * t * cols];
    let mut ac = vec![0.0; H * t * t];
    for head in 0..H {
        for i in 0..t {
            for j in 0..t {
                let mut dot = 0.0;
                for d in 0..HD {
                    let qv = qq[i * D + head * HD + d] + bias_u[head * HD + d];
                    dot += qv * kk[j * D + head * HD + d];
                }
                ac[(head * t + i) * t + j] = dot;
            }
            for j in 0..cols {
                let mut dot = 0.0;
                for d in 0..HD {
                    let qv = qq[i * D + head * HD + d] + bias_v[head * HD + d];
                    dot += qv * pp[j * D + head * HD + d];
                }
                bd[(head * t + i) * cols + j] = dot;
            }
        }
    }
    let bd = rel_shift(&bd, t, cols);
    let scale = (HD as f32).sqrt();
    let mut ctx = vec![0.0; t * D];
    for head in 0..H {
        for i in 0..t {
            let mut row = vec![0.0; t];
            for j in 0..t {
                row[j] = (ac[(head * t + i) * t + j] + bd[(head * t + i) * t + j]) / scale;
            }
            softmax_inplace(&mut row);
            for d in 0..HD {
                let mut acc = 0.0;
                for j in 0..t {
                    acc += row[j] * vv[j * D + head * HD + d];
                }
                ctx[i * D + head * HD + d] = acc;
            }
        }
    }
    o.apply(&ctx, t)
}

fn rel_shift(x: &[f32], t: usize, cols: usize) -> Vec<f32> {
    let heads = H;
    let out_c = cols / 2 + 1;
    let padded_cols = cols + 1;
    let mut y = vec![0.0; heads * t * out_c];
    for h in 0..heads {
        let xh = &x[h * t * cols..(h + 1) * t * cols];
        for p in 0..(t * cols) {
            let lin = p + t;
            let t1 = lin / padded_cols;
            let i = lin % padded_cols;
            let val = if i == 0 { 0.0 } else { xh[t1 * cols + (i - 1)] };
            let row = p / cols;
            let col = p % cols;
            if col < out_c {
                y[(h * t + row) * out_c + col] = val;
            }
        }
    }
    y
}

fn lookahead(x: &[f32], t: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let ct = transpose(x, t, D);
    let mut padded = vec![0.0; D * (t + 3)];
    for c in 0..D {
        padded[c * (t + 3)..c * (t + 3) + t].copy_from_slice(&ct[c * t..(c + 1) * t]);
    }
    let mut h = conv1d(w1, b1, &padded, D, D, t + 3, 1, 0, 4);
    for v in &mut h {
        *v = leaky(*v, 0.01);
    }
    let mut padded = vec![0.0; D * (t + 2)];
    for c in 0..D {
        padded[c * (t + 2) + 2..c * (t + 2) + 2 + t].copy_from_slice(&h[c * t..(c + 1) * t]);
    }
    let h = conv1d(w2, b2, &padded, D, D, t + 2, 1, 0, 3);
    let h = transpose(&h, D, t);
    let mut y = x.to_vec();
    for (a, b) in y.iter_mut().zip(&h) {
        *a += *b;
    }
    y
}

fn upsample(x: &[f32], t: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let ct = transpose(x, t, D);
    let t2 = t * 2;
    let mut rep = vec![0.0; D * (t2 + 4)];
    for c in 0..D {
        for i in 0..t {
            rep[c * (t2 + 4) + 4 + 2 * i] = ct[c * t + i];
            rep[c * (t2 + 4) + 4 + 2 * i + 1] = ct[c * t + i];
        }
    }
    let h = conv1d(w, b, &rep, D, D, t2 + 4, 1, 0, 5);
    transpose(&h, D, t2)
}

fn espnet_pos(t: usize) -> Vec<f32> {
    let mut pos = vec![0.0; t * D];
    let mut neg = vec![0.0; t * D];
    for i in 0..t {
        for k in (0..D).step_by(2) {
            let div = (-(10000f32.ln()) / D as f32 * k as f32).exp();
            let p = i as f32 * div;
            pos[i * D + k] = p.sin();
            pos[i * D + k + 1] = p.cos();
            neg[i * D + k] = (-p).sin();
            neg[i * D + k + 1] = (-p).cos();
        }
    }
    // flip positive, drop neg[0], cat. Result length 2t-1.
    let mut pe = Vec::with_capacity((2 * t - 1) * D);
    for i in (0..t).rev() {
        pe.extend_from_slice(&pos[i * D..(i + 1) * D]);
    }
    for i in 1..t {
        pe.extend_from_slice(&neg[i * D..(i + 1) * D]);
    }
    let mid = pe.len() / D / 2;
    // position_encoding slice: pe[:, mid - t + 1 : mid + t] when len is 2t-1,
    // mid = t-1, slice is the whole buffer.
    let start = mid + 1 - t;
    let end = mid + t;
    pe[start * D..end * D].to_vec()
}

fn sinusoid(v: f32) -> Vec<f32> {
    let half = 160;
    let freq = (10000f32).ln() / (half - 1) as f32;
    let mut e = vec![0.0; 320];
    for i in 0..half {
        let ang = 1000.0 * v * (-freq * i as f32).exp();
        e[i] = ang.sin();
        e[half + i] = ang.cos();
    }
    e
}

fn linear_tc(w: &[f32], b: &[f32], x: &[f32], t: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0.0; t * dout];
    for ti in 0..t {
        for o in 0..dout {
            let mut acc = b[o];
            let row = &x[ti * din..(ti + 1) * din];
            let ww = &w[o * din..(o + 1) * din];
            for i in 0..din {
                acc += row[i] * ww[i];
            }
            y[ti * dout + o] = acc;
        }
    }
    y
}

fn layernorm(x: &[f32], t: usize, c: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; t * c];
    for ti in 0..t {
        let row = &x[ti * c..(ti + 1) * c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            y[ti * c + i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    }
    y
}

fn ln_ct(x: &[f32], c: usize, t: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let tc = transpose(x, c, t);
    transpose(&layernorm(&tc, t, c, w, b, 1e-5), t, c)
}

fn conv1d(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    cin: usize,
    cout: usize,
    t_in: usize,
    stride: usize,
    pad: usize,
    k: usize,
) -> Vec<f32> {
    let t_out = (t_in + 2 * pad - k) / stride + 1;
    let mut y = vec![0.0; cout * t_out];
    for oc in 0..cout {
        for ot in 0..t_out {
            let mut acc = b[oc];
            for ic in 0..cin {
                for kk in 0..k {
                    let it = ot * stride + kk;
                    if it < pad {
                        continue;
                    }
                    let it = it - pad;
                    if it >= t_in {
                        continue;
                    }
                    acc += x[ic * t_in + it] * w[((oc * cin + ic) * k) + kk];
                }
            }
            y[oc * t_out + ot] = acc;
        }
    }
    y
}

fn causal(conv: &Conv, x: &[f32], t: usize) -> Vec<f32> {
    let mut padded = vec![0.0; conv.cin * (t + conv.k - 1)];
    for c in 0..conv.cin {
        padded[c * (t + conv.k - 1) + conv.k - 1..c * (t + conv.k - 1) + conv.k - 1 + t]
            .copy_from_slice(&x[c * t..(c + 1) * t]);
    }
    conv1d(
        &conv.w,
        &conv.b,
        &padded,
        conv.cin,
        conv.cout,
        t + conv.k - 1,
        1,
        0,
        conv.k,
    )
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().copied().map(silu).collect()
}
fn leaky(v: f32, s: f32) -> f32 {
    if v < 0.0 {
        v * s
    } else {
        v
    }
}
fn mish(v: f32) -> f32 {
    let sp = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
    v * sp.tanh()
}
fn mish_inplace(x: &mut [f32]) {
    for v in x {
        *v = mish(*v);
    }
}
fn mish_vec(x: &[f32]) -> Vec<f32> {
    x.iter().copied().map(mish).collect()
}
fn gelu_erf(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2))
}
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    s * (1.0 - p * (-a * a).exp())
}
fn l2(x: &mut [f32]) {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for v in x {
        *v /= n;
    }
}

fn named(st: &safetensors::SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = st.tensor(name).with_context(|| name.to_string())?;
    if view.shape() != shape {
        bail!("{name} shape {:?} != {shape:?}", view.shape());
    }
    if view.dtype() != safetensors::Dtype::F32 {
        bail!("{name} is not f32");
    }
    let mut out = Vec::with_capacity(view.data().len() / 4);
    for c in view.data().chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(out)
}

fn linear(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    din: usize,
    dout: usize,
) -> Result<Linear> {
    Ok(Linear {
        w: named(st, &format!("{prefix}.weight"), &[dout, din])?,
        b: named(st, &format!("{prefix}.bias"), &[dout])?,
        din,
        dout,
    })
}

fn embed(st: &safetensors::SafeTensors<'_>, prefix: &str) -> Result<Embed> {
    Ok(Embed {
        w: named(st, &format!("{prefix}.out.0.weight"), &[D, D])?,
        b: named(st, &format!("{prefix}.out.0.bias"), &[D])?,
        ln_w: named(st, &format!("{prefix}.out.1.weight"), &[D])?,
        ln_b: named(st, &format!("{prefix}.out.1.bias"), &[D])?,
    })
}

fn conf(st: &safetensors::SafeTensors<'_>, prefix: &str) -> Result<Conf> {
    Ok(Conf {
        q: linear(st, &format!("{prefix}.self_attn.linear_q"), D, D)?,
        k: linear(st, &format!("{prefix}.self_attn.linear_k"), D, D)?,
        v: linear(st, &format!("{prefix}.self_attn.linear_v"), D, D)?,
        o: linear(st, &format!("{prefix}.self_attn.linear_out"), D, D)?,
        pos_w: named(
            st,
            &format!("{prefix}.self_attn.linear_pos.weight"),
            &[D, D],
        )?,
        bias_u: named(st, &format!("{prefix}.self_attn.pos_bias_u"), &[H, HD])?,
        bias_v: named(st, &format!("{prefix}.self_attn.pos_bias_v"), &[H, HD])?,
        mha_w: named(st, &format!("{prefix}.norm_mha.weight"), &[D])?,
        mha_b: named(st, &format!("{prefix}.norm_mha.bias"), &[D])?,
        ff_w: named(st, &format!("{prefix}.norm_ff.weight"), &[D])?,
        ff_b: named(st, &format!("{prefix}.norm_ff.bias"), &[D])?,
        w1: linear(st, &format!("{prefix}.feed_forward.w_1"), D, 2048)?,
        w2: linear(st, &format!("{prefix}.feed_forward.w_2"), 2048, D)?,
    })
}

fn res(st: &safetensors::SafeTensors<'_>, prefix: &str, cin: usize, cout: usize) -> Result<Res> {
    Ok(Res {
        b1_w: named(
            st,
            &format!("{prefix}.block1.block.0.weight"),
            &[cout, cin, 3],
        )?,
        b1_b: named(st, &format!("{prefix}.block1.block.0.bias"), &[cout])?,
        b1_ln_w: named(st, &format!("{prefix}.block1.block.2.weight"), &[cout])?,
        b1_ln_b: named(st, &format!("{prefix}.block1.block.2.bias"), &[cout])?,
        b2_w: named(
            st,
            &format!("{prefix}.block2.block.0.weight"),
            &[cout, cout, 3],
        )?,
        b2_b: named(st, &format!("{prefix}.block2.block.0.bias"), &[cout])?,
        b2_ln_w: named(st, &format!("{prefix}.block2.block.2.weight"), &[cout])?,
        b2_ln_b: named(st, &format!("{prefix}.block2.block.2.bias"), &[cout])?,
        mlp_w: named(st, &format!("{prefix}.mlp.1.weight"), &[cout, TIME])?,
        mlp_b: named(st, &format!("{prefix}.mlp.1.bias"), &[cout])?,
        res_w: named(st, &format!("{prefix}.res_conv.weight"), &[cout, cin, 1])?,
        res_b: named(st, &format!("{prefix}.res_conv.bias"), &[cout])?,
        cin,
        cout,
    })
}

fn tr(st: &safetensors::SafeTensors<'_>, prefix: &str) -> Result<Tr> {
    Ok(Tr {
        n1_w: named(st, &format!("{prefix}.norm1.weight"), &[256])?,
        n1_b: named(st, &format!("{prefix}.norm1.bias"), &[256])?,
        q: named(st, &format!("{prefix}.attn1.to_q.weight"), &[512, 256])?,
        k: named(st, &format!("{prefix}.attn1.to_k.weight"), &[512, 256])?,
        v: named(st, &format!("{prefix}.attn1.to_v.weight"), &[512, 256])?,
        o_w: named(st, &format!("{prefix}.attn1.to_out.0.weight"), &[256, 512])?,
        o_b: named(st, &format!("{prefix}.attn1.to_out.0.bias"), &[256])?,
        n3_w: named(st, &format!("{prefix}.norm3.weight"), &[256])?,
        n3_b: named(st, &format!("{prefix}.norm3.bias"), &[256])?,
        p_w: named(st, &format!("{prefix}.ff.net.0.proj.weight"), &[1024, 256])?,
        p_b: named(st, &format!("{prefix}.ff.net.0.proj.bias"), &[1024])?,
        f_w: named(st, &format!("{prefix}.ff.net.2.weight"), &[256, 1024])?,
        f_b: named(st, &format!("{prefix}.ff.net.2.bias"), &[256])?,
    })
}

fn conv_named(
    st: &safetensors::SafeTensors<'_>,
    prefix: &str,
    cin: usize,
    cout: usize,
    k: usize,
) -> Result<Conv> {
    Ok(Conv {
        w: named(st, &format!("{prefix}.weight"), &[cout, cin, k])?,
        b: named(st, &format!("{prefix}.bias"), &[cout])?,
        cin,
        cout,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_normals_repeat() {
        let mut a = SplitMix::new(3);
        let mut b = SplitMix::new(3);
        let xs = a.fill(8);
        let ys = b.fill(8);
        assert_eq!(xs, ys);
        assert!(xs.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    #[ignore = "set PYRE_CHATTERBOX_S3 to s3gen_meanflow.safetensors and pass --ignored"]
    fn real_euler_matches_torch_on_fixed_noise() {
        let Some(path) = std::env::var_os("PYRE_CHATTERBOX_S3") else {
            eprintln!("PYRE_CHATTERBOX_S3 unset");
            return;
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
        let tokens = i32_file(&root.join("flow_tokens.i32"));
        let prompt = f32_file(&root.join("flow_prompt_mel.f32"));
        let speaker = f32_file(&root.join("flow_speaker.f32"));
        let noise_full = f32_file(&root.join("flow_noise_prompt.f32"));
        let noise_tail = f32_file(&root.join("flow_noise_gen.f32"));
        let mu_ref = f32_file(&root.join("flow_mu.f32"));
        let dx_ref = f32_file(&root.join("flow_dx0.f32"));
        let mel_ref = f32_file(&root.join("flow_mel.f32"));
        let flow = Flow::load(Path::new(&path)).unwrap();
        let mut x = Vec::new();
        for id in &tokens {
            let s = *id as usize * D;
            x.extend_from_slice(&flow.tok[s..s + D]);
        }
        let step = flow
            .mel_from_tokens(
                &tokens[..4],
                &tokens[4..],
                &prompt,
                &speaker,
                &noise_full,
                &noise_tail,
            )
            .unwrap();
        let mu_err = max_abs(&step.mu, &mu_ref);
        assert!(mu_err < 1e-3, "encoder mu max abs {mu_err}");
        let dx_err = max_abs(&step.dx0, &dx_ref);
        assert!(dx_err < 1e-3, "first euler step max abs {dx_err}");
        let mel_err = max_abs(&step.mel, &mel_ref);
        assert!(mel_err < 1e-3, "mel max abs {mel_err}");
        assert!(step.mel.iter().all(|v| v.is_finite()));
    }

    fn f32_file(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn i32_file(path: &Path) -> Vec<u32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "{} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
}
