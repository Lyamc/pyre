use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig, conv::{Conv2d, Conv2dConfig}, PaddingConfig2d},
    tensor::{activation::relu, Tensor},
};
use crate::audio::parakeet_subsampled_frames;
use crate::model::conformer::{ConformerEncoder, ConformerEncoderConfig};
use crate::model::parakeet_decoder::{ParakeetDecoder, ParakeetDecoderCache, ParakeetDecoderConfig};

#[derive(Config, Debug)]
pub struct ParakeetConfig {
    pub encoder: ConformerEncoderConfig,
    pub n_vocab: usize,
    pub n_mels: usize,
}

impl ParakeetConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Parakeet {
        let encoder = self.encoder.init(device);
        
        let conv1 = Conv2dConfig::new([1, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_stride([2, 2])
            .init(device);
        let conv2 = Conv2dConfig::new([256, 256], [3, 3])
            .with_groups(256)
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_stride([2, 2])
            .init(device);
        let conv3 = Conv2dConfig::new([256, 256], [1, 1])
            .init(device);
        let conv4 = Conv2dConfig::new([256, 256], [3, 3])
            .with_groups(256)
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_stride([2, 2])
            .init(device);
        let conv5 = Conv2dConfig::new([256, 256], [1, 1])
            .init(device);

        let mr_fixed = (self.n_mels / 8).max(1);
        let pre_encode_out = LinearConfig::new(256 * mr_fixed, self.encoder.d_model).init(device);
        
        let d_joint = 640; 
        let final_proj = LinearConfig::new(self.encoder.d_model, d_joint).init(device);
        let is_ctc = self.n_vocab <= 1025;
        let ctc_in = if is_ctc { self.encoder.d_model } else { d_joint };
        let ctc_linear = LinearConfig::new(ctc_in, self.n_vocab).init(device);
        
        let joint_pred_bias = Tensor::zeros([d_joint], device);

        // Joint head is [hidden, vocab + n_durations]. Decoder embedding is vocab only
        // (v3: 8193, v2: 1025). CTC checkpoints have no duration slots and no LSTM.
        let decoder = if is_ctc {
            None
        } else {
            let decoder_vocab = self.n_vocab.saturating_sub(5).max(1);
            Some(
                ParakeetDecoderConfig {
                    vocab_size: decoder_vocab,
                    hidden_size: d_joint,
                    num_layers: 2,
                }
                .init(device),
            )
        };

        Parakeet { 
            conv1, conv2, conv3, conv4, conv5, 
            pre_encode_out, encoder, final_proj, ctc_linear, 
            joint_pred_bias: Param::from_tensor(joint_pred_bias), 
            decoder,
            mean: None,
            std: None,
            n_mels: self.n_mels,
        }
    }
}

#[derive(Module, Debug)]
pub struct Parakeet {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub conv3: Conv2d,
    pub conv4: Conv2d,
    pub conv5: Conv2d,
    pub pre_encode_out: Linear,
    pub encoder: ConformerEncoder,
    pub final_proj: Linear,
    pub ctc_linear: Linear,
    pub joint_pred_bias: Param<Tensor<1>>,
    pub decoder: Option<ParakeetDecoder>,
    pub mean: Option<Param<Tensor<2>>>,
    pub std: Option<Param<Tensor<2>>>,
    pub n_mels: usize,
}

impl Parakeet {
    pub fn encode_hidden(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.swap_dims(1, 2).unsqueeze::<4>();

        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);
        let x = burn::tensor::activation::relu(x);
        
        let x = self.conv4.forward(x);
        let x = self.conv5.forward(x);
        let x = burn::tensor::activation::relu(x);
        
        let [b, _, sr, mr] = x.dims();

        let x = x.permute([0, 2, 1, 3]);
        let [_, _, c, _] = x.dims();

        let mr_fixed = (self.n_mels / 8).max(1);
        let x = if mr > mr_fixed {
            x.slice([0..b, 0..sr, 0..c, 0..mr_fixed])
        } else if mr < mr_fixed {
            let pad = Tensor::zeros([b, sr, c, mr_fixed - mr], &x.device());
            Tensor::cat(vec![x, pad], 3)
        } else {
            x
        };

        let x = x.reshape([b, sr, c * mr_fixed]); 
        let x = self.pre_encode_out.forward(x);
        // HF ParakeetEncoder: hidden *= sqrt(d_model) when encoder_config.scale_input.
        // CTC 0.6B sets this true; TDT v2/v3 set it false.
        let x = if self.decoder.is_none() {
            let d_model = x.dims()[2] as f32;
            x.mul_scalar(d_model.sqrt())
        } else {
            x
        };
        self.encoder.forward(x)
    }

    pub fn encode(&self, x: Tensor<3>) -> Tensor<3> {
        self.final_proj.forward(self.encode_hidden(x))
    }

    pub fn joint_step(
        &self,
        enc_frame: Tensor<2>,
        dec_frame: Tensor<2>,
    ) -> Tensor<2> {
        let joint = relu(enc_frame + dec_frame);
        self.ctc_linear.forward(joint)
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        if self.decoder.is_none() {
            // CTC: Conv1d k=1 over encoder hidden (no joint / encoder_projector).
            return self.ctc_linear.forward(self.encode_hidden(x));
        }
        let enc = self.encode(x);
        let bias = self.joint_pred_bias.val().unsqueeze::<2>().unsqueeze::<3>();
        let x = relu(enc + bias);
        self.ctc_linear.forward(x)
    }

    pub fn decode_tdt_greedy(
        &self,
        mel: Tensor<3>,
        blank_token_id: usize,
        vocab_size: usize,
        duration_start: usize,
        num_durations: usize,
        start_token_id: usize,
        verbose: bool,
    ) -> Vec<usize> {
        let decoder = match &self.decoder {
            Some(d) => d,
            None => return Vec::new(),
        };

        let [_, _, mel_frames] = mel.dims();
        let enc = self.encode(mel);
        let [_, n_frames, _] = enc.dims();
        let valid_frames = parakeet_subsampled_frames(mel_frames).min(n_frames);
        let device = enc.device();

        let mut cache = ParakeetDecoderCache::new(
            &device,
            decoder.lstm.layers.len(),
            decoder.hidden_size(),
        );

        let mut frame_idx = 0usize;
        // HF `decoder_start_token_id` is the blank (8192 on v3). Pad (2) is not a start token.
        let mut current_token = start_token_id;
        let mut decoded = Vec::new();
        let mut steps = 0usize;
        // TDT stops when the frame pointer exhausts the encoder (no RNN-T max_symbols guard).
        let max_steps = valid_frames * 64;

        while frame_idx < valid_frames && steps < max_steps {
            let joint = enc.dims()[2];
            let enc_frame = enc.clone().slice([0..1, frame_idx..frame_idx + 1, 0..joint])
                 .squeeze_dim::<2>(1);
            let update_decoder = current_token != blank_token_id;
            let dec_frame = decoder.forward_step(
                current_token,
                blank_token_id,
                &mut cache,
                update_decoder,
            );
            let logits = self.joint_step(enc_frame, dec_frame);
            let logit_vec = logits.into_data().to_vec::<f32>().unwrap();

            let mut best_token = 0usize;
            let mut best_token_score = f32::NEG_INFINITY;
            for (i, &v) in logit_vec.iter().take(vocab_size).enumerate() {
                if v > best_token_score {
                    best_token_score = v;
                    best_token = i;
                }
            }

            let mut duration = 0usize;
            let mut best_dur_score = f32::NEG_INFINITY;
            let d_end = (duration_start + num_durations).min(logit_vec.len());
            for (offset, &v) in logit_vec[duration_start..d_end].iter().enumerate() {
                if v > best_dur_score {
                    best_dur_score = v;
                    duration = offset;
                }
            }

            // HF ParakeetTDTGenerationMixin: only force progress for blank + duration 0.
            if best_token == blank_token_id && duration == 0 {
                duration = 1;
            }

            if verbose && steps < 10 {
                println!(
                    "DEBUG TDT step {}: frame={}, token={}, dur={}",
                    steps, frame_idx, best_token, duration
                );
            }
            if verbose && steps == 0 {
                println!(
                    "DEBUG TDT: mel_frames={}, enc_frames={}, valid_frames={}",
                    mel_frames, n_frames, valid_frames
                );
            }

            if best_token != blank_token_id && best_token < vocab_size {
                decoded.push(best_token);
                current_token = best_token;
            } else {
                current_token = blank_token_id;
            }

            frame_idx += duration;
            steps += 1;
        }

        decoded
    }

    /// RNN-T greedy used when the joint head has no duration bins (Nemotron-style, or a
    /// Parakeet RNNT checkpoint). Advance one frame on blank or after `max_symbols` labels.
    pub fn decode_rnnt_greedy(
        &self,
        mel: Tensor<3>,
        blank_token_id: usize,
        vocab_size: usize,
        max_symbols: usize,
        start_token_id: usize,
    ) -> Vec<usize> {
        let decoder = match &self.decoder {
            Some(d) => d,
            None => return Vec::new(),
        };
        let [_, _, mel_frames] = mel.dims();
        let enc = self.encode(mel);
        let [_, n_frames, joint] = enc.dims();
        let valid = crate::audio::parakeet_subsampled_frames(mel_frames).min(n_frames);
        let device = enc.device();
        let mut cache = ParakeetDecoderCache::new(&device, decoder.lstm.layers.len(), decoder.hidden_size());
        let mut frame = 0usize;
        let mut symbols = 0usize;
        let mut current = start_token_id;
        let mut decoded = Vec::new();
        let limit = valid.saturating_mul(max_symbols.max(1) + 1).max(1);
        let mut steps = 0usize;
        while frame < valid && steps < limit {
            let enc_frame = enc.clone().slice([0..1, frame..frame + 1, 0..joint]).squeeze_dim::<2>(1);
            let update = current != blank_token_id;
            let dec = decoder.forward_step(current, blank_token_id, &mut cache, update);
            let row = self.joint_step(enc_frame, dec).into_data().to_vec::<f32>().unwrap();
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (i, &v) in row.iter().take(vocab_size).enumerate() {
                if v > best_s {
                    best_s = v;
                    best = i;
                }
            }
            if best == blank_token_id {
                frame += 1;
                symbols = 0;
                current = blank_token_id;
            } else {
                decoded.push(best);
                current = best;
                symbols += 1;
                if symbols >= max_symbols.max(1) {
                    frame += 1;
                    symbols = 0;
                }
            }
            steps += 1;
        }
        decoded
    }

    pub fn encoder_ctx_size(&self) -> usize {
        1500
    }

    pub fn encoder_mel_size(&self) -> usize {
        self.n_mels
    }

    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }
}