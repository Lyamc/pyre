pub mod block;
mod config;
pub mod level;
pub mod norm;
mod pad;
pub mod sconv;

use burn::module::Module;
use burn::tensor::{Tensor};

pub use config::{HfTokenizerEncoderConfig, TokenizerEncoderConfig};
pub use level::EncoderLevel;
pub use sconv::SConv1d;

use block::Block1D;
use sconv::SConv1d as SConv;

#[derive(Module, Debug)]
pub struct TokenizerEncoder {
    pub stem: EncoderLevel,
    pub conv_layers: Vec<EncoderLevel>,
    pub head: SConv,
}

impl TokenizerEncoder {
    pub fn forward(&self, audio: Tensor<3>) -> Tensor<3> {
        let x = self.stem.forward(audio);
        let x = self
            .conv_layers
            .iter()
            .fold(x, |acc, layer| layer.forward(acc));
        self.head.forward(x)
    }

    /// `[batch, time, dim]` latent mean (HF permute after encoder).
    pub fn encode_mean(&self, audio: Tensor<3>) -> Tensor<3> {
        self.forward(audio).swap_dims(1, 2)
    }
}

pub fn init_encoder(cfg: &TokenizerEncoderConfig, device: &burn::tensor::Device) -> TokenizerEncoder {
    let stem = init_level(
        cfg,
        0,
        cfg.channels,
        cfg.num_filters,
        cfg.kernel_size,
        1,
        cfg.depths[0],
        device,
    );
    let conv_layers = (0..cfg.downsampling_ratios.len())
        .map(|i| {
            let ratio = cfg.downsampling_ratios[i];
            let in_ch = cfg.channels_at_level(i);
            let out_ch = cfg.channels_at_level(i + 1);
            init_level(cfg, i + 1, in_ch, out_ch, ratio * 2, ratio, cfg.depths[i + 1], device)
        })
        .collect();
    let head_in = cfg.channels_at_level(cfg.depths.len() - 1);
    let head = SConv::new(head_in, cfg.output_dim, cfg.kernel_size, 1, 1, device);
    TokenizerEncoder {
        stem,
        conv_layers,
        head,
    }
}

fn init_level(
    cfg: &TokenizerEncoderConfig,
    _level: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    depth: usize,
    device: &burn::tensor::Device,
) -> EncoderLevel {
    let ch = out_ch;
    let blocks = (0..depth)
        .map(|_| Block1D {
            norm: norm::ConvRmsNorm {
                weight: burn::module::Param::from_tensor(Tensor::ones([ch], device)),
                eps: cfg.rms_norm_eps,
            },
            mixer: SConv::new(ch, ch, cfg.kernel_size, 1, ch, device),
            gamma: burn::module::Param::from_tensor(Tensor::ones([ch], device)),
            ffn_norm: norm::ConvRmsNorm {
                weight: burn::module::Param::from_tensor(Tensor::ones([ch], device)),
                eps: cfg.rms_norm_eps,
            },
            ffn: block::init_ffn(ch, cfg.ffn_expansion * ch, device),
            ffn_gamma: burn::module::Param::from_tensor(Tensor::ones([ch], device)),
        })
        .collect();
    EncoderLevel {
        conv: SConv::new(in_ch, out_ch, kernel, stride, 1, device),
        blocks,
    }
}