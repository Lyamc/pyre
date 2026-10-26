use burn::config::Config;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HfTokenizerEncoderConfig {
    pub channels: usize,
    pub depths: Vec<usize>,
    pub downsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub ffn_expansion: usize,
    pub layer_scale_init_value: f32,
    pub rms_norm_eps: f64,
    pub vae_std: f32,
}

#[derive(Config, Debug)]
pub struct TokenizerEncoderConfig {
    pub channels: usize,
    pub output_dim: usize,
    pub num_filters: usize,
    pub depths: Vec<usize>,
    pub downsampling_ratios: Vec<usize>,
    pub kernel_size: usize,
    pub ffn_expansion: usize,
    pub layer_scale_init_value: f32,
    pub rms_norm_eps: f64,
    pub fix_std: f32,
}

impl From<HfTokenizerEncoderConfig> for TokenizerEncoderConfig {
    fn from(hf: HfTokenizerEncoderConfig) -> Self {
        Self {
            channels: hf.channels,
            output_dim: hf.hidden_size,
            num_filters: hf.num_filters,
            depths: hf.depths,
            downsampling_ratios: hf.downsampling_ratios,
            kernel_size: hf.kernel_size,
            ffn_expansion: hf.ffn_expansion,
            layer_scale_init_value: hf.layer_scale_init_value,
            rms_norm_eps: hf.rms_norm_eps,
            fix_std: hf.vae_std,
        }
    }
}

impl TokenizerEncoderConfig {
    pub fn channels_at_level(&self, level: usize) -> usize {
        self.num_filters * (1 << level)
    }
}