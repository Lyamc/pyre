use std::path::Path;

use serde::Deserialize;

use super::encoder::{HfTokenizerEncoderConfig, TokenizerEncoderConfig};

#[derive(Debug, Deserialize)]
struct VibeVoiceHfFile {
    #[serde(default)]
    acoustic_tokenizer_encoder_config: Option<HfTokenizerEncoderConfig>,
    #[serde(default)]
    semantic_tokenizer_encoder_config: Option<HfTokenizerEncoderConfig>,
    #[serde(default)]
    acoustic_tokenizer_config: Option<OriginalTokenizerConfig>,
    #[serde(default)]
    semantic_tokenizer_config: Option<OriginalTokenizerConfig>,
}

/// Original VibeVoice / BitNet tokenizer section (`acoustic_tokenizer_config`).
#[derive(Debug, Deserialize)]
struct OriginalTokenizerConfig {
    #[serde(default = "default_channels")]
    channels: usize,
    encoder_depths: String,
    encoder_n_filters: usize,
    encoder_ratios: Vec<usize>,
    vae_dim: usize,
    #[serde(default = "default_kernel")]
    kernel_size: usize,
    #[serde(default = "default_ffn")]
    ffn_expansion: usize,
    #[serde(default = "default_layer_scale")]
    layer_scale_init_value: f32,
    #[serde(default = "default_eps")]
    layernorm_eps: f64,
    #[serde(default)]
    fix_std: f32,
}

fn default_channels() -> usize {
    1
}
fn default_kernel() -> usize {
    7
}
fn default_ffn() -> usize {
    4
}
fn default_layer_scale() -> f32 {
    1e-6
}
fn default_eps() -> f64 {
    1e-5
}

impl From<OriginalTokenizerConfig> for TokenizerEncoderConfig {
    fn from(cfg: OriginalTokenizerConfig) -> Self {
        let depths: Vec<usize> = cfg
            .encoder_depths
            .split('-')
            .filter_map(|s| s.parse().ok())
            .collect();
        // TokenizerEncoder does `self.ratios = list(reversed(config.ratios))`.
        // HF ports already store the reversed list as `downsampling_ratios`.
        let mut downsampling_ratios = cfg.encoder_ratios;
        downsampling_ratios.reverse();
        Self {
            channels: cfg.channels,
            output_dim: cfg.vae_dim,
            num_filters: cfg.encoder_n_filters,
            depths,
            downsampling_ratios,
            kernel_size: cfg.kernel_size,
            ffn_expansion: cfg.ffn_expansion,
            layer_scale_init_value: cfg.layer_scale_init_value,
            rms_norm_eps: cfg.layernorm_eps,
            fix_std: cfg.fix_std,
        }
    }
}

pub fn load_encoder_configs(model_dir: &str) -> Result<(TokenizerEncoderConfig, TokenizerEncoderConfig), String> {
    let path = Path::new(model_dir).join("config.json");
    let content = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let hf: VibeVoiceHfFile =
        serde_json::from_str(&content).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let acoustic = hf
        .acoustic_tokenizer_encoder_config
        .map(Into::into)
        .or_else(|| hf.acoustic_tokenizer_config.map(Into::into))
        .ok_or_else(|| format!("{}: missing acoustic tokenizer config", path.display()))?;
    let semantic = hf
        .semantic_tokenizer_encoder_config
        .map(Into::into)
        .or_else(|| hf.semantic_tokenizer_config.map(Into::into))
        .ok_or_else(|| format!("{}: missing semantic tokenizer config", path.display()))?;
    Ok((acoustic, semantic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_config_reverses_encoder_ratios() {
        let cfg = OriginalTokenizerConfig {
            channels: 1,
            encoder_depths: "3-3-3-3-3-3-8".into(),
            encoder_n_filters: 32,
            encoder_ratios: vec![8, 5, 5, 4, 2, 2],
            vae_dim: 64,
            kernel_size: 7,
            ffn_expansion: 4,
            layer_scale_init_value: 1e-6,
            layernorm_eps: 1e-5,
            fix_std: 0.5,
        };
        let enc = TokenizerEncoderConfig::from(cfg);
        assert_eq!(enc.downsampling_ratios, vec![2, 2, 4, 5, 5, 8]);
        assert_eq!(enc.downsampling_ratios.iter().product::<usize>(), 3200);
    }
}