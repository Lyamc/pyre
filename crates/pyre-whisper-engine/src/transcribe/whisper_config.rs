use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Default)]
pub struct WhisperDecodeConfig {
    pub suppress_tokens: Vec<usize>,
    pub begin_suppress_tokens: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct HfGenerationConfig {
    #[serde(default)]
    suppress_tokens: Vec<i64>,
    #[serde(default)]
    begin_suppress_tokens: Vec<i64>,
}

pub fn load_whisper_decode_config(model_name: &str) -> WhisperDecodeConfig {
    let path = pyre_whisper_core::resolve_model_file(model_name, "config.json");
    if !Path::new(&path).exists() {
        return WhisperDecodeConfig::default();
    }

    let Ok(text) = std::fs::read_to_string(&path) else {
        return WhisperDecodeConfig::default();
    };

    let Ok(cfg) = serde_json::from_str::<HfGenerationConfig>(&text) else {
        return WhisperDecodeConfig::default();
    };

    WhisperDecodeConfig {
        suppress_tokens: positive_ids(cfg.suppress_tokens),
        begin_suppress_tokens: positive_ids(cfg.begin_suppress_tokens),
    }
}

fn positive_ids(ids: Vec<i64>) -> Vec<usize> {
    ids.into_iter()
        .filter(|&id| id >= 0)
        .map(|id| id as usize)
        .collect()
}

pub fn build_logit_mask(
    vocab_size: usize,
    special_mask: &[f32],
    decode_cfg: &WhisperDecodeConfig,
    at_generation_start: bool,
    end_token: usize,
) -> Vec<f32> {
    let mut mask = special_mask.to_vec();
    for &token in &decode_cfg.suppress_tokens {
        if token != end_token && token < vocab_size {
            mask[token] = f32::NEG_INFINITY;
        }
    }
    if at_generation_start {
        for &token in &decode_cfg.begin_suppress_tokens {
            if token < vocab_size {
                mask[token] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_tiny_en_suppress_lists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !root.join("models/tiny_en/config.json").exists() {
            return;
        }
        let _ = std::env::set_current_dir(&root);
        let cfg = load_whisper_decode_config("tiny_en");
        assert!(!cfg.suppress_tokens.is_empty());
        assert!(!cfg.begin_suppress_tokens.is_empty());
    }
}