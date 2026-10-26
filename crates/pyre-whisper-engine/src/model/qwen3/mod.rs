pub(crate) mod audio_tower;
mod mel;
mod prompt;
mod repetition;
mod runtime;
pub(crate) mod thinker;
pub mod weights;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor};

pub use audio_tower::{Qwen3AudioTower, Qwen3AudioTowerConfig};
pub use prompt::{
    build_asr_prompt, flatten_prompt, qwen3_forced_language_from_code, qwen3_language_name,
    Qwen3AsrPrompt,
};
pub use runtime::{load_qwen3_runtime, Qwen3RuntimeConfig};
pub use thinker::{Qwen3Thinker, Qwen3ThinkerConfig};
pub use weights::{load_qwen3_lm_weights, load_qwen3_weights};
pub mod summarize;

/// HF default `max_new_tokens` from `Qwen3ASRModel.from_pretrained`.
pub const QWEN3_DEFAULT_MAX_NEW_TOKENS: usize = 512;

#[derive(Config, Debug)]
pub struct Qwen3ASRConfig {
    pub audio_tower: Qwen3AudioTowerConfig,
    pub thinker: Qwen3ThinkerConfig,
}

impl Qwen3ASRConfig {
    pub fn from_runtime(runtime: &Qwen3RuntimeConfig) -> Self {
        Self {
            audio_tower: Qwen3AudioTowerConfig {
                d_model: runtime.audio_d_model,
                output_dim: runtime.audio_output_dim,
                n_layers: runtime.audio_layers,
                n_heads: runtime.audio_heads,
                n_mels: runtime.num_mel_bins,
                ffn_dim: runtime.audio_ffn_dim,
                downsample_hidden_size: runtime.downsample_hidden_size,
                n_window: runtime.n_window,
                n_window_infer: runtime.n_window_infer,
                conv_chunksize: runtime.conv_chunksize,
                max_source_positions: runtime.max_source_positions,
            },
            thinker: Qwen3ThinkerConfig {
                hidden_size: runtime.text_hidden_size,
                n_layers: runtime.text_layers,
                n_heads: runtime.text_heads,
                n_kv_heads: runtime.text_kv_heads,
                head_dim: runtime.text_head_dim,
                intermediate_size: runtime.text_intermediate_size,
                vocab_size: runtime.vocab_size,
                rms_norm_eps: runtime.rms_norm_eps,
                rope_theta: runtime.rope_theta,
                mrope_section: runtime.mrope_section.clone(),
            },
        }
    }

    pub fn init(&self, device: &burn::tensor::Device) -> Qwen3ASR {
        Qwen3ASR {
            audio_tower: self.audio_tower.init(device),
            thinker: self.thinker.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3ASR {
    pub audio_tower: Qwen3AudioTower,
    pub thinker: Qwen3Thinker,
}

impl Qwen3ASR {
    pub fn encoder_ctx_size(&self) -> usize {
        1500
    }

    pub fn encoder_mel_size(&self) -> usize {
        self.audio_tower.n_mels
    }

    pub fn sample_rate(&self) -> usize {
        16_000
    }

    pub fn device(&self) -> burn::tensor::Device {
        self.audio_tower.conv2d1.weight.device()
    }

    pub fn encode_audio(&self, mel: Tensor<3>) -> Tensor<3> {
        self.audio_tower.forward(mel)
    }

}

pub fn build_qwen3_config(model_name: &str) -> Result<Qwen3ASRConfig, String> {
    let runtime = load_qwen3_runtime(model_name);
    Ok(Qwen3ASRConfig::from_runtime(&runtime))
}

/// Parse HF-style output into `(language, transcription text)`.
pub fn parse_qwen3_asr_output_parts(raw: &str) -> (String, String) {
    parse_qwen3_asr_output_parts_with_forced(raw, None)
}

/// Parse decoded output; when `forced_language` is set, treat the whole string as text-only
/// (HF behavior when the user forces language in the prompt).
pub fn parse_qwen3_asr_output_parts_with_forced(
    raw: &str,
    forced_language: Option<&str>,
) -> (String, String) {
    let s = repetition::detect_and_fix_repetitions(raw.trim(), 20);
    if s.is_empty() {
        return (String::new(), String::new());
    }

    if let Some(lang) = forced_language.filter(|l| !l.is_empty()) {
        return (normalize_language_name(lang), s);
    }

    if !s.contains("<asr_text>") {
        return (String::new(), s);
    }

    let (meta, text) = s.split_once("<asr_text>").unwrap_or((&s, ""));
    let text = text.trim().to_string();

    if meta.to_ascii_lowercase().contains("language none") && text.is_empty() {
        return (String::new(), String::new());
    }

    let mut language = String::new();
    for line in meta.lines() {
        let line = line.trim();
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("language ") {
            let val = line["language ".len()..].trim();
            if !val.is_empty() {
                language = normalize_language_name(val);
            }
            break;
        }
    }

    (language, text)
}

pub fn parse_qwen3_asr_output(text: &str) -> String {
    parse_qwen3_asr_output_with_forced(text, None)
}

pub fn parse_qwen3_asr_output_with_forced(text: &str, forced_language: Option<&str>) -> String {
    parse_qwen3_asr_output_parts_with_forced(text, forced_language).1
}

fn normalize_language_name(language: &str) -> String {
    let s = language.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect(),
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn parses_language_and_text_tag() {
        let (lang, text) =
            parse_qwen3_asr_output_parts("language English<asr_text>Hello world.");
        assert_eq!(lang, "English");
        assert_eq!(text, "Hello world.");
    }

    #[test]
    fn parses_multiline_meta() {
        let (lang, text) = parse_qwen3_asr_output_parts(
            "language Chinese\nsome noise\n<asr_text>你好",
        );
        assert_eq!(lang, "Chinese");
        assert_eq!(text, "你好");
    }

    #[test]
    fn plain_text_without_tag() {
        let (lang, text) = parse_qwen3_asr_output_parts("plain transcript");
        assert!(lang.is_empty());
        assert_eq!(text, "plain transcript");
    }

    #[test]
    fn forced_language_treats_output_as_plain_text() {
        let (lang, text) =
            parse_qwen3_asr_output_parts_with_forced("Hello from forced mode.", Some("English"));
        assert_eq!(lang, "English");
        assert_eq!(text, "Hello from forced mode.");
    }

    #[test]
    fn collapses_repetitions_before_parse() {
        let repeated = format!("language English<asr_text>{}", "hi ".repeat(25));
        let (_, text) = parse_qwen3_asr_output_parts(&repeated);
        assert_eq!(text, "hi hi");
    }
}