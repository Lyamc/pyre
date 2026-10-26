pub mod audio;
pub mod transcribe;
pub mod download;

use clap::Args;
use crate::token::Language;
use strum::IntoEnumIterator;

use crate::model::registry::get_model_names;

#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Name of the model to use (e.g., tiny_en, base_en)
    #[arg(short, long, value_parser = get_model_names())]
    pub model: String,

    /// Language of the audio (e.g., en, fr). Defaults to English.
    #[arg(short, long, default_value = "en")]
    pub lang: String,

    /// Show detailed processing information
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable low-level debug output
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Suppress real-time transcription output
    #[arg(short, long)]
    pub quiet: bool,

    /// Verify transcription against expected test string
    #[arg(long)]
    pub verify: bool,

    /// Hugging Face authentication token for gated models
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    // ADVANCED PARAMETERS
    /// [Advanced] Beam size for search (higher = better quality, slower)
    #[arg(long, default_value_t = 5, help_heading = "Advanced")]
    pub beam_size: usize,

    /// [Advanced] Maximum tokens to generate per chunk
    #[arg(long, default_value_t = 448, help_heading = "Advanced")]
    pub max_tokens: usize,

    /// [Advanced] Padding added to audio chunks (in frames)
    #[arg(long, default_value_t = 200, help_heading = "Advanced")]
    pub padding: usize,

    /// [Advanced] Backend device to use (0, 1, ... for specific GPU, or 'cpu')
    #[arg(long, help_heading = "Advanced")]
    pub device: Option<String>,
}

pub fn parse_language(lang_str: &str) -> Language {
    if lang_str.eq_ignore_ascii_case("auto") {
        return Language::English;
    }
    match Language::iter().find(|l| l.as_str() == lang_str) {
        Some(l) => l,
        None => {
            let supported: Vec<String> = Language::iter().map(|l| l.as_str().to_string()).collect();
            eprintln!("Invalid language: {}. Supported languages: {:?}", lang_str, supported);
            std::process::exit(1);
        }
    }
}

/// `None`/`"gpu"` -> default wgpu adapter, `"cpu"` -> Burn's flex CPU backend, `"<n>"` -> discrete GPU n,
/// `"cuda"`/`"cuda:<n>"` -> CUDA (when built with the `cuda` feature).
pub fn get_device(device_arg: &Option<String>) -> burn::tensor::Device {
    use burn::tensor::Device;
    let s = device_arg.as_deref().unwrap_or("").to_lowercase();
    #[cfg(feature = "cuda")]
    if let Some(rest) = s.strip_prefix("cuda") {
        return Device::cuda(rest.trim_start_matches(':').parse::<usize>().unwrap_or(0));
    }
    #[cfg(feature = "cpu")]
    if s == "cpu" {
        return Device::flex();
    }
    #[cfg(feature = "wgpu")]
    {
        use burn::tensor::DeviceKind;
        if let Ok(idx) = s.parse::<usize>() {
            return Device::wgpu(DeviceKind::DiscreteGpu(idx));
        }
        return Device::wgpu(DeviceKind::DefaultDevice);
    }
    #[allow(unreachable_code)]
    Device::default()
}

pub fn get_backend_info(device: &burn::tensor::Device) -> String {
    format!("{device:?}")
}
