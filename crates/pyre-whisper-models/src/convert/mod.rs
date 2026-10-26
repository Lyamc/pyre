pub mod burn_mpk;
pub mod dtype;
pub mod npy;
pub mod moonshine;
pub mod parakeet;
pub mod qwen3;
pub mod tone;
pub mod vibevoice;
pub mod whisper;

pub use burn_mpk::save_model_from_npy;
pub use parakeet::{
    convert_parakeet_from_hf, is_parakeet_burn_ready, is_parakeet_model, ParakeetDecodeConfig,
};
pub use moonshine::{is_moonshine_model, prepare_moonshine_bundle};
pub use qwen3::{is_qwen3_model, prepare_qwen3_bundle, Qwen3RuntimeConfig};
pub use tone::{is_tone_model, prepare_tone_bundle};
pub use vibevoice::{is_vibevoice_model, prepare_vibevoice_bundle, VibeVoiceRuntimeConfig};

pub fn is_vad_model(name: &str) -> bool {
    matches!(name, "silero-vad" | "ten-vad")
}

pub fn is_diarize_model(name: &str) -> bool {
    matches!(name, "diarization-3.1" | "nemo-diarization")
}
pub use whisper::{convert_whisper_from_hf, is_whisper_model, needs_reconversion};

pub fn needs_reconversion_for(name: &str, model_dir: &std::path::Path) -> bool {
    if qwen3::is_qwen3_model(name) {
        qwen3::needs_qwen3_reconversion(model_dir)
    } else if vibevoice::is_vibevoice_model(name) {
        vibevoice::needs_vibevoice_reconversion(model_dir)
    } else if moonshine::is_moonshine_model(name) {
        moonshine::needs_moonshine_reconversion(model_dir)
    } else if tone::is_tone_model(name) {
        tone::needs_tone_reconversion(model_dir)
    } else {
        whisper::needs_reconversion(model_dir)
    }
}