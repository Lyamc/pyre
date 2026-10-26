use pyre_whisper_core::ModelCapability;

macro_rules! model {
    (
        $name:expr, $hf:expr, $desc:expr, $cat:expr, $vram:expr, $ready:expr,
        $size:expr, $quality:expr, $effort:expr, $efficiency:expr, $eff_pct:expr, $tagline:expr
    ) => {
        ModelInfo {
            name: $name,
            hf_id: $hf,
            description: $desc,
            category: $cat,
            vram_mb: $vram,
            burn_ready: $ready,
            size: $size,
            quality: $quality,
            effort: $effort,
            efficiency: $efficiency,
            efficiency_pct: $eff_pct,
            tagline: $tagline,
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    Low,
    Medium,
    High,
}

use serde::{Deserialize, Serialize};

pub struct ModelInfo {
    pub name: &'static str,
    pub hf_id: &'static str,
    pub description: &'static str,
    pub category: ModelCategory,
    pub vram_mb: u32,
    pub burn_ready: bool,
    pub size: ModelTier,
    pub quality: ModelTier,
    pub effort: ModelTier,
    pub efficiency: ModelTier,
    /// Approximate peak GPU utilization (0–100) on a mid-range discrete GPU.
    pub efficiency_pct: u8,
    /// Short reason to pick this model (max 32 chars).
    pub tagline: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Asr,
    Vad,
    Diarization,
    Tts,
    Llm,
}

impl ModelCategory {
    pub fn capability(&self) -> ModelCapability {
        match self {
            Self::Asr => ModelCapability::Asr,
            Self::Vad => ModelCapability::Vad,
            Self::Diarization => ModelCapability::Diarization,
            Self::Tts => ModelCapability::Tts,
            Self::Llm => ModelCapability::Llm,
        }
    }
}

pub const MODEL_REGISTRY: &[ModelInfo] = &[
    model!("tiny", "openai/whisper-tiny", "Smallest Whisper, multilingual.", ModelCategory::Asr, 400, true, ModelTier::Low, ModelTier::Low, ModelTier::Low, ModelTier::Low, 30, "Fast drafts, many langs"),
    model!("tiny_en", "openai/whisper-tiny.en", "Smallest Whisper, English.", ModelCategory::Asr, 400, true, ModelTier::Low, ModelTier::Low, ModelTier::Low, ModelTier::Low, 30, "Fastest English drafts"),
    model!("base", "openai/whisper-base", "Base Whisper, multilingual.", ModelCategory::Asr, 600, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::Low, 40, "Balanced speed & quality"),
    model!("base_en", "openai/whisper-base.en", "Base Whisper, English.", ModelCategory::Asr, 600, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::Low, 40, "Good English balance"),
    model!("small", "openai/whisper-small", "Small Whisper, multilingual.", ModelCategory::Asr, 1200, true, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, 55, "Clearer multilingual"),
    model!("small_en", "openai/whisper-small.en", "Small Whisper, English.", ModelCategory::Asr, 1200, true, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, 55, "Clearer English"),
    model!("medium", "openai/whisper-medium", "Medium Whisper, multilingual.", ModelCategory::Asr, 2500, true, ModelTier::High, ModelTier::High, ModelTier::High, ModelTier::High, 70, "High accuracy, slower"),
    model!("medium_en", "openai/whisper-medium.en", "Medium Whisper, English.", ModelCategory::Asr, 2500, true, ModelTier::High, ModelTier::High, ModelTier::High, ModelTier::High, 70, "Best Whisper English"),
    model!("large-v3-turbo", "openai/whisper-large-v3-turbo", "Fast large Whisper.", ModelCategory::Asr, 6000, true, ModelTier::High, ModelTier::High, ModelTier::Medium, ModelTier::High, 85, "Large quality, faster"),
    model!("distil-medium-en", "distil-whisper/distil-medium.en", "Distilled Whisper medium.", ModelCategory::Asr, 1500, true, ModelTier::Medium, ModelTier::Medium, ModelTier::Low, ModelTier::Medium, 60, "Fast, solid English"),
    model!("distil-large-v3", "distil-whisper/distil-large-v3", "Distilled Whisper large-v3.", ModelCategory::Asr, 3000, true, ModelTier::High, ModelTier::High, ModelTier::Medium, ModelTier::High, 75, "Large accuracy, distilled"),
    model!("parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", "Parakeet TDT v3 multilingual ASR.", ModelCategory::Asr, 1200, true, ModelTier::Medium, ModelTier::High, ModelTier::Medium, ModelTier::Medium, 65, "NVIDIA fast & accurate"),
    model!("parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v2", "Parakeet TDT v2 English ASR (NeMo .nemo converted to Burn).", ModelCategory::Asr, 1200, true, ModelTier::Medium, ModelTier::High, ModelTier::Medium, ModelTier::Medium, 65, "Parakeet v2 English"),
    model!("parakeet-ctc-0.6b", "nvidia/parakeet-ctc-0.6b", "Parakeet CTC 0.6B English ASR (Burn).", ModelCategory::Asr, 1200, true, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, ModelTier::Medium, 50, "Parakeet CTC English"),
    model!("parakeet-ctc-1.1b", "nvidia/parakeet-ctc-1.1b", "Parakeet CTC 1.1B English ASR (Burn).", ModelCategory::Asr, 2200, true, ModelTier::High, ModelTier::High, ModelTier::Medium, ModelTier::Medium, 55, "Larger Parakeet CTC"),
    model!("t-one", "t-tech/T-one", "T-Tech T-one Russian 8 kHz Conformer CTC (Burn offline greedy).", ModelCategory::Asr, 500, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::Low, 50, "Russian telephony ASR"),
    model!("moonshine-tiny", "UsefulSensors/moonshine-tiny", "Moonshine tiny English ASR (raw 16 kHz, Burn greedy decode).", ModelCategory::Asr, 300, true, ModelTier::Low, ModelTier::Low, ModelTier::Low, ModelTier::Low, 25, "Tiny edge ASR"),
    model!("moonshine-base", "UsefulSensors/moonshine-base", "Moonshine base English ASR (raw 16 kHz, Burn greedy decode).", ModelCategory::Asr, 500, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::Medium, 40, "Small edge ASR"),
    model!("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3-0.6B instruct (offline English transcript summarizer).", ModelCategory::Llm, 1800, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::High, 45, "Fast English summaries"),
    model!("qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B (Burn inference: audio tower + thinker + greedy decode).", ModelCategory::Asr, 2500, true, ModelTier::Medium, ModelTier::High, ModelTier::Medium, ModelTier::Medium, 60, "Qwen ASR, many langs"),
    model!("qwen3-asr-1.7b", "Qwen/Qwen3-ASR-1.7B-hf", "Qwen3-ASR 1.7B (Burn inference: audio tower + thinker + greedy decode).", ModelCategory::Asr, 4500, true, ModelTier::High, ModelTier::High, ModelTier::High, ModelTier::High, 75, "Top Qwen accuracy"),
    model!("vibevoice-asr", "microsoft/VibeVoice-ASR-HF", "Microsoft VibeVoice-ASR 7B (INT8 Qwen2.5 decoder + dual encoders, Burn greedy STT).", ModelCategory::Asr, 8_000, true, ModelTier::High, ModelTier::High, ModelTier::High, ModelTier::Low, 40, "Large STT, INT8 decoder"),
    model!("bitnet-asr", "microsoft/VibeVoice-ASR-BitNet", "Microsoft VibeVoice-ASR-BitNet 1.5B (ternary I2_S Qwen2.5 decoder + dual VAEs, Burn greedy STT).", ModelCategory::Asr, 2_500, true, ModelTier::Medium, ModelTier::High, ModelTier::Medium, ModelTier::High, 55, "Ternary 1.5B STT"),
    model!("silero-vad", "snakers4/silero-vad", "Silero VAD v5 (Burn/host, 16 kHz chunks).", ModelCategory::Vad, 50, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::High, 90, "Voice activity detect"),
    model!("ten-vad", "TEN-framework/ten-vad", "TEN VAD (Burn/host, 16 kHz hop 256).", ModelCategory::Vad, 50, true, ModelTier::Low, ModelTier::Medium, ModelTier::Low, ModelTier::High, 90, "TEN voice detect"),
    model!("diarization-3.1", "pyannote/wespeaker-voxceleb-resnet34-LM", "Pyannote 3.1-style clustering diarization (WeSpeaker embeddings).", ModelCategory::Diarization, 500, true, ModelTier::Medium, ModelTier::High, ModelTier::High, ModelTier::Medium, 55, "Speaker diarization"),
    model!("nemo-diarization", "nvidia/speakerverification_en_titanet_large", "NeMo clustering diarization (TitaNet/WeSpeaker embeddings).", ModelCategory::Diarization, 800, true, ModelTier::Medium, ModelTier::High, ModelTier::High, ModelTier::Medium, 55, "NeMo phone diarization"),
];

pub fn get_model_names() -> Vec<&'static str> {
    MODEL_REGISTRY.iter().map(|m| m.name).collect()
}

pub fn find_model(name: &str) -> Option<&'static ModelInfo> {
    MODEL_REGISTRY.iter().find(|m| m.name == name)
}

pub fn default_models() -> Vec<&'static str> {
    vec!["tiny_en", "tiny", "base_en", "small_en"]
}

pub fn models_by_category(category: ModelCategory) -> Vec<&'static ModelInfo> {
    MODEL_REGISTRY.iter().filter(|m| m.category == category).collect()
}

pub fn vram_fit(vram_required_mb: u32, gpu_vram_mb: Option<u32>) -> &'static str {
    match gpu_vram_mb {
        None => "unknown",
        Some(gpu) if gpu >= vram_required_mb.saturating_add(256) => "ok",
        Some(gpu) if gpu >= vram_required_mb => "tight",
        _ => "no",
    }
}