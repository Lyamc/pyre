use serde::{Deserialize, Serialize};

use crate::language::LanguageCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechTask {
    Transcribe,
    Translate,
    Diarize,
    Tts,
    Sts,
    Stt,
}

impl SpeechTask {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "transcribe" | "transcription" | "asr" => Some(Self::Transcribe),
            "translate" | "translation" => Some(Self::Translate),
            "diarize" | "diarization" => Some(Self::Diarize),
            "tts" | "text_to_speech" => Some(Self::Tts),
            "sts" | "speech_to_speech" => Some(Self::Sts),
            "stt" | "speech_to_text" => Some(Self::Stt),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
            Self::Diarize => "diarize",
            Self::Tts => "tts",
            Self::Sts => "sts",
            Self::Stt => "stt",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Asr,
    Translation,
    Diarization,
    Vad,
    Tts,
    Sts,
    Llm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOptions {
    pub task: SpeechTask,
    pub language: LanguageCode,
    pub target_language: Option<LanguageCode>,
    pub include_timestamps: bool,
    pub beam_size: usize,
    pub max_tokens: usize,
    pub padding: usize,
    pub orchestrate: bool,
    pub scout_model: Option<String>,
    /// Post-process segments into sentence groups for timed outputs (txt/srt/vtt/json sentences list)
    pub group_into_sentences: bool,
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            task: SpeechTask::Transcribe,
            language: LanguageCode::english(),
            target_language: None,
            include_timestamps: true,
            beam_size: 5,
            max_tokens: 448,
            padding: 200,
            orchestrate: false,
            scout_model: Some("tiny_en".to_string()),
            group_into_sentences: false,
        }
    }
}