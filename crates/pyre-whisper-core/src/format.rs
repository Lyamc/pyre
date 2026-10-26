use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Txt,
    Srt,
    Vtt,
    Wav,
}

impl OutputFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "txt" | "text" => Some(Self::Txt),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            "wav" => Some(Self::Wav),
            _ => None,
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Txt => "text/plain; charset=utf-8",
            Self::Srt => "application/x-subrip",
            Self::Vtt => "text/vtt",
            Self::Wav => "audio/wav",
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Txt => "txt",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Wav => "wav",
        }
    }
}