//! Model storage: the shared Hugging Face hub cache (read in place, zero extra disk) plus
//! pyre's own store (`<models-dir>/<kind>/<org>/<name>/<rev>/` + `pyre-model.toml`).
pub mod hfcache;
pub mod resolve;
pub mod store;

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Kind { Llm, Image, Stt, Tts, Tagger, Unknown }

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self { Kind::Llm => "llm", Kind::Image => "image", Kind::Stt => "stt", Kind::Tts => "tts", Kind::Tagger => "tagger", Kind::Unknown => "unknown" }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s.to_ascii_lowercase().as_str() {
            "llm" | "text" | "text-generation" | "chat" => Kind::Llm,
            "image" | "image-generation" | "sd" => Kind::Image,
            "stt" | "asr" | "speech-to-text" | "whisper" => Kind::Stt,
            "tts" | "text-to-speech" => Kind::Tts,
            "tagger" | "tag" | "tagging" | "image-classification" | "classify" => Kind::Tagger,
            _ => return None,
        })
    }
    /// OpenAI-ish task label used in /v1/models.
    pub fn task(&self) -> &'static str {
        match self { Kind::Llm => "text-generation", Kind::Image => "image-generation", Kind::Stt => "speech-to-text", Kind::Tts => "text-to-speech", Kind::Tagger => "image-classification", Kind::Unknown => "unknown" }
    }
}

/// Guess kind + architecture from the files in a model folder.
pub fn detect(dir: &Path) -> (Kind, String) {
    // Danbooru taggers (Kagami / WD / timm EVA02): tag list + timm weights.
    if dir.join("selected_tags.csv").exists() && ["weights.pt", "model.safetensors"].iter().any(|f| dir.join(f).exists()) {
        return (Kind::Tagger, "eva02-tagger".into());
    }
    if dir.join("model_index.json").exists() || dir.join("unet").is_dir() {
        return (Kind::Image, "stable-diffusion".into());
    }
    if let Ok(s) = std::fs::read_to_string(dir.join("config.json")) {
        if let Ok(j) = serde_json::from_str::<serde_json::Value>(&s) {
            let mt = j.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let arch = j.get("architectures").and_then(|a| a.get(0)).and_then(|v| v.as_str()).unwrap_or(&mt).to_string();
            let kind = match mt.as_str() {
                "whisper" | "parakeet" | "wav2vec2" => Kind::Stt,
                "kokoro" | "vits" | "speecht5" => Kind::Tts,
                _ if arch.ends_with("ForCausalLM") || j.get("num_hidden_layers").is_some() => Kind::Llm,
                _ => Kind::Unknown,
            };
            return (kind, arch);
        }
    }
    (Kind::Unknown, String::new())
}

pub fn dir_size(dir: &Path) -> u64 {
    walk(dir).iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum()
}

pub fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if std::fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(false) { stack.push(p) } else { out.push(p) }
        }
    }
    out.sort();
    out
}

pub fn human(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 { format!("{:.2} GB", f / 1e9) } else if f >= 1e6 { format!("{:.1} MB", f / 1e6) } else { format!("{:.0} KB", f / 1e3) }
}

#[cfg(test)]
mod tests;
