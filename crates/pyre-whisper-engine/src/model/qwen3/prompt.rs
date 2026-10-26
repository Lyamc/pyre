use super::mel::audio_token_count;
use super::runtime::Qwen3RuntimeConfig;

pub struct Qwen3AsrPrompt {
    pub text: String,
    pub audio_pad_count: usize,
}

/// Build HF `apply_chat_template(..., add_generation_prompt=true)` text for Qwen3-ASR.
///
/// Template (from `chat_template.json`):
/// - system + user audio placeholders + assistant generation header
/// - optional `language {Name}<asr_text>` suffix when forcing language
pub fn build_asr_prompt(
    _runtime: &Qwen3RuntimeConfig,
    mel_frames: usize,
    context: &str,
    forced_language: Option<&str>,
) -> Qwen3AsrPrompt {
    let audio_pad_count = audio_token_count(mel_frames);
    let pads = "<|audio_pad|>".repeat(audio_pad_count);
    let mut text = format!(
        "<|im_start|>system\n{context}<|im_end|>\n\
         <|im_start|>user\n<|audio_start|>{pads}<|audio_end|><|im_end|>\n\
         <|im_start|>assistant\n"
    );
    if let Some(lang) = forced_language {
        if !lang.is_empty() {
            text.push_str(&format!("language {lang}<asr_text>"));
        }
    }
    Qwen3AsrPrompt { text, audio_pad_count }
}

pub fn flatten_prompt(prompt: &Qwen3AsrPrompt, bpe: &crate::token::Gpt2Tokenizer) -> Vec<usize> {
    // HF Qwen3 chat prompts already contain special tokens; do not let the encoder
    // strip adjacent newlines (token 198) around them.
    bpe.encode_with_special_tokens(&prompt.text, false)
}

/// Qwen3 forced-language suffix for the assistant prompt, if the user requested one.
///
/// HF default is auto-detect (`None`) for `auto`. Explicit ISO codes, including `en`,
/// append `language {Name}<asr_text>` so the decoder emits transcript text only.
pub fn qwen3_forced_language_from_code(iso: &str) -> Option<&'static str> {
    if iso.eq_ignore_ascii_case("auto") {
        return None;
    }
    qwen3_language_name(iso)
}

/// Map Whisper-style ISO codes to Qwen3 canonical language names.
pub fn qwen3_language_name(iso: &str) -> Option<&'static str> {
    match iso {
        "en" => Some("English"),
        "zh" => Some("Chinese"),
        "yue" => Some("Cantonese"),
        "ar" => Some("Arabic"),
        "de" => Some("German"),
        "fr" => Some("French"),
        "es" => Some("Spanish"),
        "pt" => Some("Portuguese"),
        "id" => Some("Indonesian"),
        "it" => Some("Italian"),
        "ko" => Some("Korean"),
        "ru" => Some("Russian"),
        "th" => Some("Thai"),
        "vi" => Some("Vietnamese"),
        "ja" => Some("Japanese"),
        "tr" => Some("Turkish"),
        "hi" => Some("Hindi"),
        "ms" => Some("Malay"),
        "nl" => Some("Dutch"),
        "sv" => Some("Swedish"),
        "da" => Some("Danish"),
        "fi" => Some("Finnish"),
        "pl" => Some("Polish"),
        "cs" => Some("Czech"),
        "fil" => Some("Filipino"),
        "fa" => Some("Persian"),
        "el" => Some("Greek"),
        "ro" => Some("Romanian"),
        "hu" => Some("Hungarian"),
        "mk" => Some("Macedonian"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_hf_chat_template_shape() {
        let runtime = Qwen3RuntimeConfig::default();
        let prompt = build_asr_prompt(&runtime, 100, "", None);
        assert_eq!(prompt.audio_pad_count, 13);
        assert!(prompt.text.starts_with("<|im_start|>system\n"));
        assert!(prompt.text.contains("<|im_start|>user\n<|audio_start|>"));
        assert_eq!(prompt.text.matches("<|audio_pad|>").count(), 13);
        assert!(prompt.text.contains("<|im_start|>assistant\n"));
        assert!(!prompt.text.contains("<asr_text>"));
    }

    #[test]
    fn forced_language_appends_asr_text_marker() {
        let runtime = Qwen3RuntimeConfig::default();
        let prompt = build_asr_prompt(&runtime, 10, "ctx", Some("English"));
        assert!(prompt.text.ends_with("language English<asr_text>"));
    }

    #[test]
    fn maps_common_iso_codes() {
        assert_eq!(qwen3_language_name("en"), Some("English"));
        assert_eq!(qwen3_language_name("zh"), Some("Chinese"));
        assert_eq!(qwen3_language_name("xx"), None);
    }

    #[test]
    fn forced_language_skips_auto_only() {
        assert_eq!(qwen3_forced_language_from_code("auto"), None);
        assert_eq!(qwen3_forced_language_from_code("en"), Some("English"));
        assert_eq!(qwen3_forced_language_from_code("zh"), Some("Chinese"));
        assert_eq!(qwen3_forced_language_from_code("xx"), None);
    }
}