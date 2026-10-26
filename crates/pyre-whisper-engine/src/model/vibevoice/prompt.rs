use super::runtime::VibeVoiceRuntimeConfig;
use super::waveform::vae_token_length;

const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

pub struct VibeVoicePrompt {
    pub system_prompt: String,
    pub user_prefix_tokens: Vec<String>,
    pub pad_token: String,
    pub user_suffix: String,
    pub vae_token_count: usize,
}

pub fn build_asr_prompt(
    runtime: &VibeVoiceRuntimeConfig,
    num_samples: usize,
    hotwords: Option<&str>,
) -> VibeVoicePrompt {
    let vae_token_count = vae_token_length(num_samples, runtime.speech_compress_ratio);
    let duration = num_samples as f32 / runtime.sample_rate as f32;

    let user_suffix = if let Some(ctx) = hotwords.filter(|s| !s.is_empty()) {
        format!(
            "This is a {duration:.2} seconds audio, with extra info: {ctx}\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )
    } else {
        format!(
            "This is a {duration:.2} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )
    };

    let user_prefix_tokens = vec![
        runtime.audio_bos_token.clone(),
        runtime.audio_pad_token.clone(),
        runtime.audio_eos_token.clone(),
    ];

    VibeVoicePrompt {
        system_prompt: SYSTEM_PROMPT.to_string(),
        user_prefix_tokens,
        pad_token: runtime.audio_pad_token.clone(),
        user_suffix,
        vae_token_count,
    }
}

pub fn pad_placeholders(count: usize, pad_token: &str) -> Vec<String> {
    std::iter::repeat(pad_token)
        .take(count.saturating_sub(1))
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_prompt_with_pad_count() {
        let runtime = VibeVoiceRuntimeConfig::default();
        let prompt = build_asr_prompt(&runtime, 24_000, None);
        assert_eq!(prompt.vae_token_count, 8);
        assert!(prompt.user_suffix.contains("seconds audio"));
    }
}