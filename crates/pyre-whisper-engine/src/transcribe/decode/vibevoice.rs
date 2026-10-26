use burn::tensor::Tensor;

use crate::model::vibevoice::{
    build_asr_prompt, load_vibevoice_runtime, normalize_dbfs, VibeVoiceASR,
};
use crate::token::Gpt2Tokenizer;

pub fn decode_vibevoice(
    model: &VibeVoiceASR,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    model_name: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    let runtime = load_vibevoice_runtime(model_name);
    let target_rate = runtime.sample_rate;

    let waveform = if sample_rate == target_rate {
        waveform.to_vec()
    } else {
        pyre_whisper_audio::resample_mono(waveform, sample_rate, target_rate)
            .unwrap_or_else(|_| waveform.to_vec())
    };

    let normalized = normalize_dbfs(&waveform, runtime.target_dbfs);
    let prompt = build_asr_prompt(&runtime, normalized.len(), None);
    let device = model.decoder.lm_head.device();
    let audio = Tensor::<1>::from_floats(normalized.as_slice(), &device)
        .reshape([1, 1, normalized.len()]);
    let speech = model.encode_speech(audio);
    let [_, speech_tokens, _] = speech.dims();

    if verbose {
        println!(
            "DEBUG VibeVoice: {} samples @ {}Hz, {} speech tokens, {} prompt pads",
            normalized.len(),
            target_rate,
            speech_tokens,
            prompt.vae_token_count
        );
    }

    let pad_count = speech_tokens.max(1);
    let pads = runtime.audio_pad_token.repeat(pad_count);
    // Official VibeVoice-ASR-HF decode starts with `assistant\n[{...}]`, so the
    // model emits the chat header. Do not prefill `<|im_start|>assistant`.
    let prompt_text = format!(
        "<|im_start|>system\n{}<|im_end|>\n\
         <|im_start|>user\n{}{}{}\n{}<|im_end|>\n",
        prompt.system_prompt,
        runtime.audio_bos_token,
        pads,
        runtime.audio_eos_token,
        prompt.user_suffix
    );
    let prefix_ids = bpe.encode_with_special_tokens(&prompt_text, false);
    let pad_id = bpe
        .token_to_id(&runtime.audio_pad_token)
        .or_else(|| bpe.token_to_id("<|box_start|>"))
        .unwrap_or(0);
    let pad_positions = prefix_ids.iter().filter(|&&id| id == pad_id).count();
    let eos_ids: Vec<usize> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| bpe.token_to_id(t))
        .collect();
    let banned_ids: Vec<usize> = [
        runtime.audio_pad_token.as_str(),
        runtime.audio_bos_token.as_str(),
        runtime.audio_eos_token.as_str(),
        "<|box_start|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
    ]
    .iter()
    .filter_map(|t| bpe.token_to_id(t))
    .filter(|id| *id != 0 && !eos_ids.contains(id))
    .collect::<std::collections::BTreeSet<_>>()
    .into_iter()
    .collect();

    if verbose {
        println!(
            "DEBUG VibeVoice: pad_id={pad_id}, encoded pads={pad_positions}/{}, prefix={}, eos={eos_ids:?}, banned={banned_ids:?}",
            pad_count,
            prefix_ids.len()
        );
    }

    if pad_id == 0
        || pad_positions == 0
        || pad_positions != pad_count
        || pad_positions != speech_tokens
        || eos_ids.is_empty()
    {
        return (
            format!(
                "VibeVoice prompt mismatch: pad_id={pad_id}, encoded pads={pad_positions}/{pad_count}, \
                 eos={eos_ids:?}. Refusing to run unbounded 7B decode."
            ),
            Vec::new(),
        );
    }

    let tokens = model.decoder.generate_greedy(
        &prefix_ids,
        speech,
        pad_id,
        &eos_ids,
        &banned_ids,
        256,
        &device,
        |ids| {
            if verbose && (ids.len() == 1 || ids.len() % 8 == 0) {
                let sofar = bpe.decode(ids, true).unwrap_or_default();
                let preview: String = sofar.chars().take(160).collect();
                println!("DEBUG VibeVoice: +{} toks: {preview}", ids.len());
            }
            should_stop_generation(ids, bpe)
        },
    );
    let text = unwrap_vibevoice_text(&bpe.decode(&tokens, true).unwrap_or_default());
    (text, tokens)
}

fn should_stop_generation(ids: &[usize], bpe: &Gpt2Tokenizer) -> bool {
    if repeated_loop(ids) {
        return true;
    }
    let text = bpe.decode(ids, true).unwrap_or_default();
    json_generation_complete(&text)
}

fn repeated_loop(ids: &[usize]) -> bool {
    if ids.len() >= 8 {
        let last = ids[ids.len() - 1];
        if ids[ids.len() - 8..].iter().all(|&t| t == last) {
            return true;
        }
    }
    if ids.len() >= 24 {
        let n = 8;
        return ids[ids.len() - n * 2..ids.len() - n] == ids[ids.len() - n..];
    }
    false
}

fn strip_assistant_prefix(text: &str) -> &str {
    let t = text.trim();
    t.strip_prefix("assistant")
        .map(|s| s.trim_start_matches(['\n', '\r', ' ', ':']))
        .unwrap_or(t)
}

fn extract_json_blob(text: &str) -> Option<&str> {
    let t = strip_assistant_prefix(text);
    let start = t.find('[').or_else(|| t.find('{'))?;
    let bytes = t.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &c) in bytes[start..].iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&t[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn contents_from_json(value: &serde_json::Value) -> Option<String> {
    let items = match value {
        serde_json::Value::Array(items) => items.clone(),
        serde_json::Value::Object(_) => vec![value.clone()],
        _ => return None,
    };
    let parts: Vec<String> = items
        .iter()
        .filter_map(|item| {
            item.get("Content")
                .or_else(|| item.get("text"))
                .or_else(|| item.get("content"))
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn content_fields_loose(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let needle = "\"Content\"";
    let mut search = text;
    while let Some(pos) = search.find(needle) {
        let rest = &search[pos + needle.len()..];
        let Some(colon) = rest.find(':') else { break };
        let after = rest[colon + 1..].trim_start();
        if let Some(body) = after.strip_prefix('"') {
            let mut escaped = false;
            let mut end = None;
            for (i, c) in body.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '"' {
                    end = Some(i);
                    break;
                }
            }
            if let Some(end) = end {
                out.push(body[..end].replace("\\\"", "\"").replace("\\n", " "));
                search = &body[end + 1..];
                continue;
            }
        }
        break;
    }
    out
}

fn json_generation_complete(text: &str) -> bool {
    extract_json_blob(text)
        .and_then(|blob| serde_json::from_str::<serde_json::Value>(blob).ok())
        .and_then(|v| contents_from_json(&v))
        .is_some()
}

fn unwrap_vibevoice_text(text: &str) -> String {
    let stripped = strip_assistant_prefix(text);
    if let Some(blob) = extract_json_blob(stripped) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(blob) {
            if let Some(joined) = contents_from_json(&value) {
                return joined;
            }
        }
    }
    let loose = content_fields_loose(stripped);
    if !loose.is_empty() {
        return loose.join(" ");
    }
    stripped.to_string()
}

#[cfg(test)]
mod tests {
    use super::{json_generation_complete, unwrap_vibevoice_text};

    #[test]
    fn unwraps_official_start_content_keys() {
        let raw = r#"assistant
[{"Start":0.0,"End":11.0,"Speaker":0,"Content":"And so, my fellow Americans, ask not what your country can do for you."}]
"#;
        let text = unwrap_vibevoice_text(raw);
        assert!(text.to_lowercase().contains("ask not"));
        assert!(text.to_lowercase().contains("fellow american"));
    }

    #[test]
    fn stops_on_complete_json_array() {
        let done = r#"assistant
[{"Start":0,"End":1,"Speaker":0,"Content":"Hello."}]"#;
        assert!(json_generation_complete(done));
        assert!(!json_generation_complete("assistant\n[{\"Content\":\"Hel"));
    }
}
