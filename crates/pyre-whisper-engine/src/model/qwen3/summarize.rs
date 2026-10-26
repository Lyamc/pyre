use std::sync::Arc;


use pyre_whisper_core::{WhisburnError, WhisburnResult};

use crate::token::Gpt2Tokenizer;

use super::thinker::Qwen3ThinkerConfig;
use super::{load_qwen3_lm_weights, load_qwen3_runtime, Qwen3Thinker};

pub const DEFAULT_SUMMARIZER_MODEL: &str = "qwen3-0.6b";

/// ~1800 English words ≈ 2.4k tokens — short enough for fast greedy decode on 0.6B.
pub const CHUNK_WORDS: usize = 1800;
const CHUNK_MAX_NEW_TOKENS: usize = 256;
const MERGE_MAX_NEW_TOKENS: usize = 384;

pub type SummarizeProgress = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

const SYSTEM_PROMPT: &str = "You summarize English speech transcripts. Be concise. Use short bullet points. Cover: what it is about, key points, decisions, and action items. Do not invent facts.";

pub fn chunk_transcript(text: &str, max_words: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < words.len() {
        let mut end = (start + max_words).min(words.len());
        if end < words.len() {
            // Prefer a sentence boundary in the last quarter of the window.
            let search_from = start + (max_words * 3 / 4);
            if let Some(rel) = words[search_from..end]
                .iter()
                .rposition(|w| w.ends_with('.') || w.ends_with('?') || w.ends_with('!'))
            {
                end = search_from + rel + 1;
            }
        }
        chunks.push(words[start..end].join(" "));
        start = end;
    }
    chunks
}

pub fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    loop {
        let Some(a) = out.find("<think>") else { break };
        let close = "</think>";
        if let Some(rel) = out[a..].find(close) {
            out.replace_range(a..a + rel + close.len(), "");
        } else {
            out.replace_range(a.., "");
            break;
        }
    }
    out.trim().to_string()
}

fn build_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}\n/no_think<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn generate_completion(
    thinker: &Qwen3Thinker,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    max_new: usize,
    device: &burn::tensor::Device,
) -> WhisburnResult<String> {
    let ids = tokenizer.encode_with_special_tokens(prompt, false);
    let eos = {
        let mut ids = vec![151_643usize, 151_645];
        if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        ids
    };
    let gen = thinker.generate_greedy_text(&ids, &eos, max_new, device);
    let raw = tokenizer
        .decode(&gen, true)
        .map_err(|e| WhisburnError::Inference(e.to_string()))?;
    Ok(strip_think(&raw))
}

pub fn summarize_with_thinker(
    thinker: &Qwen3Thinker,
    tokenizer: &Gpt2Tokenizer,
    text: &str,
    device: &burn::tensor::Device,
    progress: Option<SummarizeProgress>,
) -> WhisburnResult<String> {
    let chunks = chunk_transcript(text, CHUNK_WORDS);
    if chunks.is_empty() {
        return Ok(String::new());
    }

    let n = chunks.len();
    let mut partials = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(cb) = &progress {
            cb(
                i,
                n + usize::from(n > 1),
                &format!("Summarizing section {} of {n}", i + 1),
            );
        }
        let user = if n == 1 {
            format!("Summarize this English transcript:\n\n{chunk}")
        } else {
            format!(
                "This is section {} of {n} of a long English transcript. Summarize only this section:\n\n{chunk}",
                i + 1
            )
        };
        let part = generate_completion(
            thinker,
            tokenizer,
            &build_prompt(&user),
            CHUNK_MAX_NEW_TOKENS,
            device,
        )?;
        if !part.is_empty() {
            partials.push(part);
        }
    }

    if partials.len() <= 1 {
        return Ok(partials.pop().unwrap_or_default());
    }

    if let Some(cb) = &progress {
        cb(n, n + 1, "Merging section summaries");
    }
    let joined = partials
        .iter()
        .enumerate()
        .map(|(i, p)| format!("Section {}:\n{p}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n");
    let merge_user = format!(
        "Combine these section summaries of a long English conversation into one concise summary with: overview, key points, decisions, action items.\n\n{joined}"
    );
    generate_completion(
        thinker,
        tokenizer,
        &build_prompt(&merge_user),
        MERGE_MAX_NEW_TOKENS,
        device,
    )
}

pub fn load_summarizer(
    model_name: &str,
    device: &burn::tensor::Device,
    verbose: bool,
) -> WhisburnResult<(Gpt2Tokenizer, Qwen3Thinker)> {
    let runtime = load_qwen3_runtime(model_name);
    let cfg = Qwen3ThinkerConfig {
        hidden_size: runtime.text_hidden_size,
        n_layers: runtime.text_layers,
        n_heads: runtime.text_heads,
        n_kv_heads: runtime.text_kv_heads,
        head_dim: runtime.text_head_dim,
        intermediate_size: runtime.text_intermediate_size,
        vocab_size: runtime.vocab_size,
        rms_norm_eps: runtime.rms_norm_eps,
        rope_theta: runtime.rope_theta,
        mrope_section: runtime.mrope_section,
    };
    let mut thinker = cfg.init(device);
    load_qwen3_lm_weights(&mut thinker, &pyre_whisper_core::resolve_model_dir(model_name), device, verbose)
        .map_err(|e| WhisburnError::Model(e.to_string()))?;
    let tokenizer = Gpt2Tokenizer::new(model_name)
        .map_err(|e| WhisburnError::Model(format!("summarizer tokenizer: {e}")))?;
    Ok((tokenizer, thinker))
}

#[cfg(test)]
mod tests {
    use super::{chunk_transcript, strip_think, CHUNK_WORDS};

    #[test]
    fn chunks_long_english_on_sentence_boundaries() {
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("This is sentence number {i}. "));
        }
        let chunks = chunk_transcript(&text, CHUNK_WORDS);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| !c.is_empty()));
        let total: usize = chunks.iter().map(|c| c.split_whitespace().count()).sum();
        assert_eq!(total, text.split_whitespace().count());
    }

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunk_transcript("Hello there. This is short.", 1800);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn strips_think_blocks() {
        let raw = "<think>\nplan\n</think>\n- Point one\n- Point two";
        assert_eq!(strip_think(raw), "- Point one\n- Point two");
    }
}
