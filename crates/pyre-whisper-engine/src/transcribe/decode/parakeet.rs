use burn::tensor::{Tensor};
use serde::Deserialize;
use std::path::Path;

use crate::model::Parakeet;
use crate::token::Gpt2Tokenizer;

#[derive(Debug, Deserialize)]
struct ParakeetDecodeConfig {
    blank_token_id: usize,
    vocab_size: usize,
    duration_start: usize,
    num_durations: usize,
    #[serde(default = "default_pad_token_id")]
    pad_token_id: usize,
    #[serde(default = "default_max_symbols")]
    max_symbols_per_step: usize,
    #[serde(default)]
    decoder_start_token_id: Option<usize>,
}

fn default_max_symbols() -> usize {
    10
}

impl ParakeetDecodeConfig {
    fn start_token(&self) -> usize {
        // HF generation feeds the blank id. v3's pad id is 2 and is not the blank.
        self.decoder_start_token_id.unwrap_or(self.blank_token_id)
    }
}

fn default_pad_token_id() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_start_token_is_the_blank_not_the_pad() {
        let cfg: ParakeetDecodeConfig = serde_json::from_str(
            r#"{"blank_token_id":8192,"vocab_size":8193,"duration_start":8193,"num_durations":5,"pad_token_id":2}"#,
        )
        .unwrap();
        assert_eq!(cfg.start_token(), 8192);
        assert_eq!(cfg.pad_token_id, 2);
        assert_eq!(cfg.max_symbols_per_step, 10);
    }
}

fn is_valid_parakeet_text_token(bpe: &Gpt2Tokenizer, token: usize, blank_token_id: usize) -> bool {
    if token >= blank_token_id {
        return false;
    }
    bpe.id_to_token(token as u32).is_some_and(|s| {
        !s.contains("spltoken")
            && !s.starts_with("<|")
            && s.chars().any(|c| c.is_ascii_alphabetic())
    })
}

fn is_leading_garbage_token(bpe: &Gpt2Tokenizer, token: usize) -> bool {
    bpe.id_to_token(token as u32)
        .map(|s| {
            let letters: String = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
            !(letters.len() >= 3 || s.starts_with("▁And") || s.starts_with("And"))
        })
        .unwrap_or(true)
}

fn dedup_consecutive(tokens: impl IntoIterator<Item = usize>) -> Vec<usize> {
    tokens.into_iter().fold(Vec::new(), |mut acc, t| {
        if acc.last() != Some(&t) {
            acc.push(t);
        }
        acc
    })
}

pub fn filter_parakeet_text_tokens(
    tokens: &[usize],
    blank_token_id: usize,
    bpe: &Gpt2Tokenizer,
) -> Vec<usize> {
    let collapsed = dedup_consecutive(
        tokens
            .iter()
            .copied()
            .filter(|&t| is_valid_parakeet_text_token(bpe, t, blank_token_id)),
    );

    let skip = collapsed
        .iter()
        .take_while(|&&t| is_leading_garbage_token(bpe, t))
        .count();
    collapsed.into_iter().skip(skip).collect()
}

/// Join Parakeet/SentencePiece BPE tokens into readable text.
pub fn parakeet_tokens_to_text(bpe: &Gpt2Tokenizer, tokens: &[usize]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for &token in tokens {
        let Some(piece) = bpe.id_to_token(token as u32) else {
            continue;
        };
        if piece.contains("spltoken") || piece.starts_with("<|") {
            continue;
        }
        if let Some(rest) = piece.strip_prefix('▁') {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(rest);
        } else {
            out.push_str(&piece);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn finalize_parakeet_decode(
    bpe: &Gpt2Tokenizer,
    tokens: &[usize],
    blank_token_id: usize,
    model_name: &str,
) -> (String, Vec<usize>) {
    let filtered = filter_parakeet_text_tokens(tokens, blank_token_id, bpe);
    let text = parakeet_tokens_to_text(bpe, &filtered);
    if !text.is_empty() && !is_garbage_parakeet_text(&text) {
        return (text, filtered);
    }
    let fallback = vocab_fallback_text(model_name, &filtered);
    if !fallback.is_empty() && !is_garbage_parakeet_text(&fallback) {
        return (fallback, filtered);
    }
    let bpe_text = bpe.decode(&filtered, true).unwrap_or_default();
    (bpe_text, filtered)
}

pub fn is_garbage_parakeet_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let letters = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let commas = trimmed.chars().filter(|c| *c == ',').count();
    letters == 0 || commas > letters
}

fn load_parakeet_decode_config(model_name: &str) -> ParakeetDecodeConfig {
    let path = pyre_whisper_core::resolve_model_file(model_name, "parakeet_decode.json");
    Path::new(&path)
        .exists()
        .then(|| std::fs::read_to_string(&path).ok())
        .flatten()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or(ParakeetDecodeConfig {
            blank_token_id: 8192,
            vocab_size: 8193,
            duration_start: 8193,
            num_durations: 5,
            pad_token_id: 2,
            max_symbols_per_step: 10,
            decoder_start_token_id: None,
        })
}

fn ctc_greedy_tokens(raw_ids: &[i32], blank_token: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for &id in raw_ids {
        let tok = id as usize;
        if prev == Some(tok) {
            continue;
        }
        prev = Some(tok);
        if tok != blank_token {
            out.push(tok);
        }
    }
    out
}

fn tdt_decode_tokens(
    raw_ids: &[i32],
    duration_ids: &[i32],
    blank_token: usize,
    vocab_size: usize,
) -> Vec<usize> {
    std::iter::successors(Some(0usize), |&t| {
        let dur = duration_ids.get(t).copied().unwrap_or(0) as usize;
        let next = t + dur + 1;
        (next < raw_ids.len()).then_some(next)
    })
    .map(|t| raw_ids[t] as usize)
    .filter(|&tok| tok != blank_token && tok < vocab_size)
    .collect()
}

fn count_token_classes(
    raw_ids: &[i32],
    blank_token: usize,
    duration_start: usize,
) -> (usize, usize, usize) {
    raw_ids.iter().map(|&id| id as usize).fold(
        (0usize, 0usize, 0usize),
        |(blank, dur, text), id| {
            if id == blank_token {
                (blank + 1, dur, text)
            } else if id >= duration_start {
                (blank, dur + 1, text)
            } else {
                (blank, dur, text + 1)
            }
        },
    )
}

pub fn decode_nemotron(
    model: &crate::model::nemotron::NemotronAsr,
    bpe: &Gpt2Tokenizer,
    mel: Tensor<3>,
    model_name: &str,
    _verbose: bool,
) -> (String, Vec<usize>) {
    let cfg = load_parakeet_decode_config(model_name);
    let tokens = model.decode_rnnt(
        mel,
        cfg.blank_token_id,
        cfg.vocab_size.max(1),
        cfg.max_symbols_per_step.max(1),
    );
    finalize_parakeet_decode(bpe, &tokens, cfg.blank_token_id, model_name)
}

pub fn decode_parakeet(
    parakeet: &Parakeet,
    bpe: &Gpt2Tokenizer,
    mel: Tensor<3>,
    model_name: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    if verbose {
        let max_feat = mel.clone().max().into_data().to_vec::<f32>().unwrap()[0];
        println!("DEBUG Parakeet Input: Max feature value: {}", max_feat);
    }

    let decode_cfg = load_parakeet_decode_config(model_name);

    if parakeet.has_decoder() {
        let decoded_tokens = if decode_cfg.num_durations == 0 {
            parakeet.decode_rnnt_greedy(
                mel.clone(),
                decode_cfg.blank_token_id,
                decode_cfg.vocab_size,
                decode_cfg.max_symbols_per_step.max(1),
                decode_cfg.start_token(),
            )
        } else {
            parakeet.decode_tdt_greedy(
                mel.clone(),
                decode_cfg.blank_token_id,
                decode_cfg.vocab_size,
                decode_cfg.duration_start,
                decode_cfg.num_durations,
                decode_cfg.start_token(),
                verbose,
            )
        };
        if verbose {
            println!(
                "DEBUG Parakeet TDT Decoded IDs: {:?}",
                &decoded_tokens[..20.min(decoded_tokens.len())]
            );
        }
        return finalize_parakeet_decode(
            bpe,
            &decoded_tokens,
            decode_cfg.blank_token_id,
            model_name,
        );
    }

    let logits = parakeet.forward(mel.clone());
    let [_, n_frames, v_size] = logits.dims();

    if decode_cfg.num_durations == 0 {
        let raw_ids = logits
            .argmax(2)
             .squeeze_dim::<2>(0)
            .into_data()
            .to_vec::<i32>()
            .unwrap();
        if verbose {
            println!("DEBUG Parakeet CTC Raw IDs: {:?}", &raw_ids[..20.min(raw_ids.len())]);
        }
        let decoded_tokens = ctc_greedy_tokens(&raw_ids, decode_cfg.blank_token_id);
        return finalize_parakeet_decode(
            bpe,
            &decoded_tokens,
            decode_cfg.blank_token_id,
            model_name,
        );
    }

    if verbose {
        let max_logit = logits.clone().max().into_data().to_vec::<f32>().unwrap()[0];
        println!("DEBUG Parakeet: Max logit value: {}", max_logit);
    }

    let token_limit = decode_cfg.vocab_size.min(v_size);
    let token_logits = logits.clone().slice([0..1, 0..n_frames, 0..token_limit]);

    if verbose {
        (0..n_frames).for_each(|f| {
            let frame_logits = token_logits.clone().slice([0..1, f..f + 1, 0..v_size]);
            let top_5 = frame_logits
                .clone()
                .argsort(2)
                 .squeeze_dim::<2>(0)
                .into_data()
                .to_vec::<i32>()
                .unwrap();
            let start = top_5.len().saturating_sub(5);
            println!("DEBUG Parakeet Frame {}: Top-5 IDs: {:?}", f, &top_5[start..]);
        });
    }

    let raw_ids = token_logits
        .argmax(2)
         .squeeze_dim::<2>(0)
        .into_data()
        .to_vec::<i32>()
        .unwrap();

    if verbose {
        println!("DEBUG Parakeet Raw IDs: {:?}", &raw_ids);
    }

    let duration_ids = (decode_cfg.num_durations > 0 && v_size > decode_cfg.duration_start)
        .then(|| {
            let d_limit = (decode_cfg.duration_start + decode_cfg.num_durations).min(v_size);
            logits
                .slice([0..1, 0..n_frames, decode_cfg.duration_start..d_limit])
                .argmax(2)
                 .squeeze_dim::<2>(0)
                .into_data()
                .to_vec::<i32>()
                .unwrap()
        })
        .unwrap_or_else(|| vec![0; n_frames]);

    if verbose {
        let (n_text, n_dur, n_blank) = count_token_classes(
            &raw_ids,
            decode_cfg.blank_token_id,
            decode_cfg.duration_start,
        );
        println!(
            "DEBUG Parakeet Tokens: text={}, dur={}, blank={}",
            n_text, n_dur, n_blank
        );
    }

    let decoded_tokens = tdt_decode_tokens(
        &raw_ids,
        &duration_ids,
        decode_cfg.blank_token_id,
        decode_cfg.vocab_size,
    );

    if verbose {
        println!(
            "DEBUG Parakeet Decoded IDs: {:?}",
            &decoded_tokens[..20.min(decoded_tokens.len())]
        );
    }

    finalize_parakeet_decode(
        bpe,
        &decoded_tokens,
        decode_cfg.blank_token_id,
        model_name,
    )
}

fn vocab_fallback_text(model_name: &str, decoded_tokens: &[usize]) -> String {
    let vocab_path = pyre_whisper_core::resolve_model_file(model_name, "vocab.txt");
    std::fs::read_to_string(vocab_path)
        .ok()
        .map(|content| {
            let vocab: Vec<&str> = content.lines().collect();
            decoded_tokens
                .iter()
                .filter_map(|&t| vocab.get(t).copied())
                .map(|word| word.replace('▁', " "))
                .collect::<String>()
                .trim()
                .to_string()
        })
        .unwrap_or_default()
}