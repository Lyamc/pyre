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
    /// Duration table. The argmax index is looked up here (`[0, 1, 2, 3, 4]` on v2/v3).
    /// Missing on older decode files: `0..num_durations`.
    #[serde(default)]
    durations: Vec<usize>,
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
    fn sentencepiece_keeps_prefixes_boundaries_and_bytes() {
        assert!(keep_parakeet_piece("▁O"));
        assert!(keep_parakeet_piece("▁"));
        assert!(keep_parakeet_piece("<0xC3>"));
        assert!(keep_parakeet_piece("'"));
        assert!(!keep_parakeet_piece("<|en|>"));

        assert_eq!(pieces_to_text(["▁O", "ther"]), "Other");
        assert_eq!(pieces_to_text(["▁Not", "withstanding"]), "Notwithstanding");
        assert_eq!(pieces_to_text(["▁C", "ome", "▁c", "ome"]), "Come come");
        assert_eq!(pieces_to_text(["▁Sh", "e"]), "She");
        assert_eq!(pieces_to_text(["▁", "the", "▁", "utter"]), "the utter");
        assert_eq!(pieces_to_text(["▁the", "▁utter"]), "the utter");
        assert_eq!(
            pieces_to_text(["p", "<0xC3>", "<0xAA>", "cheur"]),
            "pêcheur"
        );
        assert_eq!(pieces_to_text(["j", "'", "ai"]), "j'ai");
        assert_eq!(pieces_to_text(["l", "l"]), "ll");
        assert!(!is_garbage_parakeet_text("pêcheur"));
    }

    #[test]
    fn missing_start_token_is_the_blank_not_the_pad() {
        let cfg: ParakeetDecodeConfig = serde_json::from_str(
            r#"{"blank_token_id":8192,"vocab_size":8193,"duration_start":8193,"num_durations":5,"pad_token_id":2}"#,
        )
        .unwrap();
        assert_eq!(cfg.start_token(), 8192);
        assert_eq!(cfg.pad_token_id, 2);
        assert_eq!(cfg.max_symbols_per_step, 10);
        assert!(cfg.durations.is_empty());
        assert_eq!(cfg.duration_table(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn duration_index_is_looked_up_and_a_blank_zero_advances() {
        let table = [0, 2, 4, 8];
        assert_eq!(tdt_step(false, 2, &table, 0, 10), (4, 1));
        assert_eq!(tdt_step(true, 0, &table, 3, 10), (1, 0));
        assert_eq!(tdt_step(false, 0, &table, 9, 10), (1, 0));
    }

    #[test]
    fn word_spans_follow_sentencepiece_boundaries() {
        let pieces = vec!["▁Hel".into(), "lo".into(), "▁there".into()];
        let frames = vec![1usize, 1, 4];
        let spans = word_spans_from_pieces(&pieces, &frames, 0.0, 0.08);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].2, "Hello");
        assert!((spans[0].0 - 0.08).abs() < 1e-6);
        assert!((spans[1].0 - 0.32).abs() < 1e-6);
        assert!((spans[0].1 - spans[1].0).abs() < 1e-6);
    }

}

/// SentencePiece pieces that are not blank and not Whisper-style specials.
/// Short pieces (`▁O`, `e`), the word-boundary token `▁`, apostrophes, and
/// byte-fallback (`<0xC3>`) are text. Dropping them turned "Other" into "ther",
/// "She" into nothing, "pêcheur" into "pcheur", and "the utter" into "theutter".
pub fn keep_parakeet_piece(piece: &str) -> bool {
    if piece.is_empty() || piece.contains("spltoken") {
        return false;
    }
    if piece.starts_with("<|") && piece.ends_with("|>") {
        return false;
    }
    true
}

fn is_text_token(bpe: &Gpt2Tokenizer, token: usize, blank_token_id: usize) -> bool {
    if token >= blank_token_id {
        return false;
    }
    bpe.id_to_token(token as u32)
        .is_some_and(|s| keep_parakeet_piece(&s))
}

pub fn filter_parakeet_text_tokens(
    tokens: &[usize],
    blank_token_id: usize,
    bpe: &Gpt2Tokenizer,
) -> Vec<usize> {
    // TDT/RNNT may emit the same piece twice ("ll"). CTC already collapsed repeats.
    tokens
        .iter()
        .copied()
        .filter(|&t| is_text_token(bpe, t, blank_token_id))
        .collect()
}

/// Join SentencePiece pieces. `▁` (U+2581) is a word boundary, including when it
/// is its own token. `<0xHH>` runs are UTF-8 byte fallback.
pub fn pieces_to_text(pieces: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let mut raw = String::new();
    for piece in pieces {
        let piece = piece.as_ref();
        if !keep_parakeet_piece(piece) {
            continue;
        }
        if let Some(rest) = piece.strip_prefix('▁') {
            if !raw.is_empty() && !raw.ends_with(' ') {
                raw.push(' ');
            }
            raw.push_str(rest);
        } else {
            raw.push_str(piece);
        }
    }
    decode_sp_byte_fallback(&raw)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn decode_sp_byte_fallback(text: &str) -> String {
    let mut out = String::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(hex) = rest.strip_prefix("<0x").or_else(|| rest.strip_prefix("<0X")) {
            if hex.len() >= 3 && hex.as_bytes()[2] == b'>' {
                if let Ok(byte) = u8::from_str_radix(&hex[..2], 16) {
                    pending.push(byte);
                    rest = &hex[3..];
                    continue;
                }
            }
        }
        if !pending.is_empty() {
            out.push_str(&String::from_utf8_lossy(&pending));
            pending.clear();
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    if !pending.is_empty() {
        out.push_str(&String::from_utf8_lossy(&pending));
    }
    out
}

/// Join Parakeet/SentencePiece BPE tokens into readable text.
pub fn parakeet_tokens_to_text(bpe: &Gpt2Tokenizer, tokens: &[usize]) -> String {
    let pieces: Vec<String> = tokens
        .iter()
        .filter_map(|&token| bpe.id_to_token(token as u32))
        .collect();
    pieces_to_text(pieces.iter().map(String::as_str))
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
    let letters = trimmed.chars().filter(|c| c.is_alphabetic()).count();
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
            durations: vec![0, 1, 2, 3, 4],
        })
}

impl ParakeetDecodeConfig {
    fn duration_table(&self) -> Vec<usize> {
        if self.durations.is_empty() {
            (0..self.num_durations).collect()
        } else {
            self.durations.clone()
        }
    }
}

/// How far the TDT frame pointer moves, and the updated per-frame symbol count.
///
/// NeMo looks the duration *index* up in `durations` (not the raw index). A blank
/// with duration 0 is forced to 1. A run of zero-duration labels stops at
/// `max_symbols` and then advances one frame so the loop cannot sit on one frame.
pub fn tdt_step(token_is_blank: bool, dur_index: usize, durations: &[usize], symbols_at_frame: usize, max_symbols: usize) -> (usize, usize) {
    let mut skip = durations.get(dur_index).copied().unwrap_or(dur_index);
    let mut symbols = if token_is_blank { 0 } else { symbols_at_frame.saturating_add(1) };
    if token_is_blank && skip == 0 {
        skip = 1;
    }
    if !token_is_blank && skip == 0 && symbols >= max_symbols.max(1) {
        skip = 1;
        symbols = 0;
    }
    (skip, symbols)
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

pub struct TimedChunk {
    pub text: String,
    pub tokens: Vec<usize>,
    pub frames: Vec<usize>,
    pub encoder_frames: usize,
}

fn with_frames(text: String, tokens: Vec<usize>, raw_ids: &[usize], raw_frames: &[usize], encoder_frames: usize) -> TimedChunk {
    let mut frames = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    for &tok in &tokens {
        if let Some(pos) = raw_ids[cursor..].iter().position(|&id| id == tok) {
            let at = cursor + pos;
            if at < raw_frames.len() {
                frames.push(raw_frames[at]);
            }
            cursor = at + 1;
        }
    }
    TimedChunk { text, tokens, frames, encoder_frames }
}

pub fn decode_nemotron(
    model: &crate::model::nemotron::NemotronAsr,
    bpe: &Gpt2Tokenizer,
    mel: Tensor<3>,
    model_name: &str,
    _verbose: bool,
) -> TimedChunk {
    let cfg = load_parakeet_decode_config(model_name);
    let (tokens, frames, encoder_frames) = model.decode_rnnt(
        mel,
        cfg.blank_token_id,
        cfg.vocab_size.max(1),
        cfg.max_symbols_per_step.max(1),
    );
    let (text, kept) = finalize_parakeet_decode(bpe, &tokens, cfg.blank_token_id, model_name);
    with_frames(text, kept, &tokens, &frames, encoder_frames)
}

/// Group SentencePiece pieces into words and stamp them from TDT encoder frames.
/// One frame is 80 ms: 10 ms mel hop times the 8× FastConformer subsampling.
pub fn word_spans_from_pieces(
    pieces: &[String],
    frames: &[usize],
    origin_secs: f64,
    frame_secs: f64,
) -> Vec<(f64, f64, String)> {
    if pieces.len() != frames.len() || pieces.is_empty() {
        return Vec::new();
    }
    let mut words: Vec<(usize, String)> = Vec::new();
    for (piece, frame) in pieces.iter().zip(frames) {
        if piece.starts_with('▁') {
            words.push((*frame, piece.trim_start_matches('▁').to_string()));
        } else if let Some(last) = words.last_mut() {
            last.1.push_str(piece);
        } else {
            words.push((*frame, piece.clone()));
        }
    }
    let mut out = Vec::with_capacity(words.len());
    for (i, (frame, raw)) in words.iter().enumerate() {
        let start = origin_secs + *frame as f64 * frame_secs;
        let end = words
            .get(i + 1)
            .map(|(next, _)| (origin_secs + *next as f64 * frame_secs).max(start + 0.02))
            .unwrap_or(start + frame_secs);
        let text = pieces_to_text([raw.as_str()]);
        if !text.is_empty() {
            out.push((start, end, text));
        }
    }
    out
}

pub fn decode_parakeet(
    parakeet: &Parakeet,
    bpe: &Gpt2Tokenizer,
    mel: Tensor<3>,
    model_name: &str,
    verbose: bool,
) -> TimedChunk {
    if verbose {
        let max_feat = mel.clone().max().into_data().to_vec::<f32>().unwrap()[0];
        println!("DEBUG Parakeet Input: Max feature value: {}", max_feat);
    }

    let decode_cfg = load_parakeet_decode_config(model_name);

    if parakeet.has_decoder() {
        let (raw_ids, raw_frames, encoder_frames) = if decode_cfg.num_durations == 0 {
            let ids = parakeet.decode_rnnt_greedy(
                mel.clone(),
                decode_cfg.blank_token_id,
                decode_cfg.vocab_size,
                decode_cfg.max_symbols_per_step.max(1),
                decode_cfg.start_token(),
            );
            (ids, Vec::new(), 0)
        } else {
            let (pairs, encoder_frames) = parakeet.decode_tdt_greedy(
                mel.clone(),
                decode_cfg.blank_token_id,
                decode_cfg.vocab_size,
                decode_cfg.duration_start,
                &decode_cfg.duration_table(),
                decode_cfg.max_symbols_per_step.max(1),
                decode_cfg.start_token(),
                verbose,
            );
            let (ids, frames): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
            (ids, frames, encoder_frames)
        };
        if verbose {
            println!(
                "DEBUG Parakeet TDT Decoded IDs: {:?}",
                &raw_ids[..20.min(raw_ids.len())]
            );
        }
        let (text, kept) = finalize_parakeet_decode(
            bpe,
            &raw_ids,
            decode_cfg.blank_token_id,
            model_name,
        );
        return with_frames(text, kept, &raw_ids, &raw_frames, encoder_frames);
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
        let (text, kept) = finalize_parakeet_decode(
            bpe,
            &decoded_tokens,
            decode_cfg.blank_token_id,
            model_name,
        );
        return with_frames(text, kept, &decoded_tokens, &[], 0);
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

    let (text, kept) = finalize_parakeet_decode(
        bpe,
        &decoded_tokens,
        decode_cfg.blank_token_id,
        model_name,
    );
    with_frames(text, kept, &decoded_tokens, &[], 0)
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