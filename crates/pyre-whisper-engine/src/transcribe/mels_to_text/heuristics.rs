use crate::token::Gpt2Tokenizer;

use super::BeamSearchToken;

pub fn special_tokens_maskout(
    bpe: &Gpt2Tokenizer,
    vocab_size: usize,
    end_token: usize,
    include_timestamps: bool,
) -> Vec<f32> {
    let neg_infty = f32::NEG_INFINITY;
    (0..vocab_size)
        .map(|token| {
            if token == end_token {
                0.0
            } else if bpe.is_special(token) {
                if include_timestamps && bpe.is_timestamp(token) {
                    0.0
                } else {
                    neg_infty
                }
            } else {
                0.0
            }
        })
        .collect()
}

pub fn beamsearch_is_finished(toks: &[BeamSearchToken], end_token: usize) -> bool {
    if toks.len() >= 100 {
        return true;
    }
    let Some(btok) = toks.last() else {
        return false;
    };
    if btok.token == end_token {
        return true;
    }

    let n = toks.len();
    if n > 20 {
        let repeated = (2..10).any(|window_size| {
            if n < window_size * 2 {
                return false;
            }
            let last_window = &toks[n - window_size..];
            let prev_window = &toks[n - window_size * 2..n - window_size];
            last_window
                .iter()
                .zip(prev_window.iter())
                .all(|(a, b)| a.token == b.token)
        });
        if repeated {
            return true;
        }
    }

    if n > 10 {
        let last_3 = &toks[n - 3..];
        if last_3.iter().all(|t| t.token == last_3[0].token) {
            return true;
        }
    }

    false
}

pub fn at_generation_start(max_seq_len: usize, prompt_len: usize) -> bool {
    max_seq_len >= prompt_len && max_seq_len == prompt_len
}