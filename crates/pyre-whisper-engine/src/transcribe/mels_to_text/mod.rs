mod heuristics;
mod prompt;

use crate::beam::{beam_search, get_top_elements, BeamNode};
use crate::model::decoder::DecoderKv;
use crate::model::*;
use crate::token::{self, *};
use burn::tensor::{
    activation::log_softmax,
    ElementConversion, Int, Tensor, TensorData,
};
use std::iter;

pub use prompt::is_whisper_english_only;

use super::pass::{DecoderKind, MelTail};

/// Mel frames the encoder should see for this tail policy.
/// `ZeroPad` is the historical path (always `n_audio_ctx * 2`, clipped if longer).
pub(crate) fn encoder_mel_len(actual_frames: usize, n_audio_ctx: usize, tail: MelTail) -> usize {
    let max_frames = n_audio_ctx.saturating_mul(2).max(1);
    match tail {
        MelTail::ZeroPad => max_frames.min(actual_frames.max(max_frames)),
        MelTail::Truncate => actual_frames.min(max_frames).max(1),
    }
}

use super::DecodeTask;
use super::whisper_config::{build_logit_mask, load_whisper_decode_config};
use heuristics::{at_generation_start, beamsearch_is_finished, special_tokens_maskout};
use prompt::{build_initial_tokens, end_token};

#[derive(Clone)]
pub struct BeamSearchToken {
    pub token: usize,
    pub log_prob: f64,
}

pub fn mels_to_text(
    whisper: &Whisper,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    mels: Tensor<3>,
    _padding: usize,
    _streaming_mode: bool,
    include_timestamps: bool,
    beam_size: usize,
    max_depth: usize,
    decode_task: DecodeTask,
    model_name: &str,
) -> token::Result<(String, Vec<usize>)> {
    let device = mels.device();
    let n_ctx_max_encoder_frames = whisper.encoder_ctx_size() * 2;
    let [_, n_mel, n_ctx] = mels.dims();
    let tail = super::pass::current().tail;
    let target = encoder_mel_len(n_ctx, whisper.encoder_ctx_size(), tail);

    if n_ctx > n_ctx_max_encoder_frames {
        println!(
            "Audio has length of {} frames which exceeds maximum length {}. It will be clipped.",
            n_ctx, n_ctx_max_encoder_frames
        );
    }

    // Historical Whisper: pad with zeros in log-mel space up to one context (3000
    // frames on tiny/base). Those frames still receive absolute positional
    // embeddings inside the encoder, so dropping them changes the transcript.
    // A key mask is not a substitute: it would stop real frames from attending
    // to that positional tail.
    let mels = if n_ctx >= target {
        mels.slice([0..1, 0..n_mel, 0..target])
    } else {
        let padding_needed = target - n_ctx;
        Tensor::cat(
            vec![
                mels.slice([0..1, 0..n_mel, 0..n_ctx]),
                Tensor::zeros([1, n_mel, padding_needed], &device),
            ],
            2,
        )
    };

    let encoder_output = whisper.forward_encoder(mels);
    let end_token = end_token(bpe);
    let initial_tokens_vec = build_initial_tokens(bpe, lang, decode_task, model_name, include_timestamps);
    let prompt_len = initial_tokens_vec.len();

    let initial_seq: Vec<BeamSearchToken> = initial_tokens_vec
        .into_iter()
        .map(|tok| BeamSearchToken { token: tok, log_prob: 0.0 })
        .collect();

    let vocab_size = whisper.decoder.n_vocab;
    let special_tokens_maskout = special_tokens_maskout(bpe, vocab_size, end_token, include_timestamps);
    let decode_cfg = load_whisper_decode_config(model_name);

    let max_depth = max_depth.min(whisper.decoder_ctx_size() - initial_seq.len());
    let finished = |toks: &[BeamSearchToken]| beamsearch_is_finished(toks, end_token);
    let tokens = match super::pass::current().decoder {
        DecoderKind::Eager => eager_beam(
            whisper,
            &encoder_output,
            &device,
            initial_seq,
            prompt_len,
            vocab_size,
            &special_tokens_maskout,
            &decode_cfg,
            end_token,
        beam_size,
        max_depth,
    ),
        DecoderKind::Cached => beam_decode(
            whisper,
            &encoder_output,
            &device,
            initial_seq,
            prompt_len,
            vocab_size,
            &special_tokens_maskout,
            &decode_cfg,
            end_token,
            &finished,
            beam_size,
            max_depth,
        ),
    };

    let text_tokens: Vec<usize> = tokens
        .get(prompt_len..)
        .unwrap_or(&[])
        .iter()
        .copied()
        .take_while(|&t| t != end_token)
        .collect();

    let full_text = bpe.decode(&text_tokens, true)?;
    let final_text = super::utils::trim_trailing_hallucination(&full_text);

    let mut out_tokens = tokens;
    if let Some(eos_offset) = out_tokens
        .get(prompt_len..)
        .and_then(|tail| tail.iter().position(|&t| t == end_token))
    {
        out_tokens.truncate(prompt_len + eos_offset);
    }

    Ok((final_text, out_tokens))
}

fn eager_beam(
    whisper: &Whisper,
    encoder_output: &Tensor<3>,
    device: &burn::tensor::Device,
    initial_seq: Vec<BeamSearchToken>,
    prompt_len: usize,
    vocab_size: usize,
    special_tokens_maskout: &[f32],
    decode_cfg: &super::whisper_config::WhisperDecodeConfig,
    end_token: usize,
    beam_size: usize,
    max_depth: usize,
) -> Vec<usize> {
    let initial = BeamNode { seq: initial_seq, log_prob: 0.0 };
    let next = |beams: &[BeamNode<BeamSearchToken>]| {
        let max_seq_len = beams.iter().map(|beam| beam.seq.len()).max().unwrap_or(0);
        let flattened_tokens: Vec<i64> = beams
            .iter()
            .flat_map(|beam| {
                let additional_tokens = max_seq_len - beam.seq.len();
                beam.seq
                    .iter()
                    .map(|btok| btok.token as i64)
                    .chain(iter::once(0).cycle().take(additional_tokens))
            })
            .collect();
        let token_tensor = Tensor::<2, Int>::from_ints(
            TensorData::new(flattened_tokens, [beams.len(), max_seq_len]),
            device,
        );
        let logits = whisper.forward_decoder(
            token_tensor,
            encoder_output.clone().repeat(&[beams.len(), 1, 1]),
        );
        let logits = if max_seq_len >= prompt_len {
            let mask_vec = build_logit_mask(
                vocab_size,
                special_tokens_maskout,
                decode_cfg,
                at_generation_start(max_seq_len, prompt_len),
                end_token,
            );
            let mask = Tensor::<1>::from_floats(mask_vec.as_slice(), device);
            logits + mask.unsqueeze::<3>()
        } else {
            logits
        };
        let log_probs = log_softmax(logits, 2);
        beams
            .iter()
            .enumerate()
            .map(|(i, beam)| {
                let token_index = beam.seq.len() - 1;
                log_probs
                    .clone()
                    .slice([i..i + 1, token_index..token_index + 1])
                    .flatten::<1>(0, 2)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            })
            .zip(beams)
            .map(|(row, beam)| {
                row.into_iter()
                    .map(|log_prob| log_prob.elem::<f64>())
                    .enumerate()
                    .map(|(token_id, log_prob)| {
                        (
                            BeamSearchToken { token: token_id, log_prob },
                            beam.log_prob + log_prob,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let is_finished = |toks: &[BeamSearchToken]| beamsearch_is_finished(toks, end_token);
    beam_search(vec![initial], next, is_finished, beam_size, max_depth)
        .into_iter()
        .map(|tok| tok.token)
        .collect()
}

#[derive(Clone)]
struct Hyp {
    seq: Vec<BeamSearchToken>,
    log_prob: f64,
    kv: Option<DecoderKv>,
}

/// Beam search that reuses decoder self-attention and encoder cross-attention
/// state. Finished hypotheses are not forwarded again. Scoring matches the
/// previous full-sequence decode: the next-token distribution is the last
/// position, with the same logit mask.
fn beam_decode(
    whisper: &Whisper,
    encoder_output: &Tensor<3>,
    device: &burn::tensor::Device,
    initial_seq: Vec<BeamSearchToken>,
    prompt_len: usize,
    vocab_size: usize,
    special_tokens_maskout: &[f32],
    decode_cfg: &super::whisper_config::WhisperDecodeConfig,
    end_token: usize,
    is_finished: &dyn Fn(&[BeamSearchToken]) -> bool,
    beam_size: usize,
    max_depth: usize,
) -> Vec<usize> {
    let beam_size = beam_size.max(1);
    let mut hyps = vec![Hyp { seq: initial_seq, log_prob: 0.0, kv: None }];

    for _ in 0..max_depth {
        let best_finished = hyps
            .iter()
            .filter(|h| is_finished(&h.seq))
            .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(std::cmp::Ordering::Equal));
        let best_unfinished = hyps
            .iter()
            .filter(|h| !is_finished(&h.seq))
            .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(std::cmp::Ordering::Equal));
        let stop = match (best_finished, best_unfinished) {
            (Some(_), None) => true,
            (Some(finished), Some(unfinished)) => finished.log_prob >= unfinished.log_prob,
            _ => false,
        };
        if stop {
            break;
        }

        let dists = score_open(
            whisper,
            encoder_output,
            device,
            &mut hyps,
            prompt_len,
            vocab_size,
            special_tokens_maskout,
            decode_cfg,
            end_token,
            is_finished,
        );
        hyps = expand(hyps, dists, is_finished, beam_size);
        if hyps.is_empty() {
            break;
        }
    }

    let finished: Vec<_> = hyps.iter().filter(|h| is_finished(&h.seq)).collect();
    let best = if !finished.is_empty() {
        finished
            .into_iter()
            .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(std::cmp::Ordering::Equal))
    } else {
        hyps.iter()
            .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(std::cmp::Ordering::Equal))
    };
    best.map(|h| h.seq.iter().map(|t| t.token).collect())
        .unwrap_or_default()
}

fn expand(
    hyps: Vec<Hyp>,
    dists: Vec<Vec<(BeamSearchToken, f64)>>,
    is_finished: &dyn Fn(&[BeamSearchToken]) -> bool,
    beam_size: usize,
) -> Vec<Hyp> {
    let mut finished_beams = Vec::new();
    let mut fresh = Vec::new();
    for (hyp, dist) in hyps.into_iter().zip(dists) {
        if is_finished(&hyp.seq) {
            finished_beams.push(hyp);
        } else {
            let top = get_top_elements(&dist, |(_, log_prob)| *log_prob, beam_size);
            for (tok, log_prob) in top {
                fresh.push(Hyp {
                    seq: [hyp.seq.clone(), vec![tok.clone()]].concat(),
                    log_prob: *log_prob,
                    kv: hyp.kv.clone(),
                });
            }
        }
    }
    get_top_elements(&fresh, |h| h.log_prob, beam_size)
        .into_iter()
        .cloned()
        .chain(
            get_top_elements(&finished_beams, |h| h.log_prob, beam_size)
                .into_iter()
                .cloned(),
        )
        .collect()
}

fn score_open(
    whisper: &Whisper,
    encoder_output: &Tensor<3>,
    device: &burn::tensor::Device,
    hyps: &mut [Hyp],
    prompt_len: usize,
    vocab_size: usize,
    special_tokens_maskout: &[f32],
    decode_cfg: &super::whisper_config::WhisperDecodeConfig,
    end_token: usize,
    is_finished: &dyn Fn(&[BeamSearchToken]) -> bool,
) -> Vec<Vec<(BeamSearchToken, f64)>> {
    let mut dists = vec![Vec::new(); hyps.len()];
    let open: Vec<usize> = hyps
        .iter()
        .enumerate()
        .filter(|(_, h)| !is_finished(&h.seq) && !h.seq.is_empty())
        .map(|(i, _)| i)
        .collect();
    if open.is_empty() {
        return dists;
    }

    let prefill = hyps[open[0]].kv.is_none();
    let (token_tensor, past) = if prefill {
        let idx = open[0];
        let ids: Vec<i64> = hyps[idx].seq.iter().map(|t| t.token as i64).collect();
        let seq = ids.len();
        let tensor = Tensor::<2, Int>::from_ints(TensorData::new(ids, [1, seq]), device);
        (tensor, None)
    } else {
        let kvs: Vec<DecoderKv> = open.iter().map(|&i| hyps[i].kv.clone().expect("kv")).collect();
        let past = if kvs.len() == 1 {
            kvs.into_iter().next()
        } else {
            Some(DecoderKv::concat_batch(&kvs))
        };
        let ids: Vec<i64> = open.iter().map(|&i| hyps[i].seq.last().unwrap().token as i64).collect();
        let n = ids.len();
        let tensor = Tensor::<2, Int>::from_ints(TensorData::new(ids, [n, 1]), device);
        (tensor, past)
    };

    let (logits, new_kv) = whisper.forward_decoder_cached(token_tensor, encoder_output.clone(), past);
    let seq_for_mask = if prefill { hyps[open[0]].seq.len() } else { hyps[open[0]].seq.len() };
    let logits = if seq_for_mask >= prompt_len {
        let mask_vec = build_logit_mask(
            vocab_size,
            special_tokens_maskout,
            decode_cfg,
            at_generation_start(seq_for_mask, prompt_len),
            end_token,
        );
        let mask = Tensor::<1>::from_floats(mask_vec.as_slice(), device);
        logits + mask.unsqueeze::<3>()
    } else {
        logits
    };
    let log_probs = log_softmax(logits, 2);
    let [rows, _, vocab] = log_probs.dims();
    let flat = log_probs.into_data().to_vec::<f32>().expect("whisper logprobs");
    let parts = new_kv.split_batch();
    let scored = if prefill { 1 } else { open.len() };
    let rows = rows.min(scored).min(parts.len());
    for batch_i in 0..rows {
        let idx = if prefill { open[0] } else { open[batch_i] };
        let start = batch_i * vocab;
        let row = &flat[start..start + vocab];
        dists[idx] = row
            .iter()
            .enumerate()
            .map(|(token, lp)| {
                let log_prob = lp.elem::<f64>();
                (
                    BeamSearchToken { token, log_prob },
                    hyps[idx].log_prob + log_prob,
                )
            })
            .collect();
        hyps[idx].kv = Some(parts[batch_i].clone());
    }
    dists
}