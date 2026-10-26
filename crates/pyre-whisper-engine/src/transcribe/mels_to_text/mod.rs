mod heuristics;
mod prompt;

use crate::beam::{beam_search, BeamNode};
use crate::model::*;
use crate::token::{self, *};
use burn::tensor::{
    activation::log_softmax,
    ElementConversion, Int, Tensor, TensorData,
};
use std::iter;

pub use prompt::is_whisper_english_only;

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

    let clip_len = n_ctx.min(n_ctx_max_encoder_frames);

    if n_ctx > n_ctx_max_encoder_frames {
        println!(
            "Audio has length of {} frames which exceeds maximum length {}. It will be clipped.",
            n_ctx, n_ctx_max_encoder_frames
        );
    }

    let mels = if clip_len < n_ctx_max_encoder_frames {
        let padding_needed = n_ctx_max_encoder_frames - clip_len;
        Tensor::cat(
            vec![
                mels.slice([0..1, 0..n_mel, 0..clip_len]),
                Tensor::zeros([1, n_mel, padding_needed], &device),
            ],
            2,
        )
    } else {
        mels.slice([0..1, 0..n_mel, 0..n_ctx_max_encoder_frames])
    };

    let encoder_output = whisper.forward_encoder(mels);
    let end_token = end_token(bpe);
    let initial_tokens_vec = build_initial_tokens(bpe, lang, decode_task, model_name, include_timestamps);
    let prompt_len = initial_tokens_vec.len();

    type BeamNodeTy = BeamNode<BeamSearchToken>;
    let initial_tokens = BeamNodeTy {
        seq: initial_tokens_vec
            .into_iter()
            .map(|tok| BeamSearchToken {
                token: tok,
                log_prob: 0.0,
            })
            .collect(),
        log_prob: 0.0,
    };

    let vocab_size = whisper.decoder.n_vocab;
    let special_tokens_maskout = special_tokens_maskout(bpe, vocab_size, end_token, include_timestamps);
    let decode_cfg = load_whisper_decode_config(model_name);

    let beamsearch_next = |beams: &[BeamNodeTy]| {
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
            &device,
        );

        let logits = whisper.forward_decoder(
            token_tensor,
            encoder_output.clone().repeat(&[beams.len(), 1, 1]),
        );

        let logits = if max_seq_len >= prompt_len {
            let mask_vec = build_logit_mask(
                vocab_size,
                &special_tokens_maskout,
                &decode_cfg,
                at_generation_start(max_seq_len, prompt_len),
                end_token,
            );
            let mask = Tensor::<1>::from_floats(mask_vec.as_slice(), &device);
            logits + mask.unsqueeze::<3>()
        } else {
            logits
        };

        let log_probs = log_softmax(logits, 2);

        beams
            .iter()
            .enumerate()
            .map(|(i, beam)| {
                let batch = i;
                let token_index = beam.seq.len() - 1;
                log_probs
                    .clone()
                    .slice([batch..batch + 1, token_index..token_index + 1])
                    .flatten::<1>(0, 2)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            })
            .zip(beams)
            .map(|(log_probs, beam)| {
                log_probs
                    .into_iter()
                    .map(|log_prob| log_prob.elem::<f64>())
                    .enumerate()
                    .map(|(token_id, log_prob)| {
                        (
                            BeamSearchToken {
                                token: token_id,
                                log_prob,
                            },
                            beam.log_prob + log_prob,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let beamsearch_is_finished = |toks: &[BeamSearchToken]| beamsearch_is_finished(toks, end_token);

    let max_depth = max_depth.min(whisper.decoder_ctx_size() - initial_tokens.seq.len());
    let tokens: Vec<_> = beam_search(
        vec![initial_tokens],
        beamsearch_next,
        beamsearch_is_finished,
        beam_size,
        max_depth,
    )
    .into_iter()
    .map(|btok| btok.token)
    .collect();

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