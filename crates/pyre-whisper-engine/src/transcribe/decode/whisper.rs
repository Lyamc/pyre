use burn::tensor::{Tensor};
use std::error::Error;

use crate::model::Whisper;
use crate::token::{Gpt2Tokenizer, Language};
use crate::transcribe::mels_to_text::mels_to_text;

use super::DecodeTask;

pub fn decode_whisper(
    whisper: &Whisper,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    mel: Tensor<3>,
    padding: usize,
    streaming_mode: bool,
    include_timestamps: bool,
    beam_size: usize,
    max_depth: usize,
    decode_task: DecodeTask,
    model_name: &str,
) -> Result<(String, Vec<usize>, Vec<usize>), Box<dyn Error + Send + Sync>> {
    let (text, tokens) = mels_to_text(
        whisper,
        bpe,
        lang,
        mel,
        padding,
        streaming_mode,
        include_timestamps,
        beam_size,
        max_depth,
        decode_task,
        model_name,
    )?;
    // Carry each `<|t|>` timestamp forward to the text tokens it introduces. The chunk
    // stitcher uses these frames to drop the left-context overlap instead of matching words.
    let mut frame = 0usize;
    let frames = tokens
        .iter()
        .map(|&token| {
            if let Some(ts) = bpe.token_to_timestamp(token) {
                // Whisper timestamp tokens are seconds; the stitcher works in 20 ms frames.
                frame = (ts * 50.0).max(0.0) as usize;
            }
            frame
        })
        .collect();
    Ok((text, tokens, frames))
}