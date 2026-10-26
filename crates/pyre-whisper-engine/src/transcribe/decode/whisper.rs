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
) -> Result<(String, Vec<usize>), Box<dyn Error + Send + Sync>> {
    mels_to_text(
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
    )
}