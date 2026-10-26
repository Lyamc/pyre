use crate::token::{Gpt2Tokenizer, Language, SpecialToken};

use crate::transcribe::DecodeTask;

pub fn is_whisper_english_only(model_name: &str) -> bool {
    model_name.ends_with("_en") || model_name.contains(".en")
}

pub fn build_initial_tokens(
    bpe: &Gpt2Tokenizer,
    lang: Language,
    decode_task: DecodeTask,
    model_name: &str,
    include_timestamps: bool,
) -> Vec<usize> {
    let start_token = bpe
        .special_token(SpecialToken::StartofTranscript)
        .or_else(|| bpe.token_to_id("<|audio_start|>"))
        .or_else(|| bpe.token_to_id("<|im_start|>"))
        .unwrap_or(0);

    let task_token = match decode_task {
        DecodeTask::Translate => bpe
            .special_token(SpecialToken::Translate)
            .or_else(|| bpe.token_to_id("<|translate|>")),
        DecodeTask::Transcribe => bpe
            .special_token(SpecialToken::Transcribe)
            .or_else(|| bpe.token_to_id("<|transcribe|>"))
            .or_else(|| bpe.token_to_id("<|transcription|>")),
    };

    let lang_token = match decode_task {
        DecodeTask::Translate => bpe.special_token(SpecialToken::Language(Language::English)),
        DecodeTask::Transcribe => bpe.special_token(SpecialToken::Language(lang)),
    };

    let mut tokens = vec![start_token];

    if !is_whisper_english_only(model_name) {
        lang_token.into_iter().chain(task_token).for_each(|t| tokens.push(t));
    }

    if include_timestamps {
        bpe.special_token(SpecialToken::Timestamp(0.0))
            .into_iter()
            .for_each(|t| tokens.push(t));
    } else {
        bpe.special_token(SpecialToken::NoTimeStamps)
            .into_iter()
            .for_each(|t| tokens.push(t));
    }

    tokens
}

pub fn end_token(bpe: &Gpt2Tokenizer) -> usize {
    bpe.special_token(SpecialToken::EndofText)
        .or_else(|| bpe.token_to_id("<|endoftext|>"))
        .or_else(|| bpe.token_to_id("<|im_end|>"))
        .unwrap_or(50256)
}