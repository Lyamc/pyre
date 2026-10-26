use tokenizers::AddedToken;
use crate::token::language::{Language, LANGUAGES};

pub enum SpecialToken {
    EndofText,
    StartofTranscript,
    Translate,
    Transcribe,
    StartofLM,
    StartofPrev,
    NoSpeech,
    NoTimeStamps,
    Language(Language),
    Timestamp(f64),
}

impl ToString for SpecialToken {
    fn to_string(&self) -> String {
        match self {
            SpecialToken::EndofText => "<|endoftext|>".into(),
            SpecialToken::StartofTranscript => "<|startoftranscript|>".into(),
            SpecialToken::Translate => "<|translate|>".into(),
            SpecialToken::Transcribe => "<|transcribe|>".into(),
            SpecialToken::StartofLM => "<|startoflm|>".into(),
            SpecialToken::StartofPrev => "<|startofprev|>".into(),
            SpecialToken::NoSpeech => "<|nospeech|>".into(),
            SpecialToken::NoTimeStamps => "<|notimestamps|>".into(),
            SpecialToken::Language(lang) => format!("<|{}|>", lang.as_str()),
            SpecialToken::Timestamp(val) => format!("<|{:.2}|>", val),
        }
    }
}

pub fn construct_special_tokens() -> Vec<AddedToken> {
    const SPEC1: [&str; 2] = ["<|endoftext|>", "<|startoftranscript|>"];

    let lang_keys = LANGUAGES.iter().map(|lang| format!("<|{}|>", lang));

    const SPEC2: [&str; 6] = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ];

    let range_keys = (0..1501)
        .into_iter()
        .map(|i| i as f64 * 0.02)
        .map(|f| format!("<|{:.2}|>", f));

    SPEC1
        .into_iter()
        .map(String::from)
        .chain(lang_keys.into_iter())
        .chain(SPEC2.into_iter().map(String::from))
        .chain(range_keys.into_iter())
        .map(|tok| AddedToken::from(tok, true))
        .collect()
}
