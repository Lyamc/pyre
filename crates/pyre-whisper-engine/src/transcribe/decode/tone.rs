
use crate::model::tone::weights::load_tone_runtime;
use crate::model::tone::{greedy_ctc_text, tone_log_mel, TONE};
use crate::token::Gpt2Tokenizer;

pub fn decode_tone(
    model: &TONE,
    _bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    model_name: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    let runtime = load_tone_runtime(model_name);
    let device = model.device();
    let mel = tone_log_mel(waveform, sample_rate, &device);
    if verbose {
        let [_, mels, frames] = mel.dims();
        println!(
            "DEBUG T-one: {} samples @ {}Hz → log-mel [{mels}, {frames}] (target {}Hz)",
            waveform.len(),
            sample_rate,
            runtime.sample_rate
        );
    }
    let logits = model.logits(mel);
    let (text, tokens) = greedy_ctc_text(logits, &runtime.vocab, runtime.blank_id);
    if verbose {
        println!(
            "DEBUG T-one: {} CTC tokens, text preview: {:?}",
            tokens.len(),
            text.chars().take(80).collect::<String>()
        );
    }
    (text, tokens)
}
