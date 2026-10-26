
use crate::model::qwen3::{
    build_asr_prompt, flatten_prompt, load_qwen3_runtime, parse_qwen3_asr_output_with_forced,
    Qwen3ASR,
};
use crate::token::Gpt2Tokenizer;

pub fn decode_qwen3(
    model: &Qwen3ASR,
    bpe: &Gpt2Tokenizer,
    mel: burn::tensor::Tensor<3>,
    model_name: &str,
    forced_language: Option<&str>,
    max_new_tokens: usize,
    verbose: bool,
) -> (String, Vec<usize>) {
    let runtime = load_qwen3_runtime(model_name);
    let mel_frames = mel.dims()[2];
    let prompt = build_asr_prompt(&runtime, mel_frames, "", forced_language);
    let audio_features = model.encode_audio(mel);
    let [_, tokens, hidden] = audio_features.dims();

    if verbose {
        println!(
            "DEBUG Qwen3: mel_frames={mel_frames}, audio_tokens={tokens}, hidden={hidden}, \
             pad_placeholders={}",
            prompt.audio_pad_count
        );
    }

    if runtime.inference_status != "ready" {
        let msg = format!(
            "Qwen3-ASR bundle loaded but Burn inference is not ready (status: {}). \
             Prepared {tokens} audio token slots from {mel_frames} mel frames.",
            runtime.inference_status
        );
        return (msg, Vec::new());
    }

    if tokens != prompt.audio_pad_count {
        return (
            format!(
                "Qwen3 audio token mismatch: encoder={tokens}, prompt pads={}",
                prompt.audio_pad_count
            ),
            Vec::new(),
        );
    }

    let device = model.device();
    let input_ids = flatten_prompt(&prompt, bpe);
    let encoded_pads = input_ids
        .iter()
        .filter(|&&id| id == runtime.audio_pad_token_id)
        .count();
    if encoded_pads != tokens {
        return (
            format!(
                "Qwen3 encoded pad mismatch: encoder={tokens}, encoded pads={encoded_pads} \
                 (pad_id={}). Tokenizer did not emit one token per <|audio_pad|>.",
                runtime.audio_pad_token_id
            ),
            Vec::new(),
        );
    }
    let generated = model.thinker.generate_greedy(
        &input_ids,
        audio_features,
        runtime.audio_pad_token_id,
        &runtime.eos_token_ids,
        max_new_tokens.max(1),
        &device,
    );

    let raw = bpe.decode(&generated, true).unwrap_or_default();
    let text = parse_qwen3_asr_output_with_forced(&raw, forced_language);
    (text, generated)
}