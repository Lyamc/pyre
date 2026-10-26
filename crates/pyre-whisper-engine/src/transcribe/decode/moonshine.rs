use burn::tensor::Tensor;

use crate::model::moonshine::weights::load_moonshine_runtime;
use crate::model::moonshine::{decode_moonshine_tokens, MoonshineASR};
use crate::token::Gpt2Tokenizer;

pub fn decode_moonshine(
    model: &MoonshineASR,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    model_name: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    let runtime = load_moonshine_runtime(model_name);
    let target_rate = runtime.sample_rate;
    let waveform = if sample_rate == target_rate {
        waveform.to_vec()
    } else {
        pyre_whisper_audio::resample_mono(waveform, sample_rate, target_rate)
            .unwrap_or_else(|_| waveform.to_vec())
    };

    let device = model.device();
    let audio = Tensor::<1>::from_floats(waveform.as_slice(), &device).reshape([1, waveform.len()]);
    let hidden = model.encode(audio);
    if verbose {
        let [_, frames, dim] = hidden.dims();
        println!(
            "DEBUG Moonshine: {} samples @ {}Hz → encoder [{frames}, {dim}]",
            waveform.len(),
            target_rate
        );
    }
    let tokens = model.generate_greedy(hidden, verbose);
    let text = decode_moonshine_tokens(bpe, &tokens, runtime.eos_token_id, runtime.decoder_start_token_id);
    (text, tokens)
}
