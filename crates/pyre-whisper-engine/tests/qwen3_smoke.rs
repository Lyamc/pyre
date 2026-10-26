use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::model::qwen3::parse_qwen3_asr_output_parts;
use pyre_whisper_engine::model::Model;
use pyre_whisper_engine::audio::prep_audio;
use pyre_whisper_engine::transcribe::decode::decode_qwen3;

#[test]
fn qwen3_load_and_decode_smoke() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let model_dir = root.join("models/qwen3-asr-0.6b");
    let weights = model_dir.join("model.safetensors");
    let runtime = model_dir.join("qwen3_runtime.json");
    if !weights.exists() || !runtime.exists() {
        eprintln!("skip: download qwen3-asr-0.6b first");
        return;
    }

    let device = WgpuDevice::DefaultDevice;
    let (bpe, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, true).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3 model");
    };

    // 1 second of silence at 16 kHz — should not crash the full pipeline.
    let samples = vec![0.0f32; 16_000];
    let w = Tensor::<Wgpu, 1>::from_floats(samples.as_slice(), &device).unsqueeze();
    let mel = prep_audio(w, 16_000.0, qwen.encoder_mel_size(), false, "qwen3-asr-0.6b");
    let (text, _tokens) =
        decode_qwen3(&qwen, &bpe, mel, "qwen3-asr-0.6b", None, 64, true);
    eprintln!("qwen3 smoke decode: {text:?}");

    let (lang, body) = parse_qwen3_asr_output_parts(&format!("language English<asr_text>{text}"));
    assert_eq!(lang, "English");
    assert_eq!(body, text);
}