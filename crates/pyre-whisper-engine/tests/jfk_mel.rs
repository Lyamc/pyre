use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::cast::ToElement;
use pyre_whisper_audio::decode_to_mono_pcm;
use pyre_whisper_engine::audio::prep_audio;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::transcribe::{waveform_to_text, DecodeTask};
use pyre_whisper_engine::token::Language;
use std::path::Path;

#[test]
fn jfk_mel_stats_and_transcription() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("samples/jfk.wav");
    if !path.exists() {
        eprintln!("skip: jfk.wav missing at {}", path.display());
        return;
    }
    let _ = std::env::set_current_dir(&root);
    if !Path::new("models/tiny_en/model.mpk").exists() {
        eprintln!("skip: tiny_en model missing");
        return;
    }

    let wf = decode_to_mono_pcm(&path).expect("decode");
    let device = WgpuDevice::DefaultDevice;
    let samples = wf.samples.clone();
    let tensor = burn::tensor::Tensor::<Wgpu, 1>::from_floats(
        samples.as_slice(),
        &device,
    )
    .unsqueeze();
    let mel = prep_audio(tensor, wf.sample_rate as f64, 80, false, "tiny_en");
    let min = mel.clone().min().into_scalar().to_f32();
    let max = mel.clone().max().into_scalar().to_f32();
    let mean = mel.clone().mean().into_scalar().to_f32();
    let dims = mel.dims();
    eprintln!("mel dims={dims:?} min={min} max={max} mean={mean}");

    let (bpe, _cfg, model) = load_model::<Wgpu>("tiny_en", &device, false).expect("load");
    let (text, _segs) = waveform_to_text(
        &model,
        "tiny_en",
        &bpe,
        Language::English,
        "en",
        samples,
        None,
        wf.sample_rate,
        false,
        true,
        false,
        false,
        false,
        "test".into(),
        5,
        448,
        200,
        None,
        Some(DecodeTask::Transcribe),
    )
    .expect("transcribe");
    eprintln!("text: {text:?}");
    let lower = text.to_lowercase();
    assert!(
        lower.contains("fellow") && lower.contains("country"),
        "expected JFK-like transcript, got: {text}"
    );
    assert!(
        !lower.contains("godfrey"),
        "trailing hallucination suffix: {text:?}"
    );
}