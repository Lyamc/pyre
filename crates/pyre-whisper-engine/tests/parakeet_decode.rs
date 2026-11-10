use std::path::Path;

use pyre_whisper_audio::decode_bytes_to_mono_pcm;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::model::Model;
use pyre_whisper_engine::transcribe::decode::is_garbage_parakeet_text;
use pyre_whisper_engine::transcribe::{waveform_to_text, DecodeTask};

#[test]
fn parakeet_jfk_decode_is_readable_not_commas() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let wav_path = root.join("samples/jfk.wav");
    let mpk = root.join("models/parakeet-tdt-0.6b-v3/model.bpk");
    let legacy = root.join("models/parakeet-tdt-0.6b-v3/model.mpk");
    if !wav_path.exists() || !(mpk.exists() || legacy.exists()) {
        eprintln!("skip: missing jfk.wav or parakeet model");
        return;
    }

    let bytes = std::fs::read(&wav_path).unwrap();
    let wf = decode_bytes_to_mono_pcm(&bytes, "wav").unwrap();
    let device = pyre_whisper_engine::cli::get_device(&Some("cpu".into()));
    let (bpe, _, model) =
        load_model("parakeet-tdt-0.6b-v3", &device, false).expect("load parakeet");
    let Model::Parakeet(_) = model else {
        panic!("expected parakeet");
    };

    let (text, _) = waveform_to_text(
        &model,
        "parakeet-tdt-0.6b-v3",
        &bpe,
        pyre_whisper_engine::token::Language::English,
        "en",
        wf.samples,
        None,
        wf.sample_rate,
        false,
        false,
        true,
        true,
        true,
        "test".to_string(),
        1,
        448,
        0,
        None,
        Some(DecodeTask::Transcribe),
    )
    .expect("transcribe");

    assert!(!is_garbage_parakeet_text(&text), "garbage output: {text:?}");
    let lower = text.to_lowercase();
    assert!(
        lower.contains("america") || lower.contains("country") || lower.contains("fellow"),
        "unexpected transcript: {text}"
    );
}