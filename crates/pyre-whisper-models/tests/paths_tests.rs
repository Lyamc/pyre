use std::fs;
use tempfile::tempdir;

use pyre_whisper_models::paths::{has_burn_bundle, model_dir};

#[test]
fn burn_bundle_detection_requires_weights_config_and_tokenizer() {
    let dir = tempdir().unwrap();
    let model = "tiny_en";
    let path = dir.path().join(model);
    fs::create_dir_all(&path).unwrap();

    assert!(!has_burn_bundle(&path, model));

    fs::write(path.join("tokenizer.json"), "{}").unwrap();
    fs::write(path.join("config.cfg"), "").unwrap();
    fs::write(path.join("model.mpk"), b"stub").unwrap();

    assert!(has_burn_bundle(&path, model));
}

#[test]
fn model_dir_joins_under_models_root() {
    let dir = model_dir("tiny_en");
    assert!(dir.ends_with("models/tiny_en") || dir.ends_with("models\\tiny_en"));
}

#[test]
fn qwen3_bundle_accepts_tokenizer_json_without_vocab_merges() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("qwen3-asr-1.7b");
    fs::create_dir_all(&path).unwrap();

    fs::write(path.join("config.json"), "{}").unwrap();
    fs::write(path.join("model.safetensors"), b"stub").unwrap();
    fs::write(path.join("qwen3_runtime.json"), "{}").unwrap();
    assert!(!has_burn_bundle(&path, "qwen3-asr-1.7b"));

    fs::write(path.join("tokenizer.json"), "{}").unwrap();
    assert!(has_burn_bundle(&path, "qwen3-asr-1.7b"));
}