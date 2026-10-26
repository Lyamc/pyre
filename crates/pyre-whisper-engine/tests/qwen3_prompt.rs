use std::path::Path;

use pyre_whisper_engine::model::qwen3::{
    build_asr_prompt, flatten_prompt, load_qwen3_runtime,
};
use pyre_whisper_engine::token::Gpt2Tokenizer;

#[test]
fn prompt_special_token_ids_match_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let model_dir = root.join("models/qwen3-asr-0.6b");
    if !model_dir.join("vocab.json").exists() {
        eprintln!("skip: download qwen3-asr-0.6b first");
        return;
    }

    let runtime = load_qwen3_runtime("qwen3-asr-0.6b");
    let bpe = Gpt2Tokenizer::new("qwen3-asr-0.6b").expect("tokenizer");

    let prompt = build_asr_prompt(&runtime, 100, "", None);
    let ids = flatten_prompt(&prompt, &bpe);

    assert_eq!(prompt.audio_pad_count, 13);
    assert!(
        ids.iter().filter(|&&id| id == runtime.im_start_token_id).count() >= 3,
        "expected system/user/assistant im_start tokens, got {ids:?}"
    );

    let pad_count = ids
        .iter()
        .filter(|&&id| id == runtime.audio_pad_token_id)
        .count();
    assert_eq!(pad_count, 13, "audio pad token count mismatch");

    assert!(ids.contains(&runtime.audio_start_token_id));
    assert!(ids.contains(&runtime.audio_end_token_id));
    assert!(!ids.contains(&runtime.asr_text_token_id));

    let forced = build_asr_prompt(&runtime, 10, "", Some("English"));
    let forced_ids = flatten_prompt(&forced, &bpe);
    assert!(forced_ids.contains(&runtime.asr_text_token_id));
    assert_eq!(forced_ids.last().copied(), Some(runtime.asr_text_token_id));
}

#[test]
fn prompt_ids_match_hf_reference_for_jfk() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let hf_path = root.join("temp_dump_qwen3-asr-0.6b/hf_input_ids.json");
    let model_dir = root.join("models/qwen3-asr-0.6b");
    if !hf_path.exists() || !model_dir.join("vocab.json").exists() {
        eprintln!("skip: run scripts/export_qwen3_hf_refs.py first");
        return;
    }

    let hf: Vec<usize> = serde_json::from_str(&std::fs::read_to_string(&hf_path).expect("hf ids"))
        .expect("parse hf ids");
    let runtime = load_qwen3_runtime("qwen3-asr-0.6b");
    let bpe = Gpt2Tokenizer::new("qwen3-asr-0.6b").expect("tokenizer");

    let prompt = build_asr_prompt(&runtime, 1100, "", None);
    let ids = flatten_prompt(&prompt, &bpe);

    eprintln!("hf_len={} rust_len={}", hf.len(), ids.len());
    if ids != hf {
        for (i, (a, b)) in ids.iter().zip(hf.iter()).enumerate() {
            if a != b {
                eprintln!("first diff at {i}: rust={a} hf={b}");
                break;
            }
        }
    }
    assert_eq!(ids, hf, "prompt token ids must match HF processor");
}