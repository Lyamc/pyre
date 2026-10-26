use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_audio::decode_to_mono_pcm;
use pyre_whisper_engine::audio::prep_audio;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::model::qwen3::{build_asr_prompt, flatten_prompt, load_qwen3_runtime};
use pyre_whisper_engine::model::Model;
use pyre_whisper_engine::transcribe::decode::decode_qwen3;

#[test]
fn greedy_tokens_match_hf_prefix() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let meta_path = root.join("temp_dump_qwen3-asr-0.6b/hf_refs.json");
    let weights = root.join("models/qwen3-asr-0.6b/model.safetensors");
    let jfk = root.join("samples/jfk.wav");
    if !meta_path.exists() || !weights.exists() || !jfk.exists() {
        eprintln!("skip: run scripts/export_qwen3_hf_refs.py first");
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).expect("read meta")).expect("json");
    let hf_ids: Vec<usize> = meta["new_token_ids"]
        .as_array()
        .expect("new_token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as usize)
        .collect();

    let wf = decode_to_mono_pcm(&jfk).expect("decode");
    let device = WgpuDevice::DefaultDevice;
    let w = Tensor::<Wgpu, 1>::from_floats(wf.samples.as_slice(), &device).unsqueeze();
    let mel = prep_audio(w, wf.sample_rate as f64, 128, false, "qwen3-asr-0.6b");

    let (bpe, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, false).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3");
    };

    let (_text, tokens) = decode_qwen3(
        &qwen,
        &bpe,
        mel.clone(),
        "qwen3-asr-0.6b",
        None,
        32,
        true,
    );

    let runtime = load_qwen3_runtime("qwen3-asr-0.6b");
    let prompt = build_asr_prompt(&runtime, mel.dims()[2], "", None);
    let prefix = flatten_prompt(&prompt, &bpe);
    eprintln!("prefix_len={} generated_len={}", prefix.len(), tokens.len());
    eprintln!("hf_prefix={:?}", &hf_ids[..hf_ids.len().min(8)]);
    eprintln!("rust_prefix={:?}", &tokens[..tokens.len().min(8)]);

    let n = hf_ids.len().min(tokens.len()).min(12);
    for i in 0..n {
        assert_eq!(
            tokens[i], hf_ids[i],
            "token mismatch at step {i}: rust={} hf={}",
            tokens[i], hf_ids[i]
        );
    }
}