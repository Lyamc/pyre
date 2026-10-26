use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::model::qwen3::{build_asr_prompt, flatten_prompt, load_qwen3_runtime};
use pyre_whisper_engine::model::Model;

#[test]
fn greedy_with_hf_audio_features_matches_hf_tokens() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let hf_audio = root.join("temp_dump_qwen3-asr-0.6b/hf_audio_features.bin");
    let meta_path = root.join("temp_dump_qwen3-asr-0.6b/hf_refs.json");
    let weights = root.join("models/qwen3-asr-0.6b/model.safetensors");
    if !hf_audio.exists() || !meta_path.exists() || !weights.exists() {
        eprintln!("skip: run scripts/export_qwen3_hf_refs.py first");
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).expect("meta")).expect("json");
    let hf_ids: Vec<usize> = meta["new_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let hf_tokens = meta["audio_tokens"].as_u64().unwrap() as usize;

    let bytes = std::fs::read(&hf_audio).expect("audio bin");
    let hf_vec: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let device = WgpuDevice::DefaultDevice;
    let (bpe, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, false).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3");
    };

    let runtime = load_qwen3_runtime("qwen3-asr-0.6b");
    let prompt = build_asr_prompt(&runtime, 1100, "", None);
    let prefix = flatten_prompt(&prompt, &bpe);

    let audio = Tensor::<Wgpu, 1>::from_floats(hf_vec.as_slice(), &device)
        .reshape([1, hf_tokens, 1024]);
    let prefix_ids = prefix.clone();
    let hidden = qwen.thinker.merge_audio_embeds(
        &prefix_ids,
        audio.clone(),
        runtime.audio_pad_token_id,
        &device,
    );
    let logits = qwen.thinker.forward_logits(hidden);
    let [_, seq, vocab] = logits.dims();
    let first = logits
        .slice([0..1, seq - 1..seq, 0..vocab])
        .argmax(2)
        .into_data()
        .to_vec::<i32>()
        .unwrap()[0] as usize;
    eprintln!("prefix_last_logits_argmax={first} (hf expects 11528)");

    let generated = qwen.thinker.generate_greedy(
        &prefix,
        audio,
        runtime.audio_pad_token_id,
        &runtime.eos_token_ids,
        32,
        &device,
    );

    eprintln!("hf_prefix={:?}", &hf_ids[..8.min(hf_ids.len())]);
    eprintln!("rust_prefix={:?}", &generated[..8.min(generated.len())]);

    let n = hf_ids.len().min(generated.len()).min(12);
    for i in 0..n {
        assert_eq!(
            generated[i], hf_ids[i],
            "thinker token mismatch at {i}: rust={} hf={}",
            generated[i], hf_ids[i]
        );
    }
}