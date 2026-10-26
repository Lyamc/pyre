use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::model::qwen3::{build_asr_prompt, flatten_prompt, load_qwen3_runtime};
use pyre_whisper_engine::model::Model;

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / a.len() as f32;
    mse.sqrt()
}

#[test]
fn merged_embeds_and_hidden_match_hf() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let hf_merged = root.join("temp_dump_qwen3-asr-0.6b/hf_merged_embeds.bin");
    let hf_hidden = root.join("temp_dump_qwen3-asr-0.6b/hf_last_hidden.bin");
    let hf_audio = root.join("temp_dump_qwen3-asr-0.6b/hf_audio_features.bin");
    let weights = root.join("models/qwen3-asr-0.6b/model.safetensors");
    if !hf_merged.exists() || !hf_hidden.exists() || !hf_audio.exists() || !weights.exists() {
        eprintln!("skip: export HF refs first (hf_merged_embeds.bin, hf_last_hidden.bin)");
        return;
    }

    let merged_bytes = std::fs::read(&hf_merged).expect("merged bin");
    let hidden_bytes = std::fs::read(&hf_hidden).expect("hidden bin");
    let audio_bytes = std::fs::read(&hf_audio).expect("audio bin");

    let hf_merged: Vec<f32> = merged_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let hf_hidden: Vec<f32> = hidden_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let hf_audio: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let seq = hf_merged.len() / 1024;
    let audio_tokens = hf_audio.len() / 1024;
    assert_eq!(seq, 158);
    assert_eq!(audio_tokens, 143);

    let device = WgpuDevice::DefaultDevice;
    let (bpe, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, false).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3");
    };

    let runtime = load_qwen3_runtime("qwen3-asr-0.6b");
    let prompt = build_asr_prompt(&runtime, 1100, "", None);
    let prefix = flatten_prompt(&prompt, &bpe);

    let audio = Tensor::<Wgpu, 1>::from_floats(hf_audio.as_slice(), &device)
        .reshape([1, audio_tokens, 1024]);
    let rust_merged = qwen.thinker.merge_audio_embeds(
        &prefix,
        audio,
        runtime.audio_pad_token_id,
        &device,
    );
    let rust_vec = rust_merged.clone().into_data().to_vec::<f32>().unwrap();

    let merged_rmse = rmse(&rust_vec, &hf_merged);
    eprintln!("merged_embeds_rmse={merged_rmse:.6}");

    let rust_hidden = qwen
        .thinker
        .forward_hidden(rust_merged)
        .slice([0..1, seq - 1..seq, 0..1024])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let hidden_rmse = rmse(&rust_hidden, &hf_hidden);
    eprintln!("last_hidden_rmse={hidden_rmse:.6}");

    assert!(
        merged_rmse < 0.01,
        "merged embeds diverge from HF: rmse={merged_rmse}"
    );
    assert!(
        hidden_rmse < 0.5,
        "thinker hidden diverges from HF: rmse={hidden_rmse}"
    );
}