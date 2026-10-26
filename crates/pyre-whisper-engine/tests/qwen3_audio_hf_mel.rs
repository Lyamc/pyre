use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_engine::model::load_model;
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
fn audio_encoder_with_hf_mel_matches_hf_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let hf_mel = root.join("temp_dump_qwen3-asr-0.6b/hf_mel.bin");
    let hf_audio = root.join("temp_dump_qwen3-asr-0.6b/hf_audio_features.bin");
    let weights = root.join("models/qwen3-asr-0.6b/model.safetensors");
    if !hf_mel.exists() || !hf_audio.exists() || !weights.exists() {
        eprintln!("skip: export hf_mel.bin and hf_audio_features.bin first");
        return;
    }

    let mel_bytes = std::fs::read(&hf_mel).expect("mel");
    let mel_vec: Vec<f32> = mel_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let audio_bytes = std::fs::read(&hf_audio).expect("audio");
    let hf_audio_vec: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let device = WgpuDevice::DefaultDevice;
    let (_, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, false).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3");
    };

    let mel = Tensor::<Wgpu, 1>::from_floats(mel_vec.as_slice(), &device).reshape([1, 128, 1100]);
    let enc = qwen.encode_audio(mel);
    let rust_vec = enc.into_data().to_vec::<f32>().unwrap();
    let enc_rmse = rmse(&rust_vec, &hf_audio_vec);
    eprintln!("audio_encoder_with_hf_mel_rmse={enc_rmse:.6}");
    assert!(
        enc_rmse < 0.01,
        "audio tower diverges even with HF mel: rmse={enc_rmse}"
    );
}