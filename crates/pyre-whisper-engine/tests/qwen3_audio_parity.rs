use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_audio::decode_to_mono_pcm;
use pyre_whisper_engine::audio::prep_audio;
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
fn qwen3_audio_encoder_matches_hf_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let hf_path = root.join("temp_dump_qwen3-asr-0.6b/hf_audio_features.bin");
    let meta_path = root.join("temp_dump_qwen3-asr-0.6b/hf_refs.json");
    let weights = root.join("models/qwen3-asr-0.6b/model.safetensors");
    let jfk = root.join("samples/jfk.wav");
    if !hf_path.exists() || !meta_path.exists() || !weights.exists() || !jfk.exists() {
        eprintln!("skip: run scripts/export_qwen3_hf_refs.py first");
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).expect("read meta")).expect("json");
    let hf_tokens = meta["audio_tokens"].as_u64().expect("audio_tokens") as usize;
    let bytes = std::fs::read(&hf_path).expect("read bin");
    let hf_vec: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(hf_vec.len(), hf_tokens * 1024);

    let wf = decode_to_mono_pcm(&jfk).expect("decode");
    let device = WgpuDevice::DefaultDevice;
    let w = Tensor::<Wgpu, 1>::from_floats(wf.samples.as_slice(), &device).unsqueeze();
    let mel = prep_audio(w, wf.sample_rate as f64, 128, false, "qwen3-asr-0.6b");

    let (_, _, model) =
        load_model::<Wgpu>("qwen3-asr-0.6b", &device, false).expect("load qwen3");
    let Model::Qwen3(qwen) = model else {
        panic!("expected qwen3");
    };

    let enc = qwen.encode_audio(mel);
    let [_, tokens, hidden] = enc.dims();
    assert_eq!(hidden, 1024);
    assert_eq!(tokens, hf_tokens, "audio token count");

    let rust_vec = enc.into_data().to_vec::<f32>().unwrap();
    let whole_rmse = rmse(&rust_vec, &hf_vec);
    let first_rmse = rmse(&rust_vec[..8], &hf_vec[..8]);
    eprintln!(
        "audio encoder rmse: whole={whole_rmse:.6} first8={first_rmse:.6} tokens={tokens}"
    );

    assert!(
        whole_rmse < 0.05,
        "audio encoder diverges from HF (rmse={whole_rmse:.6})"
    );
}