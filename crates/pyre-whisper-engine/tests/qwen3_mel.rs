use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;
use pyre_whisper_audio::decode_to_mono_pcm;
use pyre_whisper_engine::audio::prep_audio;

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
fn jfk_mel_values_match_hf_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let hf_mel = root.join("temp_dump_qwen3-asr-0.6b/hf_mel.bin");
    let path = root.join("samples/jfk.wav");
    if !hf_mel.exists() || !path.exists() {
        eprintln!("skip: export hf_mel.bin first");
        return;
    }
    let _ = std::env::set_current_dir(&root);

    let bytes = std::fs::read(&hf_mel).expect("hf mel");
    let hf: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let wf = decode_to_mono_pcm(&path).expect("decode");
    let device = WgpuDevice::DefaultDevice;
    let w = Tensor::<Wgpu, 1>::from_floats(wf.samples.as_slice(), &device).unsqueeze();
    let mel = prep_audio(w, wf.sample_rate as f64, 128, false, "qwen3-asr-0.6b");
    let rust = mel.into_data().to_vec::<f32>().unwrap();
    let mel_rmse = rmse(&rust, &hf);
    eprintln!("mel_rmse={mel_rmse:.6}");
    assert!(mel_rmse < 0.01, "mel diverges from HF: rmse={mel_rmse}");
}

#[test]
fn jfk_mel_frame_count_matches_hf_whisper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("samples/jfk.wav");
    if !path.exists() {
        eprintln!("skip: jfk.wav missing");
        return;
    }
    let _ = std::env::set_current_dir(&root);

    let wf = decode_to_mono_pcm(&path).expect("decode");
    let device = WgpuDevice::DefaultDevice;
    let w = Tensor::<Wgpu, 1>::from_floats(wf.samples.as_slice(), &device).unsqueeze();
    let mel = prep_audio(w, wf.sample_rate as f64, 128, false, "qwen3-asr-0.6b");
    let frames = mel.dims()[2];
    assert_eq!(frames, 1100, "HF Qwen3 WhisperFeatureExtractor uses 1100 frames for JFK");
}