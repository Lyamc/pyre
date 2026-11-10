use std::path::Path;

use burn::tensor::Tensor;
use npy::NpyData;
use pyre_whisper_engine::model::load_model;

fn load_f32_npy(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    let npy = NpyData::<f32>::from_bytes(&bytes).ok()?;
    Some(npy.to_vec())
}



#[test]
fn parakeet_hf_mel_encode_parity() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);

    let mpk = root.join("models/parakeet-tdt-0.6b-v3/model.bpk");
    let legacy = root.join("models/parakeet-tdt-0.6b-v3/model.mpk");
    let hf_mel = root.join("temp_dump_parakeet-tdt-0.6b-v3/hf_mel_bct.npy");
    let hf_enc = root.join("temp_dump_parakeet-tdt-0.6b-v3/hf_enc_pooler.npy");
    let hf_pre = root.join("temp_dump_parakeet-tdt-0.6b-v3/hf_enc_preproj.npy");
    if !(mpk.exists() || legacy.exists()) || !hf_mel.exists() || !hf_enc.exists() || !hf_pre.exists() {
        eprintln!("skip: missing parakeet mpk or HF reference npy");
        return;
    }

    let mel_vec = load_f32_npy(&hf_mel).expect("hf mel");
    let enc_vec = load_f32_npy(&hf_enc).expect("hf enc");
    let pre_vec = load_f32_npy(&hf_pre).expect("hf preproj");
    let mel_frames = 1101usize;
    let n_mels = 128usize;
    let enc_frames = enc_vec.len() / 640;

    let device = pyre_whisper_engine::cli::get_device(&Some("cpu".into()));
    let mel = Tensor::<1>::from_data(
        burn::tensor::TensorData::new(mel_vec.clone(), [mel_vec.len()]),
        &device,
    )
    .reshape([1, n_mels, mel_frames]);
    let (_, _, model) = load_model("parakeet-tdt-0.6b-v3", &device, false).expect("load parakeet");
    let Model::Parakeet(parakeet) = model else {
        panic!("expected parakeet model");
    };

    let dec_norm = parakeet
        .decoder
        .as_ref()
        .map(|d| {
            d.embedding
                .weight
                .val()
                .clone()
                .powf_scalar(2.0)
                .sum()
                .into_scalar::<f32>()
                .sqrt()
        })
        .unwrap_or(0.0);
    eprintln!("decoder embedding L2 norm: {dec_norm:.4}");

    let hidden = parakeet.encode_hidden(mel.clone());
    let enc = parakeet.encode(mel);
    let hidden0 = hidden
        .clone()
        .slice([0..1, 0..1, 0..1024])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let hf_hidden0: Vec<f32> = pre_vec[..1024].to_vec();
    let hidden_rmse = rmse(&hidden0, &hf_hidden0);
    eprintln!(
        "preproj frame0: rust_norm={:.4} hf_norm={:.4} rmse={hidden_rmse:.6}",
        l2(&hidden0),
        l2(&hf_hidden0)
    );

    let frame0 = enc.clone().slice([0..1, 0..1, 0..640]).into_data().to_vec::<f32>().unwrap();
    let hf_frame0: Vec<f32> = enc_vec[..640].to_vec();

    let rmse = rmse(&frame0, &hf_frame0);
    let max_diff = max_abs(&frame0, &hf_frame0);
    let rust_norm = l2(&frame0);
    let hf_norm = l2(&hf_frame0);

    eprintln!(
        "encode frame0: rust_norm={rust_norm:.4} hf_norm={hf_norm:.4} rmse={rmse:.6} max_diff={max_diff:.6}"
    );
    eprintln!("enc shape {:?} hf enc frames {enc_frames}", enc.dims());

    assert!(dec_norm > 1.0, "decoder weights look uninitialized: {dec_norm}");
    assert!(
        hidden_rmse < 0.05,
        "encoder hidden mismatch vs HF (rmse={hidden_rmse})"
    );
    assert!(
        rmse < 0.2,
        "encoder pooler mismatch vs HF (rmse={rmse}, max_diff={max_diff})"
    );
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let mut sq = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sq += d * d;
    }
    (sq / a.len() as f32).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

use pyre_whisper_engine::model::Model;
use pyre_whisper_engine::transcribe::{drop_left_context_tokens, experimental_gate};

/// Runs in CI with no checkpoint and no HF dump. The neural parity test above
/// still skips until those files exist; this one does not.
#[test]
fn public_domain_reference_stitch_and_gates() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let audio = root.join("tests/fixtures/jfk.flac");
    let reference = std::fs::read_to_string(root.join("tests/fixtures/jfk.txt")).expect("jfk.txt");
    let reference = reference.trim();
    assert!(audio.metadata().expect("jfk.flac").len() > 10_000, "bundled clip is empty");
    assert!(reference.split_whitespace().count() > 10);

    assert_eq!(word_error_rate(reference, reference), 0.0);
    let shortened = "and so my fellow americans ask not what your country can do for you";
    let score = word_error_rate(reference, shortened);
    assert!(score > 0.15 && score < 0.5, "wer {score}");

    // First window keeps the opening tokens. A later window drops the left context.
    assert_eq!(drop_left_context_tokens(&[7, 8, 9], &[0, 3, 40], 0), vec![7, 8, 9]);
    assert_eq!(drop_left_context_tokens(&[7, 8, 9], &[0, 3, 40], 8), vec![9]);

    let refused = experimental_gate("parakeet-tdt-0.6b-v3", 16_000, 16_000).unwrap_err();
    assert!(refused.contains("experimental"));
    assert!(experimental_gate("nemotron-speech-streaming-en-0.6b", 16_000, 16_000).is_err());
    assert!(experimental_gate("parakeet-tdt-0.6b-v2", 16_000 * 20, 16_000).is_ok());
    assert!(experimental_gate("parakeet-tdt-0.6b-v2", 16_000 * 120, 16_000).is_err());
    assert!(experimental_gate("base", 16_000 * 400, 16_000).is_ok());
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<String> = reference.split_whitespace().map(|w| w.to_ascii_lowercase()).collect();
    let h: Vec<String> = hypothesis.split_whitespace().map(|w| w.to_ascii_lowercase()).collect();
    let n = r.len();
    let m = h.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n][m] as f64 / n.max(1) as f64
}