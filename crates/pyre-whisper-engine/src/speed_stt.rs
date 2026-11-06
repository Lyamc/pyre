//! Measured cost of the Whisper tiny decode path on this machine.
//! Weights are random; tensor shapes match tiny (decoder) so the times are real
//! compute, not a fabricated transcript runtime.
use std::time::Instant;

use burn::tensor::Distribution;
use burn::tensor::{Int, Tensor, TensorData};

use crate::audio::stft::{gather_frames, gather_frames_collect};
use crate::model::{AudioEncoderConfig, TextDecoder, TextDecoderConfig};
use crate::transcribe::mels_to_text::whisper_encoder_mel_frames;

fn cpu() -> burn::tensor::Device {
    crate::cli::get_device(&Some("cpu".into()))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn ids(tokens: &[i64], device: &burn::tensor::Device) -> Tensor<2, Int> {
    let n = tokens.len();
    Tensor::<2, Int>::from_ints(TensorData::new(tokens.to_vec(), [1, n]), device)
}

fn eager_last(dec: &TextDecoder, tokens: &[i64], xa: Tensor<3>) -> Vec<f32> {
    let device = xa.device();
    let logits = dec.forward(ids(tokens, &device), xa);
    let [_, seq, vocab] = logits.dims();
    logits
        .slice([0..1, seq - 1..seq, 0..vocab])
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

#[test]
fn cached_decoder_matches_eager_last_token_logits() {
    let device = cpu();
    let dec = TextDecoderConfig::new(64, 16, 32, 4, 2).init(&device);
    let xa = Tensor::<3>::random([1, 6, 32], Distribution::Normal(0.0, 0.4), &device);
    let tokens = [3i64, 7, 1, 9, 4, 2];
    let mut kv = None;
    for t in 0..tokens.len() {
        let step = [tokens[t]];
        let (logits, next) = dec.forward_cached(ids(&step, &device), xa.clone(), kv);
        kv = Some(next);
        let got = logits.into_data().to_vec::<f32>().unwrap();
        let expect = eager_last(&dec, &tokens[..=t], xa.clone());
        let diff = max_abs(&got, &expect);
        assert!(
            diff < 1e-3,
            "step {t} max abs logit diff {diff} (cached vs full forward)"
        );
    }
}

#[test]
fn cached_decoder_matches_eager_for_a_batch_of_beams() {
    let device = cpu();
    let dec = TextDecoderConfig::new(40, 12, 32, 4, 2).init(&device);
    let xa = Tensor::<3>::random([1, 5, 32], Distribution::Normal(0.0, 0.3), &device);
    // Prefill two identical prompts as a batch of 1, then step two different tokens.
    let prompt = [1i64, 4, 6];
    let (logits, kv) = dec.forward_cached(ids(&prompt, &device), xa.clone(), None);
    let _ = logits.into_data();
    let parts = kv.split_batch();
    assert_eq!(parts.len(), 1);
    let past = crate::model::decoder::DecoderKv::concat_batch(&[parts[0].clone(), parts[0].clone()]);
    let flat = vec![8i64, 3i64];
    let tokens = Tensor::<2, Int>::from_ints(TensorData::new(flat, [2, 1]), &device);
    let (batched, _) = dec.forward_cached(tokens, xa.clone(), Some(past));
    let got = batched.into_data().to_vec::<f32>().unwrap();
    let vocab = got.len() / 2;
    let left = eager_last(&dec, &[1, 4, 6, 8], xa.clone());
    let right = eager_last(&dec, &[1, 4, 6, 3], xa);
    assert!(max_abs(&got[..vocab], &left) < 1e-3, "beam0");
    assert!(max_abs(&got[vocab..], &right) < 1e-3, "beam1");
}

fn tiny_decoder(device: &burn::tensor::Device) -> TextDecoder {
    // openai/whisper-tiny decoder: vocab 51865, width 384, 6 heads, 4 layers.
    // Context is only long enough for this bench; the mask is not on the step hot path.
    TextDecoderConfig::new(51_865, 24, 384, 6, 4).init(device)
}

fn run_eager(dec: &TextDecoder, xa: &Tensor<3>, steps: usize) {
    let device = xa.device();
    let mut seq = vec![1i64, 2, 3, 4];
    for _ in 0..steps {
        let logits = dec.forward(ids(&seq, &device), xa.clone());
        let [_, s, v] = logits.dims();
        let _ = logits.slice([0..1, s - 1..s, 0..v]).into_data();
        seq.push(5);
    }
}

fn run_cached(dec: &TextDecoder, xa: &Tensor<3>, steps: usize) {
    let device = xa.device();
    let prompt = [1i64, 2, 3, 4];
    let (logits, kv) = dec.forward_cached(ids(&prompt, &device), xa.clone(), None);
    let _ = logits.into_data();
    let mut past = Some(kv);
    for _ in 0..steps {
        let (logits, next) = dec.forward_cached(ids(&[5], &device), xa.clone(), past);
        let _ = logits.into_data();
        past = Some(next);
    }
}

#[test]
fn tiny_cpu_cached_steps_are_faster_than_full_recompute() {
    let device = cpu();
    let dec = tiny_decoder(&device);
    // 1500 encoder positions: tiny's padded 30 s cross-attention width.
    let xa = Tensor::<3>::random([1, 1500, 384], Distribution::Normal(0.0, 0.2), &device);
    let steps = 4usize;
    run_eager(&dec, &xa, 1);
    run_cached(&dec, &xa, 1);
    let mut ratios = Vec::with_capacity(3);
    let mut eager_s = 0.0;
    let mut cached_s = 0.0;
    for _ in 0..3 {
        let eager = Instant::now();
        run_eager(&dec, &xa, steps);
        eager_s = eager.elapsed().as_secs_f64();
        let cached = Instant::now();
        run_cached(&dec, &xa, steps);
        cached_s = cached.elapsed().as_secs_f64();
        ratios.push(eager_s / cached_s.max(1e-9));
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ratio = ratios[1];
    let xa_short = Tensor::<3>::random([1, 100, 384], Distribution::Normal(0.0, 0.2), &device);
    run_cached(&dec, &xa_short, 1);
    let short = Instant::now();
    run_cached(&dec, &xa_short, steps);
    let short_s = short.elapsed().as_secs_f64();
    println!(
        "BENCH tiny-decoder-cpu eager_s={eager_s:.4} cached_s={cached_s:.4} median_ratio={ratio:.2} ratios={ratios:?} steps={steps} audio_ctx=1500 vocab=51865 cached_audio100_s={short_s:.4}"
    );
    assert!(
        ratio > 1.2,
        "cached decode median ratio {ratio:.2} (last pair eager {eager_s:.3}s cached {cached_s:.3}s) was not faster than full recompute"
    );
    assert!(
        short_s < cached_s,
        "decoder with 100 encoder positions {short_s:.3}s was not faster than 1500 {cached_s:.3}s"
    );
}

#[test]
fn tiny_encoder_cost_grows_with_padded_mel_frames() {
    let device = cpu();
    let enc = AudioEncoderConfig::new(80, 1500, 384, 6, 4).init(&device);
    let short = Tensor::<3>::random([1, 80, 200], Distribution::Normal(0.0, 0.2), &device);
    let long = Tensor::<3>::random([1, 80, 3000], Distribution::Normal(0.0, 0.2), &device);
    let _ = enc.forward(short.clone()).into_data();
    let _ = enc.forward(long.clone()).into_data();
    let t_short = Instant::now();
    let _ = enc.forward(short).into_data();
    let short_s = t_short.elapsed().as_secs_f64();
    let t_long = Instant::now();
    let _ = enc.forward(long).into_data();
    let long_s = t_long.elapsed().as_secs_f64();
    println!(
        "BENCH tiny-encoder-cpu mel200_s={short_s:.4} mel3000_s={long_s:.4} ratio={:.2}",
        long_s / short_s.max(1e-9)
    );
    assert!(long_s > short_s * 8.0, "padded 30s encoder pass should cost more than a short clip");
}

#[test]
fn short_clips_are_not_padded_to_thirty_seconds() {
    assert_eq!(whisper_encoder_mel_frames(200, 1500), 200);
    assert_eq!(whisper_encoder_mel_frames(3000, 1500), 3000);
    assert_eq!(whisper_encoder_mel_frames(4000, 1500), 3000);
}

fn wave(n: usize, device: &burn::tensor::Device) -> Tensor<2> {
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    Tensor::<2>::from_data(TensorData::new(data, [1, n]), device)
}

fn gather_secs(input: &Tensor<2>, n_fft: usize, hop: usize, frames: usize, host: bool) -> f64 {
    let start = Instant::now();
    let t = if host {
        gather_frames(input, n_fft, hop, frames)
    } else {
        gather_frames_collect(input, n_fft, hop, frames)
    };
    let _ = t.into_data();
    start.elapsed().as_secs_f64()
}

#[test]
fn stft_gather_matches_per_frame_collect_and_is_faster_on_multisecond_audio() {
    let device = cpu();
    let n_fft = 400usize;
    let hop = 160usize;
    let short_n = 8_000usize;
    let short_frames = (short_n - n_fft) / hop + 1;
    let short = wave(short_n, &device);
    let a = gather_frames(&short, n_fft, hop, short_frames).into_data().to_vec::<f32>().unwrap();
    let b = gather_frames_collect(&short, n_fft, hop, short_frames).into_data().to_vec::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    assert_eq!(max_abs(&a, &b), 0.0, "host gather must match the per-frame slice collection");

    let n = 4 * 16_000;
    let frames = (n - n_fft) / hop + 1;
    let input = wave(n, &device);
    let _ = gather_secs(&input, n_fft, hop, frames, true);
    let _ = gather_secs(&input, n_fft, hop, frames, false);
    let host_s = gather_secs(&input, n_fft, hop, frames, true);
    let collect_s = gather_secs(&input, n_fft, hop, frames, false);
    let n8 = 8 * 16_000;
    let frames8 = (n8 - n_fft) / hop + 1;
    let input8 = wave(n8, &device);
    let host8 = gather_secs(&input8, n_fft, hop, frames8, true);
    let collect8 = gather_secs(&input8, n_fft, hop, frames8, false);
    println!(
        "BENCH stft-gather 4s host_s={host_s:.4} collect_s={collect_s:.4} ratio={:.2} | 8s host_s={host8:.4} collect_s={collect8:.4} ratio={:.2} collect_scale={:.2}",
        collect_s / host_s.max(1e-9),
        collect8 / host8.max(1e-9),
        collect8 / collect_s.max(1e-9)
    );
    assert!(
        host_s * 1.5 < collect_s,
        "host gather {host_s:.3}s was not 1.5x faster than per-frame collect {collect_s:.3}s"
    );
}
