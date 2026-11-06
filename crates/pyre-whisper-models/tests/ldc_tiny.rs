//! Real-weight ablation of the Whisper tiny path on public LDC93S1.
//!
//! Requires `WHISBURN_MODELS_DIR` to contain a Hugging Face `tiny` checkout
//! (`config.json`, `tokenizer.json`, `model.safetensors`) and `LDC93S1_WAV`.
//! Skips when those files are absent so a normal `cargo test` does not download.

use std::path::PathBuf;
use std::time::Instant;

use pyre_whisper_engine::model::load_model;
use pyre_whisper_engine::token::Language;
use pyre_whisper_engine::transcribe::{
    waveform_to_text, DecodeTask, DecoderKind, MelTail, WhisperPass, Windowing,
};
use pyre_whisper_models::convert::convert_whisper_from_hf;
use pyre_whisper_models::download::DownloadOptions;

fn wav_samples(path: &std::path::Path) -> (Vec<f32>, usize) {
    let mut reader = hound::WavReader::open(path).expect("wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "LDC93S1 is mono");
    let samples = reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect::<Vec<_>>();
    (samples, spec.sample_rate as usize)
}

fn stretch(samples: &[f32], rate: usize, seconds: f64) -> Vec<f32> {
    let n = (seconds * rate as f64).round() as usize;
    let mut out = samples.to_vec();
    if out.len() < n {
        out.resize(n, 0.0);
    } else {
        out.truncate(n);
    }
    out
}

fn pass(tail: MelTail, decoder: DecoderKind, windows: Windowing) -> WhisperPass {
    WhisperPass { tail, decoder, windows }
}

struct Run {
    text: String,
    seg: String,
    seconds: f64,
    tokens_note: String,
}

fn transcribe(
    model: &pyre_whisper_engine::model::Model,
    bpe: &pyre_whisper_engine::token::Gpt2Tokenizer,
    samples: &[f32],
    rate: usize,
    beam: usize,
    how: WhisperPass,
) -> Run {
    let t = Instant::now();
    let (text, segs) = pyre_whisper_engine::transcribe::with_whisper_pass(how, || {
        waveform_to_text(
            model,
            "tiny",
            bpe,
            Language::English,
            "en",
            samples.to_vec(),
            None,
            rate,
            false,
            false,
            false,
            true,
            true,
            "cpu".into(),
            beam,
            448,
            200,
            None,
            Some(DecodeTask::Transcribe),
        )
        .expect("transcribe")
    });
    let seg = segs
        .iter()
        .map(|s| format!("{:.3}-{:.3}:{}", s.start, s.end, s.text.trim()))
        .collect::<Vec<_>>()
        .join(" | ");
    Run {
        text: text.trim().to_string(),
        seg,
        seconds: t.elapsed().as_secs_f64(),
        tokens_note: String::new(),
    }
}

#[test]
fn tiny_ldc93s1_padding_and_cache_ablation() {
    let wav = std::env::var("LDC93S1_WAV").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp/stt-fixtures/LDC93S1.wav"));
    let models = std::env::var("WHISBURN_MODELS_DIR").unwrap_or_else(|_| "/tmp/models".into());
    std::env::set_var("WHISBURN_MODELS_DIR", &models);
    let dir = PathBuf::from(&models).join("tiny");
    if !wav.is_file() || !dir.join("model.safetensors").is_file() {
        eprintln!("skip ldc ablation: missing {wav:?} or {}/model.safetensors", dir.display());
        return;
    }
    if !dir.join("model.bpk").is_file() {
        convert_whisper_from_hf(&dir, "tiny", &DownloadOptions { verbose: true, ..DownloadOptions::default() })
            .expect("convert tiny");
    }

    let device = pyre_whisper_engine::cli::get_device(&Some("cpu".into()));
    let (bpe, _cfg, model) = load_model("tiny", &device, false).expect("load tiny");
    let (full, rate) = wav_samples(&wav);
    let dur = full.len() as f64 / rate as f64;
    eprintln!("LDC93S1 samples={} rate={} dur={dur:.4}", full.len(), rate);

    let legacy = Windowing::Legacy;
    let specs = [
        ("padded_eager_beam5", pass(MelTail::ZeroPad, DecoderKind::Eager, legacy), 5usize, dur),
        ("nopad_eager_beam5", pass(MelTail::Truncate, DecoderKind::Eager, legacy), 5, dur),
        ("padded_cache_beam5", pass(MelTail::ZeroPad, DecoderKind::Cached, legacy), 5, dur),
        ("nopad_cache_beam5", pass(MelTail::Truncate, DecoderKind::Cached, legacy), 5, dur),
        ("padded_eager_beam1", pass(MelTail::ZeroPad, DecoderKind::Eager, legacy), 1, dur),
        ("padded_cache_beam1", pass(MelTail::ZeroPad, DecoderKind::Cached, legacy), 1, dur),
    ];

    let mut rows: Vec<(&str, Run)> = Vec::new();
    for (name, how, beam, seconds) in specs {
        let audio = stretch(&full, rate, seconds);
        let run = transcribe(&model, &bpe, &audio, rate, beam, how);
        eprintln!("ABLATE {name} {seconds:.3}s wall={:.3}s text={:?} seg={}", run.seconds, run.text, run.seg);
        rows.push((name, run));
    }

    let baseline = &rows[0].1;
    let nopad = &rows[1].1;
    let cached = &rows[2].1;
    let greedy = &rows[4].1;
    let greedy_cache = &rows[5].1;

    assert!(
        baseline.text.to_lowercase().contains("suit") && baseline.text.to_lowercase().contains("year"),
        "padded eager lost the LDC sentence: {:?}",
        baseline.text
    );
    assert!(
        !baseline.seg.contains("13.000") && !baseline.seg.starts_with("0.000-13"),
        "segment mapped to the 13s window instead of the media: {}",
        baseline.seg
    );
    let audio_end = full.len() as f64 / rate as f64;
    for part in baseline.seg.split(" | ") {
        if let Some(times) = part.split(':').next() {
            if let Some((a, b)) = times.split_once('-') {
                let end: f64 = b.parse().unwrap_or(99.0);
                let start: f64 = a.parse().unwrap_or(99.0);
                assert!(start <= audio_end + 0.02 && end <= audio_end + 0.02, "{part}");
            }
        }
    }
    assert_eq!(cached.text, baseline.text, "beam-5 cache text diverged from padded eager");
    assert_eq!(greedy_cache.text, greedy.text, "beam-1 cache text diverged from padded eager");
    eprintln!(
        "ABLATE nopad_vs_pad same_text={} nopad={:?}",
        nopad.text == baseline.text,
        nopad.text
    );

    for seconds in [0.8_f64, 2.0, 12.0, 13.2, 30.0, 35.0] {
        let audio = stretch(&full, rate, seconds);
        let eager = transcribe(&model, &bpe, &audio, rate, 5, pass(MelTail::ZeroPad, DecoderKind::Eager, Windowing::Legacy));
        let cover = transcribe(&model, &bpe, &audio, rate, 5, pass(MelTail::ZeroPad, DecoderKind::Eager, Windowing::Covering));
        let cache = transcribe(&model, &bpe, &audio, rate, 5, pass(MelTail::ZeroPad, DecoderKind::Cached, Windowing::Legacy));
        eprintln!(
            "ABLATE dur={seconds:.1} eager_wall={:.3}s text={:?} seg={} | cover_same={} cover={:?} cover_seg={} | cache_same={}",
            eager.seconds, eager.text, eager.seg, cover.text == eager.text, cover.text, cover.seg, cache.text == eager.text
        );
        assert_eq!(cache.text, eager.text, "cache diverged at {seconds}s");
        assert!(
            !eager.seg.contains("-13.000") || seconds >= 13.0,
            "short clip segment used a 13s window: {}",
            eager.seg
        );
    }

    let prod = WhisperPass::historical();
    assert_eq!(prod.tail, MelTail::ZeroPad);
    assert_eq!(prod.decoder, DecoderKind::Cached);
    assert_eq!(prod.windows, Windowing::Covering);
    let shipped = transcribe(&model, &bpe, &full, rate, 5, prod);
    eprintln!("ABLATE production_default wall={:.3}s text={:?} seg={}", shipped.seconds, shipped.text, shipped.seg);
    assert_eq!(shipped.text, baseline.text);
}
