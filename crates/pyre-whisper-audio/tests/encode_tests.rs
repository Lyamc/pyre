use pyre_whisper_audio::{encode_result, encode_result_with_options, format_srt, group_segments_into_sentences};
use pyre_whisper_core::{OutputFormat, TranscriptResult, TranscriptSegment};

#[test]
fn srt_formatting_matches_expected_pattern() {
    let segments = vec![TranscriptSegment::new(0.0, 2.0, "Hello world")];
    let srt = format_srt(&segments, false);
    assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
    assert!(srt.contains("Hello world"));
}

#[test]
fn json_encoding_round_trips_fields() {
    let result = TranscriptResult {
        text: "hi".to_string(),
        segments: vec![TranscriptSegment::new(0.0, 1.0, "hi")],
        language: Some("en".to_string()),
        model: "tiny_en".to_string(),
        task: "transcribe".to_string(),
    };

    let bytes = encode_result(&result, OutputFormat::Json).unwrap();
    let parsed: TranscriptResult = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed.text, "hi");
    assert_eq!(parsed.model, "tiny_en");
}

#[test]
fn sentence_grouping_merges_by_punct() {
    let segs = vec![
        TranscriptSegment::new(0.0, 0.6, "Hello"),
        TranscriptSegment::new(0.6, 1.0, "there."),
        TranscriptSegment::new(1.1, 1.8, "How are"),
        TranscriptSegment::new(1.8, 2.0, "you?"),
        TranscriptSegment::new(2.1, 2.5, "Fine thanks."),
    ];
    let grouped = group_segments_into_sentences(segs);
    // First two segments -> "Hello there."
    // Next two -> "How are you?"
    // Last -> "Fine thanks."
    assert_eq!(grouped.len(), 3);
    assert!(grouped[0].text.contains("Hello") && grouped[0].text.contains("there."));
    assert!(grouped[1].text.contains("you?"));
    assert!(grouped[2].text.contains("Fine"));
}

#[test]
fn encode_with_sentences_adds_field_in_json() {
    let result = TranscriptResult {
        text: "A. B?".into(),
        segments: vec![
            TranscriptSegment::new(0.0, 0.5, "A."),
            TranscriptSegment::new(0.5, 1.0, "B?"),
        ],
        language: Some("en".into()),
        model: "x".into(),
        task: "transcribe".into(),
    };
    let bytes = encode_result_with_options(&result, OutputFormat::Json, true, true).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("sentences").is_some());
    assert!(v["sentences"].as_array().unwrap().len() >= 1);
}

/// Generate verification artifacts exercising formats + diarize + sentences for representative "model" outputs.
/// Run with: cargo test -p pyre-whisper-audio --test encode_tests verify_model_output_matrix -- --nocapture
#[test]
fn verify_model_output_matrix() {
    use std::fs;

    let out_dir = "verify-outputs";
    let _ = fs::create_dir_all(out_dir);

    // Simulate results from different model families (some fragments to demo sentence grouping)
    let base_result = |model: &str, has_speakers: bool| -> TranscriptResult {
        let mut segs = vec![
            TranscriptSegment::new(0.00, 0.70, "Hello"),
            TranscriptSegment::new(0.70, 1.23, "and welcome."),
            TranscriptSegment::new(1.23, 2.00, "This is"),
            TranscriptSegment::new(2.00, 3.10, "a test of the speech system."),
            TranscriptSegment::new(3.10, 4.00, "It supports"),
            TranscriptSegment::new(4.00, 4.80, "multiple output formats."),
            TranscriptSegment::new(4.90, 5.60, "Speaker changes"),
            TranscriptSegment::new(5.60, 6.20, "and sentences too?"),
        ];
        if has_speakers {
            segs[0].speaker = Some("SPEAKER_00".into());
            segs[1].speaker = Some("SPEAKER_00".into());
            segs[2].speaker = Some("SPEAKER_01".into());
            segs[3].speaker = Some("SPEAKER_01".into());
        }
        TranscriptResult {
            text: segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" "),
            segments: segs,
            language: Some("en".into()),
            model: model.into(),
            task: if has_speakers { "diarize".into() } else { "transcribe".into() },
        }
    };

    let models = ["tiny_en", "parakeet-tdt-0.6b-v3", "qwen3-asr-0.6b"];
    let formats = [OutputFormat::Json, OutputFormat::Txt, OutputFormat::Srt, OutputFormat::Vtt];
    let mut wrote = 0usize;

    for &m in &models {
        for with_diar in [false, true] {
            let res = base_result(m, with_diar);
            for &fmt in &formats {
                for sent in [false, true] {
                    let data = encode_result_with_options(&res, fmt, true, sent).expect("encode");
                    let tag = format!(
                        "{m}__{f}__{d}__{s}",
                        m = m,
                        f = fmt.file_extension(),
                        d = if with_diar { "diarize" } else { "plain" },
                        s = if sent { "sentences" } else { "segments" }
                    );
                    let path = format!("{}/{}.{}", out_dir, tag, fmt.file_extension());
                    fs::write(&path, &data).unwrap();
                    wrote += 1;
                    if sent && fmt == OutputFormat::Json {
                        // also sanity check sentences present
                        let v: serde_json::Value = serde_json::from_slice(&data).unwrap();
                        assert!(v.get("sentences").is_some() || v.get("segments").is_some());
                    }
                }
            }
        }
    }
    println!("verify: wrote {} artifacts under {}", wrote, out_dir);
}

/// Specific verification for Parakeet + SRT.
/// Parakeet (currently) produces only a single coarse segment (no <timestamp> tokens in its vocab),
/// so SRT must still emit a valid, usable single-block SRT using the fallback timing.
#[test]
fn parakeet_srt_output_works() {
    // Simulate what runtime + manager actually feed for a Parakeet result
    // (one segment, full text, no per-word timestamps from the model).
    let full_text = "ask not what your country can do for you ask what you can do for your country";
    let parakeet_like = TranscriptResult {
        text: full_text.to_string(),
        segments: vec![TranscriptSegment {
            start: 0.0,
            end: 11.8, // approximate duration that the fallback logic would produce
            text: full_text.to_string(),
            speaker: None,
        }],
        language: Some("en".to_string()),
        model: "parakeet-tdt-0.6b-v3".to_string(),
        task: "transcribe".to_string(),
    };

    // Plain SRT (the common case: whisburn transcribe --model parakeet... --format srt)
    let srt = String::from_utf8(encode_result_with_options(&parakeet_like, OutputFormat::Srt, true, false).unwrap()).unwrap();
    println!("PARAKEET_PLAIN_SRT:\n{}", srt);
    assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:11,800\n"), "bad header: {srt}");
    assert!(srt.contains("ask not what your country"), "text missing");
    assert!(srt.trim().ends_with("country"), "text truncated");
    // Exactly one block
    assert_eq!(srt.matches('\n').count(), 3, "expected 3 newlines for single block SRT");

    // With sentences=true (grouping on a single segment is a no-op)
    let srt_sent = String::from_utf8(encode_result_with_options(&parakeet_like, OutputFormat::Srt, true, true).unwrap()).unwrap();
    assert_eq!(srt_sent, srt, "sentences grouping must not change single-segment SRT");

    // Now simulate what diarize path does (alternating speaker labels applied in manager)
    let mut with_speaker = parakeet_like.clone();
    with_speaker.segments[0].speaker = Some("SPEAKER_00".to_string());
    with_speaker.task = "diarize".to_string();
    let srt_diar = String::from_utf8(encode_result_with_options(&with_speaker, OutputFormat::Srt, true, false).unwrap()).unwrap();
    println!("PARAKEET_DIARIZE_SRT:\n{}", srt_diar);
    assert!(srt_diar.contains("[SPEAKER_00]"), "speaker tag missing in diarize SRT");
    assert!(srt_diar.contains("ask not"));

    // When caller passes use_segment_timestamps=false (e.g. for en-style), SRT still uses segments
    // (current design: SRT always emits timed blocks from whatever segments exist)
    let srt_no_ts_flag = String::from_utf8(encode_result_with_options(&parakeet_like, OutputFormat::Srt, false, false).unwrap()).unwrap();
    assert!(srt_no_ts_flag.contains("00:00:00,000 --> 00:00:11,800"));

    // Also write the canonical "Parakeet SRT" examples to the verify dir for inspection
    use std::fs;
    let _ = fs::create_dir_all("verify-outputs");
    fs::write("verify-outputs/parakeet_sim__srt__plain__segments.srt", &srt).unwrap();
    fs::write("verify-outputs/parakeet_sim__srt__diarize__segments.srt", &srt_diar).unwrap();
    println!("Wrote parakeet-specific SRT simulations to verify-outputs/parakeet_sim__srt__*.srt");
}