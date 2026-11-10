pub mod mel;
pub mod mels_to_text;
pub mod pass;
pub mod utils;
pub mod decode;
pub mod task;
pub mod whisper_config;

pub use pass::{with_whisper_pass, DecoderKind, MelTail, WhisperPass, Windowing};

pub use task::DecodeTask;

pub use utils::{get_output_path, format_srt, format_timestamp, parse_time, TranscriptSegment, append_without_word_overlap, clean_text, drop_left_context_tokens, find_chunk_overlap, timestamps_collapsed, trim_trailing_hallucination};

use crate::audio::max_waveform_samples;
use crate::model::{get_model_stats, Model};
use crate::token::{self, *};
use self::mel::slice_to_mel;
use self::decode::*;

use indicatif::{ProgressBar, ProgressStyle};

/// CLI gate. The HTTP server applies the same rule from the request flag before it
/// reaches the engine, so this only consults `PYRE_STT_EXPERIMENTAL`.
pub fn experimental_gate(model_name: &str, n_samples: usize, sample_rate: usize) -> std::result::Result<(), String> {
    let opted = matches!(
        std::env::var("PYRE_STT_EXPERIMENTAL").ok().as_deref(),
        Some("1" | "true" | "yes" | "on")
    );
    if opted {
        return Ok(());
    }
    let key = model_name.to_ascii_lowercase();
    let v3 = key.contains("parakeet-tdt-0.6b-v3");
    let nemotron = key.contains("nemotron-speech") || key.contains("nemotron-3.5-asr");
    let new_model = v3 || nemotron || key.contains("parakeet");
    let secs = n_samples as f64 / sample_rate.max(1) as f64;
    if v3 || nemotron {
        return Err(format!(
            "{model_name} is experimental and off by default. Set PYRE_STT_EXPERIMENTAL=1 to transcribe with it."
        ));
    }
    if new_model && secs > 60.0 {
        return Err(format!(
            "long audio ({secs:.0}s) on {model_name} is experimental and off by default. Set PYRE_STT_EXPERIMENTAL=1 to run the chunked path."
        ));
    }
    Ok(())
}

pub fn waveform_to_text(
    model: &Model,
    model_name: &str,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    language_code: &str,
    waveform: Vec<f32>,
    stereo_audio: Option<Vec<[f32; 2]>>,
    sample_rate: usize,
    streaming_mode: bool,
    verbose: bool,
    debug: bool,
    quiet: bool,
    include_timestamps: bool,
    backend_info: String,
    beam_size: usize,
    max_tokens: usize,
    padding: usize,
    chunk_size_samples: Option<usize>,
    decode_task: Option<DecodeTask>,
) -> token::Result<(String, Vec<TranscriptSegment>)> {
    let decode_task = decode_task.unwrap_or(DecodeTask::Transcribe);

    if let Model::Moonshine(m) = model {
        let (text, _tokens) =
            decode_moonshine(m, bpe, &waveform, sample_rate, model_name, verbose);
        let final_text = trim_trailing_hallucination(&clean_text(&text));
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        return Ok((final_text, segments));
    }

    if let Model::TONE(m) = model {
        let (text, _tokens) = decode_tone(m, bpe, &waveform, sample_rate, model_name, verbose);
        let final_text = trim_trailing_hallucination(&clean_text(&text));
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        return Ok((final_text, segments));
    }

    if let Model::VibeVoice(m) = model {
        let (text, _tokens) =
            decode_vibevoice(m, bpe, &waveform, sample_rate, model_name, verbose);
        let final_text = trim_trailing_hallucination(&clean_text(&text));
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        return Ok((final_text, segments));
    }

    if let Model::Qwen3(m) = model {
        use burn::tensor::Tensor;
        use crate::audio::prep_audio;
        use crate::audio::qwen3_chunk::{
            split_audio_into_chunks, QWEN3_MAX_ASR_CHUNK_SEC, QWEN3_MIN_ASR_CHUNK_SEC,
        };
        use crate::model::qwen3::{qwen3_forced_language_from_code, QWEN3_DEFAULT_MAX_NEW_TOKENS};

        let device = m.device();
        let n_mels = m.encoder_mel_size();
        let forced_language = qwen3_forced_language_from_code(language_code);
        let _ = lang;

        let chunks = split_audio_into_chunks(
            &waveform,
            sample_rate,
            QWEN3_MAX_ASR_CHUNK_SEC,
            5.0,
            100.0,
            QWEN3_MIN_ASR_CHUNK_SEC,
        );
        let mut accumulated = String::new();
        let mut all_tokens: Vec<usize> = Vec::new();

        for (chunk_idx, (chunk_wav, _offset_sec)) in chunks.into_iter().enumerate() {
            if verbose && !quiet {
                println!(
                    "Qwen3: processing chunk {} ({} samples)",
                    chunk_idx + 1,
                    chunk_wav.len()
                );
            }
            let w = Tensor::<1>::from_floats(chunk_wav.as_slice(), &device).unsqueeze();
            let mel = prep_audio(w, sample_rate as f64, n_mels, false, model_name);
            let qwen3_max_tokens = max_tokens.max(QWEN3_DEFAULT_MAX_NEW_TOKENS);
            let (text, tokens) = decode_qwen3(
                m,
                bpe,
                mel,
                model_name,
                forced_language,
                qwen3_max_tokens,
                verbose,
            );
            let cleaned = clean_text(&text);
            if !cleaned.is_empty() {
                accumulated.push_str(&cleaned);
            }
            all_tokens.extend(tokens);
        }

        let final_text = trim_trailing_hallucination(&accumulated);
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        let _ = all_tokens;
        return Ok((final_text, segments));
    }

    let device = match model {
        Model::Whisper(m) => m.encoder.conv1.weight.device(),
        Model::Parakeet(m) => m.encoder.layers[0].final_ln.gamma.device(),
        Model::TONE(_) => unreachable!("t-one handled before mel loop"),
        Model::Qwen3(_) => unreachable!("qwen3 handled before mel loop"),
        Model::VibeVoice(m) => m.acoustic_connector.fc1.weight.device(),
        Model::Moonshine(_) => unreachable!("moonshine handled before mel loop"),
        Model::Nemotron(m) => m.final_proj.weight.device(),
    };

    let n_waveform_samples_per_window = chunk_size_samples.unwrap_or(match model {
        Model::Whisper(_) => max_waveform_samples(model.encoder_ctx_size() - padding),
        // Parakeet/Nemotron are not Whisper's 1500-frame encoder. A 15 s cut
        // (1500 * 160 samples) dropped the start of every window. 60 s covers
        // a LibriSpeech or MLS utterance in one pass.
        Model::Parakeet(_) | Model::Nemotron(_) => sample_rate.saturating_mul(60),
        _ => model.encoder_ctx_size() * 160,
    });

    let is_nemo = matches!(model, Model::Parakeet(_) | Model::Nemotron(_));
    let n_mels = model.encoder_mel_size();

    if debug {
        println!("DEBUG: Waveform samples: {}, model: {}", waveform.len(), model_name);
    }

    let overlap_secs = if matches!(model, Model::Parakeet(_) | Model::Nemotron(_)) { 8 } else { 3 };
    let overlap = sample_rate.saturating_mul(overlap_secs);
    let (starts, shift) = window_starts(
        waveform.len(),
        n_waveform_samples_per_window,
        overlap,
        pass::current().windows,
    );
    let mut processor = crate::audio::AudioProcessor::new(&device, sample_rate as f64, n_mels, is_nemo, model_name);
    if is_nemo {
        let dir = pyre_whisper_core::resolve_model_dir(model_name);
        let mean_path = dir.join("mean.npy");
        let std_path = dir.join("std.npy");
        if mean_path.exists() && std_path.exists() {
            let dir_str = dir.to_string_lossy();
            let mean = crate::model::load::helpers::load_tensor::<2>("mean", dir_str.as_ref(), &device).unwrap();
            let std = crate::model::load::helpers::load_tensor::<2>("std", dir_str.as_ref(), &device).unwrap();
            processor = processor.with_stats(mean, std);
        }
    }

    let (mean, std) = get_model_stats(model_name, &device);
    let total_chunks = starts.len();
    let samples_total = waveform.len();
    if debug { println!("DEBUG WAVEFORM_TO_TEXT: total_chunks: {}", total_chunks); }
    if total_chunks > 0 {
        crate::job_hooks::report(crate::job_hooks::ChunkReport {
            chunks_done: 0,
            chunks_total: total_chunks,
            samples_done: 0,
            samples_total,
        });
    }
    
    let pb = if !streaming_mode && !quiet {
        let pb = ProgressBar::new(total_chunks as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {per_sec} | ETA: {eta} | {msg}")
            .unwrap()
            .progress_chars("#>-"));
        pb.set_message(backend_info);
        pb.enable_steady_tick(std::time::Duration::from_secs(1));
        Some(pb)
    } else {
        None
    };

    let chunk_duration = n_waveform_samples_per_window as f64 / sample_rate as f64;
    let chunk_overlap = overlap_secs as f64;
    let shift_duration = chunk_duration - chunk_overlap;

    let mut global_last_end = 0.0;
    let mut recent_texts: std::collections::VecDeque<String> = std::collections::VecDeque::with_capacity(10);
    let mut tokens: Vec<usize> = Vec::new();
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    let mut accumulated_text = String::new();

    for (i, &start) in starts.iter().enumerate() {
        if crate::job_hooks::cancelled() {
            return Err("cancelled".into());
        }
        let end = (start + n_waveform_samples_per_window).min(samples_total);
        let mut mel = slice_to_mel(&waveform, start, end, &processor, &device);
        if let (Some(m), Some(s)) = (&mean, &std) {
            mel = (mel - m.clone().unsqueeze()) / s.clone().unsqueeze();
        }
        if verbose && !quiet {
            let msg = format!("Processing chunk {}/{}...", i + 1, total_chunks);
            if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
        }

        let (new_text, new_tokens, token_frames, encoder_frames) = match model {
            Model::Whisper(m) => {
                let (text, tokens) = decode_whisper(
                    m,
                    bpe,
                    lang,
                    mel,
                    padding,
                    streaming_mode,
                    include_timestamps,
                    beam_size,
                    max_tokens,
                    decode_task,
                    model_name,
                )?;
                (text, tokens, Vec::new(), 0)
            }
            Model::Parakeet(m) => {
                let chunk = decode_parakeet(m, bpe, mel, model_name, verbose);
                (chunk.text, chunk.tokens, chunk.frames, chunk.encoder_frames)
            }
            Model::Nemotron(m) => {
                let chunk = decode_nemotron(m, bpe, mel, model_name, verbose);
                (chunk.text, chunk.tokens, chunk.frames, chunk.encoder_frames)
            }
            Model::TONE(_) => unreachable!("t-one handled before mel loop"),
            Model::Qwen3(_) => unreachable!("qwen3 handled before mel loop"),
            Model::VibeVoice(_) => unreachable!("vibevoice handled before mel loop"),
            Model::Moonshine(_) => unreachable!("moonshine handled before mel loop"),
        };

        let use_frames = encoder_frames > 0 && token_frames.len() == new_tokens.len();
        let frame_ids = if use_frames {
            let window_samples = end.saturating_sub(start).max(1);
            let min_frame = if i == 0 {
                0
            } else {
                overlap.saturating_mul(encoder_frames) / window_samples
            };
            drop_left_context_tokens(&new_tokens, &token_frames, min_frame)
        } else {
            Vec::new()
        };
        // v2 timestamped one whole window inside the 8 s overlap, and the frame
        // drop deleted it (151 consecutive words). Fall back to the word stitch.
        let collapsed = i > 0 && use_frames && timestamps_collapsed(&frame_ids, &new_tokens);
        let mut cleaned = if use_frames && !collapsed {
            clean_text(&parakeet_tokens_to_text(bpe, &frame_ids))
        } else {
            clean_text(&new_text)
        };
        // A 60 s window that decodes blank loses ~50 s. One split into halves
        // covers that span at a length v2 already transcribes (single utterances
        // are well under 30 s). v3's good windows are non-empty and skip this.
        let mut recovered = false;
        if cleaned.trim().is_empty()
            && end.saturating_sub(start) > sample_rate.saturating_mul(25)
            && matches!(model, Model::Parakeet(_))
        {
            let mid = start + (end - start) / 2;
            let mut joined = String::new();
            for (a, b) in [(start, mid), (mid, end)] {
                let mut half = slice_to_mel(&waveform, a, b, &processor, &device);
                if let (Some(m), Some(s)) = (&mean, &std) {
                    half = (half - m.clone().unsqueeze()) / s.clone().unsqueeze();
                }
                let text = if let Model::Parakeet(m) = model {
                    clean_text(&decode_parakeet(m, bpe, half, model_name, verbose).text)
                } else {
                    String::new()
                };
                let piece = text.trim();
                if !piece.is_empty() {
                    if !joined.is_empty() {
                        joined.push(' ');
                    }
                    joined.push_str(piece);
                }
            }
            if !joined.is_empty() {
                cleaned = joined;
                recovered = true;
            }
        }
        if use_frames && !collapsed && !recovered {
            let piece = cleaned.trim();
            if !piece.is_empty() {
                if !accumulated_text.is_empty() {
                    accumulated_text.push(' ');
                }
                accumulated_text.push_str(piece);
            }
        } else {
            append_without_word_overlap(&mut accumulated_text, &cleaned);
        }

        let chunk_offset = i as f64 * shift_duration;
        if include_timestamps {
            let mut current_start = chunk_offset.max(global_last_end);
            let mut current_text_tokens = Vec::new();
            for &token in &new_tokens {
                if let Some(ts) = bpe.token_to_timestamp(token) {
                    let absolute_ts = chunk_offset + ts;
                    if !current_text_tokens.is_empty() {
                        let text = bpe.decode(&current_text_tokens, true)?;
                        let trimmed = clean_text(&text);
                        if !trimmed.is_empty() {
                            let mut segment = TranscriptSegment {
                                start: current_start,
                                end: absolute_ts.max(current_start + 0.05),
                                text: trimmed,
                                diarization: None,
                                location: None,
                            };
                            if let Some(ref stereo) = stereo_audio {
                                let s_idx = (segment.start * sample_rate as f64) as usize;
                                let e_idx = (segment.end * sample_rate as f64).min(stereo.len() as f64) as usize;
                                if e_idx > s_idx {
                                    let mut sum_l = 0.0; let mut sum_r = 0.0;
                                    for j in s_idx..e_idx { sum_l += stereo[j][0].abs(); sum_r += stereo[j][1].abs(); }
                                    let pan = ((sum_r - sum_l) / (sum_l + sum_r + 1e-6)) as f64;
                                    segment.location = Some(pan);
                                    segment.diarization = Some(if pan < 0.0 { "Speaker 1".to_string() } else { "Speaker 2".to_string() });
                                }
                            }
                            if segment.end > segment.start {
                                if !recent_texts.contains(&segment.text) {
                                    if !quiet {
                                        let msg = format!("[{:.2}s - {:.2}s]: {}", segment.start, segment.end, segment.text);
                                        if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
                                    }
                                    global_last_end = segment.end;
                                    if recent_texts.len() == 10 { recent_texts.pop_front(); }
                                    recent_texts.push_back(segment.text.clone());
                                    segments.push(segment);
                                }
                            }
                        }
                        current_text_tokens.clear();
                    }
                    current_start = absolute_ts.max(current_start);
                } else if !bpe.is_special(token) {
                    current_text_tokens.push(token);
                }
            }
        }

        if let Some((prev_index, curr_index)) = find_chunk_overlap(&tokens[..], &new_tokens[..], 40, 3) {
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            tokens.extend(new_tokens);
        }

        if !include_timestamps && !quiet && !new_text.trim().is_empty() {
            let msg = format!("[{:.2}s - {:.2}s]: {}", chunk_offset, chunk_offset + chunk_duration, new_text.trim());
            if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
        }

        if let Some(ref p) = pb { p.inc(1); }
        let samples_done = if i + 1 == total_chunks {
            samples_total
        } else {
            (start + shift).min(samples_total)
        };
        crate::job_hooks::report(crate::job_hooks::ChunkReport {
            chunks_done: i + 1,
            chunks_total: total_chunks,
            samples_done,
            samples_total,
        });
    }

    if let Some(p) = pb { p.finish_with_message("Transcription complete."); }

    let final_text = if accumulated_text.is_empty() {
        trim_trailing_hallucination(&clean_text(&bpe.decode(&tokens[..], true)?))
    } else {
        trim_trailing_hallucination(&accumulated_text)
    };
    let audio_end = samples_total as f64 / sample_rate.max(1) as f64;
    if segments.is_empty() {
        segments.push(TranscriptSegment {
            start: 0.0,
            end: audio_end,
            text: final_text.clone(),
            diarization: None,
            location: None,
        });
    }
    for segment in &mut segments {
        segment.start = segment.start.clamp(0.0, audio_end);
        segment.end = segment.end.clamp(segment.start, audio_end);
    }

    Ok((final_text, segments))
}

fn window_starts(len: usize, window: usize, overlap: usize, mode: pass::Windowing) -> (Vec<usize>, usize) {
    let window = window.max(1);
    let overlap = overlap.min(window.saturating_sub(1));
    let shift = (window - overlap).max(1);
    let starts = match mode {
        pass::Windowing::Covering => {
            pyre_whisper_core::plan_windows(len, window, overlap).starts
        }
        pass::Windowing::Legacy => {
            if len == 0 {
                Vec::new()
            } else {
                let n = (len - 1) / shift + 1;
                (0..n).map(|i| i * shift).filter(|start| *start < len).collect()
            }
        }
    };
    (starts, shift)
}
