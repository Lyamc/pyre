pub mod mel;
pub mod mels_to_text;
pub mod utils;
pub mod decode;
pub mod task;
pub mod whisper_config;

pub use task::DecodeTask;

pub use utils::{get_output_path, format_srt, format_timestamp, parse_time, TranscriptSegment, clean_text, find_chunk_overlap, trim_trailing_hallucination};

use crate::audio::max_waveform_samples;
use crate::model::{get_model_stats, Model};
use crate::token::{self, *};
use self::mel::waveform_to_mel_tensor;
use self::decode::*;

use indicatif::{ProgressBar, ProgressStyle};

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
    };

    let n_waveform_samples_per_window = chunk_size_samples.unwrap_or(match model {
        Model::Whisper(_) => max_waveform_samples(model.encoder_ctx_size() - padding),
        _ => model.encoder_ctx_size() * 160,
    });

    let is_nemo = matches!(model, Model::Parakeet(_));
    let n_mels = model.encoder_mel_size();

    if debug {
        println!("DEBUG: Waveform samples: {}, model: {}", waveform.len(), model_name);
    }

    let mel_iter = waveform_to_mel_tensor(waveform, sample_rate, n_waveform_samples_per_window, &device, n_mels, is_nemo, model_name);
    
    // Apply Global Normalization if stats exist
    let (mean, std) = get_model_stats(model_name, &device);
    
    let mel_vec: Vec<_> = if let (Some(m), Some(s)) = (mean, std) {
        mel_iter.map(|mel| (mel - m.clone().unsqueeze()) / s.clone().unsqueeze()).collect()
    } else {
        mel_iter.collect()
    };

    let total_chunks = mel_vec.len();
    if debug { println!("DEBUG WAVEFORM_TO_TEXT: total_chunks: {}", total_chunks); }
    
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
    let chunk_overlap = 3.0; 
    let shift_duration = chunk_duration - chunk_overlap;

    let mut global_last_end = 0.0;
    let mut recent_texts: std::collections::VecDeque<String> = std::collections::VecDeque::with_capacity(10);
    let mut tokens: Vec<usize> = Vec::new();
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    let mut accumulated_text = String::new();

    for (i, mel) in mel_vec.into_iter().enumerate() {
        if verbose && !quiet {
            let msg = format!("Processing chunk {}/{}...", i + 1, total_chunks);
            if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
        }

        let (new_text, new_tokens) = match model {
            Model::Whisper(m) => decode_whisper(
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
            )?,
            Model::Parakeet(m) => decode_parakeet(m, bpe, mel, model_name, verbose),
            Model::TONE(_) => unreachable!("t-one handled before mel loop"),
            Model::Qwen3(_) => unreachable!("qwen3 handled before mel loop"),
            Model::VibeVoice(_) => unreachable!("vibevoice handled before mel loop"),
            Model::Moonshine(_) => unreachable!("moonshine handled before mel loop"),
        };

        let cleaned = clean_text(&new_text);
        if !cleaned.is_empty() {
            if !accumulated_text.is_empty() && !cleaned.starts_with(' ') {
                accumulated_text.push(' ');
            }
            accumulated_text.push_str(&cleaned);
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
    }

    if let Some(p) = pb { p.finish_with_message("Transcription complete."); }

    let final_text = if accumulated_text.is_empty() {
        trim_trailing_hallucination(&clean_text(&bpe.decode(&tokens[..], true)?))
    } else {
        trim_trailing_hallucination(&accumulated_text)
    };
    if segments.is_empty() {
        segments.push(TranscriptSegment {
            start: 0.0,
            end: (total_chunks as f64 * shift_duration) + chunk_overlap,
            text: final_text.clone(),
            diarization: None,
            location: None,
        });
    }

    Ok((final_text, segments))
}
