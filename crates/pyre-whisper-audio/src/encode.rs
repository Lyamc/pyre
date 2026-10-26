use pyre_whisper_core::{group_segments_into_sentences, OutputFormat, TranscriptResult, WhisburnError};

use crate::waveform::Waveform;

pub fn encode_result(result: &TranscriptResult, format: OutputFormat) -> Result<Vec<u8>, WhisburnError> {
    encode_result_with_options(result, format, true, false)
}

pub fn encode_result_with_options(
    result: &TranscriptResult,
    format: OutputFormat,
    use_segment_timestamps: bool,
    group_sentences: bool,
) -> Result<Vec<u8>, WhisburnError> {
    let segments = if group_sentences && !result.segments.is_empty() {
        group_segments_into_sentences(result.segments.clone())
    } else {
        result.segments.clone()
    };
    let has_speakers = segments.iter().any(|s| s.speaker.is_some());
    match format {
        OutputFormat::Json => {
            if group_sentences {
                // For JSON when grouping sentences, attach a sentences field while keeping original
                let mut v = serde_json::to_value(result)?;
                if let Some(obj) = v.as_object_mut() {
                    let sents = group_segments_into_sentences(result.segments.clone());
                    obj.insert("sentences".to_string(), serde_json::to_value(&sents).unwrap_or(serde_json::Value::Null));
                }
                Ok(serde_json::to_vec_pretty(&v)?)
            } else {
                Ok(serde_json::to_vec_pretty(result)?)
            }
        }
        OutputFormat::Txt => {
            let body = if use_segment_timestamps && !segments.is_empty() {
                format_txt_timestamped(&segments, has_speakers)
            } else {
                result.text.clone()
            };
            Ok(body.into_bytes())
        }
        OutputFormat::Srt => Ok(format_srt(&segments, has_speakers).into_bytes()),
        OutputFormat::Vtt => Ok(format_vtt(&segments, has_speakers).into_bytes()),
        OutputFormat::Wav => Err(WhisburnError::InvalidRequest(
            "wav output requires synthesized audio, not transcript".into(),
        )),
    }
}

pub fn encode_wav(waveform: &Waveform) -> Result<Vec<u8>, WhisburnError> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: waveform.sample_rate as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::new(&mut cursor, spec).map_err(|e| WhisburnError::Audio(e.to_string()))?;
        for sample in &waveform.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let int_sample = (clamped * i16::MAX as f32) as i16;
            writer
                .write_sample(int_sample)
                .map_err(|e| WhisburnError::Audio(e.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|e| WhisburnError::Audio(e.to_string()))?;
    }
    Ok(cursor.into_inner())
}

fn format_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours:02}:{mins:02}:{secs:02},{ms:03}")
}

fn format_vtt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3600;
    format!("{hours:02}:{mins:02}:{secs:02}.{ms:03}")
}

fn format_txt_clock(seconds: f64) -> String {
    format!("{:.3}", seconds)
}

pub fn format_txt_timestamped(
    segments: &[pyre_whisper_core::TranscriptSegment],
    include_speaker: bool,
) -> String {
    segments
        .iter()
        .map(|seg| {
            let speaker = if include_speaker {
                seg.speaker
                    .as_ref()
                    .map(|s| format!("{s}: "))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            format!(
                "[{} --> {}] {}{}",
                format_txt_clock(seg.start),
                format_txt_clock(seg.end),
                speaker,
                seg.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_srt(
    segments: &[pyre_whisper_core::TranscriptSegment],
    include_speaker: bool,
) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let text = if include_speaker {
                seg.speaker
                    .as_ref()
                    .map(|s| format!("[{s}] {}", seg.text.trim()))
                    .unwrap_or_else(|| seg.text.trim().to_string())
            } else {
                seg.text.trim().to_string()
            };
            format!(
                "{}\n{} --> {}\n{}\n",
                i + 1,
                format_timestamp(seg.start),
                format_timestamp(seg.end),
                text
            )
        })
        .collect()
}

pub fn format_vtt(
    segments: &[pyre_whisper_core::TranscriptSegment],
    include_speaker: bool,
) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        let text = if include_speaker {
            seg.speaker
                .as_ref()
                .map(|s| format!("<v {s}>{}</v>", seg.text.trim()))
                .unwrap_or_else(|| seg.text.trim().to_string())
        } else {
            seg.text.trim().to_string()
        };
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_vtt_timestamp(seg.start),
            format_vtt_timestamp(seg.end),
            text
        ));
    }
    out
}