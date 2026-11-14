use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use pyre_whisper_core::WhisburnError;

use crate::resample::resample_mono;
use crate::waveform::{Waveform, TARGET_SAMPLE_RATE};

/// Decode/transcribe in ~10-minute windows so multi-hour files never allocate full PCM.
pub const TRANSCRIBE_SEGMENT_SECS: f64 = 600.0;
pub const TRANSCRIBE_SEGMENT_OVERLAP_SECS: f64 = 5.0;
/// Below this duration (and modest file size) a single in-memory decode is fine.
pub const INLINE_DECODE_MAX_SECS: f64 = 20.0 * 60.0;
pub const LARGE_UPLOAD_BYTES: usize = 8 * 1024 * 1024;

struct ProbedAudio {
    format: Box<dyn FormatReader>,
    track_id: u32,
    sample_rate: usize,
    probed_duration_secs: Option<f64>,
}

pub fn decode_to_mono_pcm(path: &Path) -> Result<Waveform, WhisburnError> {
    let file = File::open(path).map_err(|e| WhisburnError::Audio(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    decode_from_mss(mss, &hint, None)
}

pub fn decode_bytes_to_mono_pcm(data: &[u8], extension: &str) -> Result<Waveform, WhisburnError> {
    decode_bytes_to_mono_pcm_with_max_duration(data.to_vec(), extension, None)
}

pub fn decode_bytes_to_mono_pcm_with_max_duration(
    data: Vec<u8>,
    extension: &str,
    max_seconds: Option<u64>,
) -> Result<Waveform, WhisburnError> {
    let cursor = std::io::Cursor::new(data);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    decode_from_mss(mss, &hint, max_seconds)
}

pub fn should_transcribe_in_segments(byte_len: usize, duration_secs: Option<f64>) -> bool {
    if byte_len > LARGE_UPLOAD_BYTES {
        return true;
    }
    match duration_secs {
        Some(secs) => secs > INLINE_DECODE_MAX_SECS,
        None => byte_len > 1024 * 1024,
    }
}

/// Decode compressed audio in time-based segments, invoking `callback` for each (~16 kHz mono chunk).
pub fn for_each_decoded_segment<F>(
    data: Vec<u8>,
    extension: &str,
    segment_secs: f64,
    overlap_secs: f64,
    max_segments: Option<usize>,
    mut callback: F,
) -> Result<(), WhisburnError>
where
    F: FnMut(Waveform) -> Result<(), WhisburnError>,
{
    let cursor = std::io::Cursor::new(data);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    let probed = probe_audio(mss, &hint)?;
    let ProbedAudio {
        mut format,
        track_id,
        sample_rate,
        ..
    } = probed;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.id == track_id)
        .ok_or_else(|| WhisburnError::Audio("audio track disappeared after probe".into()))?;

    let mut decoder = crate::opus::codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(map_symphonia_error)?;
    // The decoder knows its output rate (Opus always decodes at 48 kHz, whatever the container says).
    let sample_rate = decoder.codec_params().sample_rate.map(|r| r as usize).unwrap_or(sample_rate);

    let segment_samples = (segment_secs * sample_rate as f64).round() as usize;
    let overlap_samples = (overlap_secs * sample_rate as f64).round() as usize;
    let mut mono_samples = Vec::new();
    let mut segments_emitted = 0usize;

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_decoded(&mut mono_samples, decoded);
                while mono_samples.len() >= segment_samples {
                    flush_decode_segment(
                        &mut mono_samples,
                        sample_rate,
                        segment_samples,
                        overlap_samples,
                        &mut callback,
                    )?;
                    segments_emitted += 1;
                    if max_segments.is_some_and(|max| segments_emitted >= max) {
                        return Ok(());
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(map_symphonia_error(e)),
        }
    }

    if !mono_samples.is_empty() && !max_segments.is_some_and(|max| segments_emitted >= max) {
        let samples = resample_to_target(&mono_samples, sample_rate)?;
        callback(Waveform::new(samples, TARGET_SAMPLE_RATE))?;
    }

    Ok(())
}

fn flush_decode_segment<F>(
    mono_samples: &mut Vec<f32>,
    sample_rate: usize,
    segment_samples: usize,
    overlap_samples: usize,
    callback: &mut F,
) -> Result<(), WhisburnError>
where
    F: FnMut(Waveform) -> Result<(), WhisburnError>,
{
    let emit = mono_samples[..segment_samples].to_vec();
    let overlap_start = segment_samples.saturating_sub(overlap_samples);
    let remainder = mono_samples[overlap_start..].to_vec();
    mono_samples.clear();
    mono_samples.extend_from_slice(&remainder);

    let samples = resample_to_target(&emit, sample_rate)?;
    callback(Waveform::new(samples, TARGET_SAMPLE_RATE))
}

fn resample_to_target(mono_samples: &[f32], sample_rate: usize) -> Result<Vec<f32>, WhisburnError> {
    if sample_rate == TARGET_SAMPLE_RATE {
        Ok(mono_samples.to_vec())
    } else {
        resample_mono(mono_samples, sample_rate, TARGET_SAMPLE_RATE)
    }
}

pub fn probe_bytes_duration_secs(data: &[u8], extension: &str) -> Result<Option<f64>, WhisburnError> {
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(extension);

    let probed = probe_audio(mss, &hint)?;
    Ok(probed.probed_duration_secs)
}

fn decode_from_mss(
    mss: MediaSourceStream,
    hint: &Hint,
    max_seconds: Option<u64>,
) -> Result<Waveform, WhisburnError> {
    let probed = probe_audio(mss, hint)?;
    if let Some(max_seconds) = max_seconds {
        reject_if_over_limit(probed.probed_duration_secs, max_seconds)?;
    }

    let ProbedAudio {
        mut format,
        track_id,
        sample_rate,
        probed_duration_secs,
    } = probed;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.id == track_id)
        .ok_or_else(|| WhisburnError::Audio("audio track disappeared after probe".into()))?;

    let mut decoder = crate::opus::codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(map_symphonia_error)?;
    // The decoder knows its output rate (Opus always decodes at 48 kHz, whatever the container says).
    let sample_rate = decoder.codec_params().sample_rate.map(|r| r as usize).unwrap_or(sample_rate);

    let max_native_samples = max_seconds.map(|max| {
        sample_rate
            .saturating_mul(max as usize)
            .saturating_add(sample_rate)
    });

    let mut mono_samples = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_decoded(&mut mono_samples, decoded);
                if let Some(limit) = max_native_samples {
                    if mono_samples.len() > limit {
                        let secs = probed_duration_secs.unwrap_or_else(|| {
                            mono_samples.len() as f64 / sample_rate as f64
                        });
                        return Err(duration_limit_error(secs, max_seconds.unwrap()));
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(map_symphonia_error(e)),
        }
    }

    if mono_samples.is_empty() {
        return Err(WhisburnError::Audio("decoded audio is empty".into()));
    }

    if let Some(max_seconds) = max_seconds {
        let secs = mono_samples.len() as f64 / sample_rate as f64;
        reject_if_over_limit(Some(secs), max_seconds)?;
    }

    let samples = resample_to_target(&mono_samples, sample_rate)?;

    Ok(Waveform::new(samples, TARGET_SAMPLE_RATE))
}

fn probe_audio(mss: MediaSourceStream, hint: &Hint) -> Result<ProbedAudio, WhisburnError> {
    let probed = symphonia::default::get_probe()
        .format(hint, mss, &Default::default(), &MetadataOptions::default())
        .map_err(map_symphonia_error)?;

    let format = probed.format;
    // Video containers list a video track first. An audio track is the one that reports a sample rate.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.sample_rate.is_some() && t.codec_params.codec != CODEC_TYPE_NULL)
        .or_else(|| format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL))
        .ok_or_else(|| WhisburnError::Audio("no audio track found".into()))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| WhisburnError::Audio("unknown sample rate".into()))? as usize;
    let probed_duration_secs = track_duration_secs(&track.codec_params);

    Ok(ProbedAudio {
        format,
        track_id,
        sample_rate,
        probed_duration_secs,
    })
}

fn track_duration_secs(params: &CodecParameters) -> Option<f64> {
    if let (Some(n_frames), Some(time_base)) = (params.n_frames, params.time_base) {
        let time = time_base.calc_time(n_frames);
        return Some(time.seconds as f64 + time.frac);
    }

    if let (Some(n_frames), Some(sample_rate)) = (params.n_frames, params.sample_rate) {
        if sample_rate > 0 {
            return Some(n_frames as f64 / sample_rate as f64);
        }
    }

    None
}

fn reject_if_over_limit(duration_secs: Option<f64>, max_seconds: u64) -> Result<(), WhisburnError> {
    if let Some(secs) = duration_secs {
        if secs > max_seconds as f64 {
            return Err(duration_limit_error(secs, max_seconds));
        }
    }
    Ok(())
}

fn duration_limit_error(secs: f64, max_seconds: u64) -> WhisburnError {
    WhisburnError::Audio(format!(
        "audio is {:.0}s long; server limit is {max_seconds}s — trim or split the file before uploading",
        secs
    ))
}

fn append_decoded(out: &mut Vec<f32>, decoded: AudioBufferRef<'_>) {
    match decoded {
        AudioBufferRef::F32(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame];
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S16(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / i16::MAX as f32;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S32(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / i32::MAX as f32;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S24(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame].inner() as f32 / 8_388_607.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S8(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / i8::MAX as f32;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U8(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += (buf.chan(ch)[frame] as f32 - 128.0) / 128.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::F64(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f64;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame];
                }
                out.push((sum / channels as f64) as f32);
            }
        }
        AudioBufferRef::U16(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / u16::MAX as f32;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U24(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame].inner() as f32 / 16_777_215.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U32(buf) => {
            let channels = buf.spec().channels.count();
            for frame in 0..buf.frames() {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / u32::MAX as f32;
                }
                out.push(sum / channels as f32);
            }
        }
    }
}

fn map_symphonia_error(err: SymphoniaError) -> WhisburnError {
    WhisburnError::Audio(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn sample_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples").join(name)
    }

    #[test]
    fn probe_jfk_duration_is_short() {
        let path = sample_path("jfk.wav");
        if !path.exists() {
            return;
        }
        let data = std::fs::read(path).unwrap();
        let secs = probe_bytes_duration_secs(&data, "wav").unwrap().unwrap();
        assert!(secs > 5.0 && secs < 15.0, "unexpected jfk duration: {secs}");
    }

    #[test]
    fn probe_long_mp3_reports_over_limit() {
        let path = sample_path("recording (42).mp3");
        if !path.exists() {
            return;
        }
        let data = std::fs::read(path).unwrap();
        let secs = probe_bytes_duration_secs(&data, "mp3").unwrap().unwrap();
        assert!(secs > 3600.0, "expected 3h+ recording, got {secs}s");
    }

    #[test]
    fn long_mp3_decodes_in_multiple_segments() {
        let path = sample_path("recording (42).mp3");
        if !path.exists() {
            return;
        }
        let mut data = std::fs::read(path).unwrap();
        data.truncate(4 * 1024 * 1024);
        let mut chunks = 0usize;
        for_each_decoded_segment(data, "mp3", 60.0, 5.0, Some(3), |_| {
            chunks += 1;
            Ok(())
        })
        .unwrap();
        assert!(chunks >= 2, "expected multiple segments, got {chunks}");
    }
}