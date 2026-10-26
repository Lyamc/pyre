use std::io::Write;
use std::process::Stdio;

use pyre_whisper_core::WhisburnError;
use tempfile::Builder;

use crate::encode::encode_wav;
use crate::waveform::Waveform;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioExportFormat {
    Wav,
    OpusOgg,
}

impl AudioExportFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "opus" | "ogg" | "opus-ogg" | "audio/ogg" => Some(Self::OpusOgg),
            _ => None,
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::OpusOgg => "audio/ogg",
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::OpusOgg => "ogg",
        }
    }
}

/// Speech-oriented Opus bitrates for preview / streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBitrate {
    Low,
    Medium,
}

impl OpusBitrate {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "low" | "32" | "32k" => Some(Self::Low),
            "medium" | "med" | "64" | "64k" => Some(Self::Medium),
            _ => None,
        }
    }

    pub fn kbps(&self) -> u32 {
        match self {
            Self::Low => 32,
            Self::Medium => 64,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
        }
    }
}

pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Transcode compressed upload bytes via ffmpeg (no full PCM decode in memory).
pub fn transcode_bytes(
    data: &[u8],
    extension: &str,
    format: AudioExportFormat,
    opus_bitrate: OpusBitrate,
    max_seconds: Option<u64>,
) -> Result<Vec<u8>, WhisburnError> {
    if !ffmpeg_available() {
        return Err(WhisburnError::UnsupportedCapability {
            model: "transcode".into(),
            capability: "transcoding requires ffmpeg in PATH".into(),
        });
    }

    let suffix = extension.trim_start_matches('.');
    let mut temp = Builder::new()
        .suffix(&format!(".{suffix}"))
        .tempfile()
        .map_err(|e| WhisburnError::Audio(format!("temp file: {e}")))?;
    temp.write_all(data)
        .map_err(|e| WhisburnError::Audio(format!("temp write: {e}")))?;
    temp.flush()
        .map_err(|e| WhisburnError::Audio(format!("temp flush: {e}")))?;

    let input = temp.path().to_string_lossy().into_owned();
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
    ];
    if let Some(max) = max_seconds {
        args.push("-t".to_string());
        args.push(max.to_string());
    }
    args.push("-i".to_string());
    args.push(input);

    match format {
        AudioExportFormat::Wav => {
            args.extend([
                "-vn".to_string(),
                "-ac".to_string(),
                "1".to_string(),
                "-ar".to_string(),
                "16000".to_string(),
                "-c:a".to_string(),
                "pcm_s16le".to_string(),
                "-f".to_string(),
                "wav".to_string(),
            ]);
        }
        AudioExportFormat::OpusOgg => {
            let kbps = opus_bitrate.kbps();
            args.extend([
                "-vn".to_string(),
                "-c:a".to_string(),
                "libopus".to_string(),
                "-b:a".to_string(),
                format!("{kbps}k"),
                "-application".to_string(),
                "voip".to_string(),
                "-f".to_string(),
                "ogg".to_string(),
            ]);
        }
    }
    args.push("pipe:1".to_string());

    let output = std::process::Command::new("ffmpeg")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| WhisburnError::Audio(format!("failed to spawn ffmpeg: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(WhisburnError::Audio(format!(
            "ffmpeg transcode failed: {stderr}"
        )));
    }

    if output.stdout.is_empty() {
        return Err(WhisburnError::Audio("ffmpeg produced empty output".into()));
    }

    Ok(output.stdout)
}

pub fn transcode_waveform(
    waveform: &Waveform,
    format: AudioExportFormat,
    opus_bitrate: OpusBitrate,
) -> Result<Vec<u8>, WhisburnError> {
    match format {
        AudioExportFormat::Wav => encode_wav(waveform),
        AudioExportFormat::OpusOgg => encode_opus_ogg(waveform, opus_bitrate),
    }
}

fn encode_opus_ogg(waveform: &Waveform, bitrate: OpusBitrate) -> Result<Vec<u8>, WhisburnError> {
    if !ffmpeg_available() {
        return Err(WhisburnError::UnsupportedCapability {
            model: "transcode".into(),
            capability: "opus encoding requires ffmpeg in PATH".into(),
        });
    }

    let wav_bytes = encode_wav(waveform)?;
    let kbps = bitrate.kbps();

    let mut child = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-c:a",
            "libopus",
            "-b:a",
            &format!("{kbps}k"),
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| WhisburnError::Audio(format!("failed to spawn ffmpeg: {e}")))?;

    use std::io::Write;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&wav_bytes)
            .map_err(|e| WhisburnError::Audio(format!("ffmpeg stdin: {e}")))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| WhisburnError::Audio(format!("ffmpeg wait: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(WhisburnError::Audio(format!(
            "ffmpeg opus transcode failed: {stderr}"
        )));
    }

    if output.stdout.is_empty() {
        return Err(WhisburnError::Audio("ffmpeg produced empty opus output".into()));
    }

    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::TARGET_SAMPLE_RATE;

    #[test]
    fn wav_transcode_round_trip() {
        let wf = Waveform::new(vec![0.0, 0.25, -0.25, 0.5], TARGET_SAMPLE_RATE);
        let out = transcode_waveform(&wf, AudioExportFormat::Wav, OpusBitrate::Low).unwrap();
        assert!(out.starts_with(b"RIFF"));
    }
}