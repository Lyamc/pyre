mod host;
mod segments;
mod silero;
mod ten;
pub mod weights;

pub use segments::{probs_to_segments, VadSegment, VadSettings};
pub use silero::SileroVad;
pub use ten::{TenFeatures, TenVad};

use pyre_whisper_core::{TranscriptResult, TranscriptSegment};

pub fn is_vad_model(name: &str) -> bool {
    matches!(name, "silero-vad" | "ten-vad")
}

pub fn run_vad(
    model_name: &str,
    samples: &[f32],
    sample_rate: usize,
) -> Result<TranscriptResult, String> {
    let dir = pyre_whisper_core::resolve_model_dir(model_name);
    let segs = match model_name {
        "silero-vad" => SileroVad::from_dir(&dir)?.segments(samples, sample_rate),
        "ten-vad" => TenVad::from_dir(&dir)?.segments(samples, sample_rate),
        other => return Err(format!("not a VAD model: {other}")),
    };
    let duration = samples.len() as f64 / sample_rate.max(1) as f64;
    let segments: Vec<TranscriptSegment> = segs
        .into_iter()
        .map(|s| {
            let end = s.end.min(duration);
            TranscriptSegment::new(s.start.min(end), end, "SPEECH")
        })
        .collect();
    let text = if segments.is_empty() {
        String::new()
    } else {
        format!("{} speech region(s)", segments.len())
    };
    Ok(TranscriptResult {
        text,
        segments,
        language: None,
        model: model_name.to_string(),
        task: "vad".into(),
    })
}
