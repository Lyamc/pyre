mod cluster;
mod embed;
mod energy;

pub use embed::SpeakerEmbedder;

use pyre_whisper_core::{TranscriptResult, TranscriptSegment};

use crate::model::vad::VadSegment;

pub fn is_diarize_model(name: &str) -> bool {
    matches!(name, "diarization-3.1" | "nemo-diarization")
}

pub fn run_diarize(
    model_name: &str,
    samples: &[f32],
    sample_rate: usize,
) -> Result<TranscriptResult, String> {
    let labeled = speaker_timeline(model_name, samples, sample_rate)?;
    let duration = samples.len() as f64 / sample_rate.max(1) as f64;
    let segments: Vec<TranscriptSegment> = labeled
        .into_iter()
        .map(|(start, end, spk)| {
            TranscriptSegment::new(start, end.min(duration), "").with_speaker(spk)
        })
        .collect();
    let n_spk = segments
        .iter()
        .filter_map(|s| s.speaker.as_deref())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let text = format!("{n_spk} speaker(s)");
    Ok(TranscriptResult {
        text,
        segments,
        language: None,
        model: model_name.to_string(),
        task: "diarize".into(),
    })
}

pub fn assign_speakers(
    mut result: TranscriptResult,
    samples: &[f32],
    sample_rate: usize,
    diarizer: &str,
) -> TranscriptResult {
    let Ok(timeline) = speaker_timeline(diarizer, samples, sample_rate) else {
        return result;
    };
    for seg in &mut result.segments {
        let mid = 0.5 * (seg.start + seg.end);
        if let Some((_, _, spk)) = timeline
            .iter()
            .find(|(s, e, _)| mid >= *s && mid < *e)
            .cloned()
        {
            seg.speaker = Some(spk);
        }
    }
    result.task = "diarize".into();
    result
}

fn speaker_timeline(
    model_name: &str,
    samples: &[f32],
    sample_rate: usize,
) -> Result<Vec<(f64, f64, String)>, String> {
    let dir = pyre_whisper_core::resolve_model_dir(model_name);
    let embedder = SpeakerEmbedder::from_dir(&dir)?;
    let speech = speech_regions(samples, sample_rate);
    if speech.is_empty() {
        return Ok(Vec::new());
    }
    let max_win = 2 * sample_rate;
    let mut embeds = Vec::new();
    let mut spans = Vec::new();
    for region in &speech {
        let a = (region.start * sample_rate as f64) as usize;
        let b = ((region.end * sample_rate as f64) as usize).min(samples.len());
        if b.saturating_sub(a) < sample_rate / 4 {
            continue;
        }
        let mid = (a + b) / 2;
        let half = max_win / 2;
        let i = a.max(mid.saturating_sub(half));
        let j = b.min(i + max_win);
        let e = embedder.embed(&samples[i..j], sample_rate)?;
        embeds.push(e);
        spans.push((
            a as f64 / sample_rate as f64,
            b as f64 / sample_rate as f64,
        ));
    }
    if embeds.is_empty() {
        return Ok(Vec::new());
    }
    let labels = cluster::agglomerative(&embeds, 0.65);
    let mut out = Vec::new();
    for (idx, (start, end)) in spans.into_iter().enumerate() {
        let spk = format!("SPEAKER_{:02}", labels[idx]);
        if let Some(last) = out.last_mut() {
            let (_ls, le, lspk): &mut (f64, f64, String) = last;
            if *lspk == spk && start <= *le + 0.2 {
                *le = end.max(*le);
                continue;
            }
        }
        out.push((start, end, spk));
    }
    Ok(out)
}

fn speech_regions(samples: &[f32], sample_rate: usize) -> Vec<VadSegment> {
    let silero_dir = pyre_whisper_core::resolve_model_dir("silero-vad");
    if silero_dir.join("model.safetensors").exists() {
        if let Ok(v) = crate::model::vad::SileroVad::from_dir(&silero_dir) {
            let segs = v.segments(samples, sample_rate);
            if !segs.is_empty() {
                return segs;
            }
        }
    }
    energy::energy_segments(samples, sample_rate)
}
