use pyre_whisper_core::TranscriptResult;

/// Lightweight placeholder diarization until Burn backends for pyannote/NeMo land.
/// Alternates speakers on segment boundaries to provide usable RTTM/SRT speaker tags.
pub fn assign_alternating_speakers(result: TranscriptResult) -> TranscriptResult {
    let segments = result
        .segments
        .into_iter()
        .enumerate()
        .map(|(i, mut seg)| {
            seg.speaker = Some(format!("SPEAKER_{:02}", i % 2));
            seg
        })
        .collect();

    TranscriptResult { segments, ..result }
}

pub fn assign_single_speaker(result: TranscriptResult, speaker: &str) -> TranscriptResult {
    let segments = result
        .segments
        .into_iter()
        .map(|mut seg| {
            seg.speaker = Some(speaker.to_string());
            seg
        })
        .collect();

    TranscriptResult { segments, ..result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_whisper_core::TranscriptSegment;

    #[test]
    fn alternating_speakers_toggle() {
        let result = TranscriptResult {
            text: "a b".into(),
            segments: vec![
                TranscriptSegment::new(0.0, 1.0, "a"),
                TranscriptSegment::new(1.0, 2.0, "b"),
                TranscriptSegment::new(2.0, 3.0, "c"),
            ],
            language: Some("en".into()),
            model: "tiny_en".into(),
            task: "diarize".into(),
        };

        let out = assign_alternating_speakers(result);
        assert_eq!(out.segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(out.segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(out.segments[2].speaker.as_deref(), Some("SPEAKER_00"));
    }
}