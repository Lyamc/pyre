use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

impl TranscriptSegment {
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
            speaker: None,
        }
    }

    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptResult {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub language: Option<String>,
    pub model: String,
    pub task: String,
}

impl TranscriptResult {
    pub fn empty(model: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            text: String::new(),
            segments: Vec::new(),
            language: None,
            model: model.into(),
            task: task.into(),
        }
    }
}

/// Group consecutive segments into sentences using simple terminal punctuation heuristics.
/// Produces fewer, longer segments whose text ends with . ! ? (or last segment).
pub fn group_segments_into_sentences(segments: Vec<TranscriptSegment>) -> Vec<TranscriptSegment> {
    if segments.is_empty() {
        return segments;
    }
    let mut out: Vec<TranscriptSegment> = Vec::new();
    let mut cur_start = segments[0].start;
    let mut cur_end = segments[0].end;
    let mut cur_speaker = segments[0].speaker.clone();
    let mut buf = String::new();

    let flush = |buf: &mut String, start: f64, end: f64, spk: &Option<String>, dst: &mut Vec<TranscriptSegment>| {
        let t = buf.trim().to_string();
        if !t.is_empty() {
            let mut seg = TranscriptSegment { start, end, text: t, speaker: spk.clone() };
            if let Some(s) = spk { seg = seg.with_speaker(s.clone()); }
            dst.push(seg);
        }
        buf.clear();
    };

    for s in segments {
        let text = s.text.trim();
        if text.is_empty() { continue; }
        if buf.is_empty() {
            cur_start = s.start;
            cur_end = s.end;
            cur_speaker = s.speaker.clone();
        } else {
            cur_end = s.end.max(cur_end);
            // speaker change forces sentence boundary
            if s.speaker != cur_speaker {
                flush(&mut buf, cur_start, cur_end, &cur_speaker, &mut out);
                cur_start = s.start;
                cur_end = s.end;
                cur_speaker = s.speaker.clone();
            }
        }
        if !buf.is_empty() { buf.push(' '); }
        buf.push_str(text);

        let ends_sentence = text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
            || text.ends_with(".\")") || text.ends_with("!\")") || text.ends_with("?\")");

        if ends_sentence {
            flush(&mut buf, cur_start, cur_end, &cur_speaker, &mut out);
            // reset cur speaker tracking implicitly on next iter
        }
    }
    if !buf.trim().is_empty() {
        flush(&mut buf, cur_start, cur_end, &cur_speaker, &mut out);
    }
    out
}