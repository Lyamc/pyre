#[derive(Clone, Debug, PartialEq)]
pub struct VadSegment {
    pub start: f64,
    pub end: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct VadSettings {
    pub speech_pad: f64,
    pub min_speech: f64,
    pub min_silence: f64,
    pub onset: f32,
    pub offset: f32,
}

impl VadSettings {
    pub fn silero() -> Self {
        Self {
            onset: 0.5,
            offset: 0.35,
            min_speech: 0.25,
            min_silence: 0.10,
            speech_pad: 0.03,
        }
    }

    pub fn ten() -> Self {
        Self {
            onset: 0.5,
            offset: 0.35,
            min_speech: 0.25,
            min_silence: 0.10,
            speech_pad: 0.02,
        }
    }
}

pub fn probs_to_segments(probs: &[f32], hop_secs: f64, cfg: VadSettings) -> Vec<VadSegment> {
    if probs.is_empty() {
        return Vec::new();
    }
    let mut segs = Vec::new();
    let mut start: Option<usize> = None;
    let mut last_speech = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        let speaking = match start {
            None => p >= cfg.onset,
            Some(_) => p >= cfg.offset,
        };
        if speaking {
            last_speech = i;
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let silence = (i.saturating_sub(last_speech) as f64) * hop_secs;
            if silence >= cfg.min_silence {
                let end = last_speech + 1;
                push_seg(&mut segs, s, end, hop_secs, cfg);
                start = None;
            }
        }
    }
    if let Some(s) = start {
        push_seg(&mut segs, s, probs.len(), hop_secs, cfg);
    }
    segs
}

fn push_seg(out: &mut Vec<VadSegment>, s: usize, e: usize, hop: f64, cfg: VadSettings) {
    let start = (s as f64 * hop - cfg.speech_pad).max(0.0);
    let end = e as f64 * hop + cfg.speech_pad;
    if end - start >= cfg.min_speech {
        if let Some(prev) = out.last_mut() {
            if start <= prev.end + cfg.min_silence {
                prev.end = end.max(prev.end);
                return;
            }
        }
        out.push(VadSegment { start, end });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_short_gaps() {
        let mut probs = vec![0.1; 40];
        for p in &mut probs[5..20] {
            *p = 0.9;
        }
        for p in &mut probs[21..30] {
            *p = 0.9;
        }
        let segs = probs_to_segments(&probs, 0.032, VadSettings::silero());
        assert_eq!(segs.len(), 1);
        assert!(segs[0].start < 0.3);
        assert!(segs[0].end > 0.9);
    }
}
