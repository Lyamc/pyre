use super::super::vad::VadSegment;

pub fn energy_segments(samples: &[f32], sample_rate: usize) -> Vec<VadSegment> {
    let hop = (sample_rate as f64 * 0.02).round() as usize; // 20 ms
    let hop = hop.max(80);
    let mut rms = Vec::new();
    let mut i = 0usize;
    while i < samples.len() {
        let n = (samples.len() - i).min(hop);
        let e = (samples[i..i + n].iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        rms.push(e);
        i += hop;
    }
    if rms.is_empty() {
        return Vec::new();
    }
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let thr = (median * 2.5).max(0.01);
    let probs: Vec<f32> = rms
        .iter()
        .map(|&e| if e >= thr { 0.9 } else { 0.1 })
        .collect();
    let hop_secs = hop as f64 / sample_rate.max(1) as f64;
    crate::model::vad::probs_to_segments(
        &probs,
        hop_secs,
        crate::model::vad::VadSettings {
            onset: 0.5,
            offset: 0.35,
            min_speech: 0.30,
            min_silence: 0.20,
            speech_pad: 0.05,
        },
    )
}
