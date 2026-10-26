/// HF `split_audio_into_chunks` — split at low-energy boundaries, lossless concat.
pub fn split_audio_into_chunks(
    wav: &[f32],
    sample_rate: usize,
    max_chunk_sec: f64,
    search_expand_sec: f64,
    min_window_ms: f64,
    min_chunk_sec: f64,
) -> Vec<(Vec<f32>, f64)> {
    if wav.is_empty() {
        return Vec::new();
    }

    let total_len = wav.len();
    let total_sec = total_len as f64 / sample_rate as f64;
    if total_sec <= max_chunk_sec {
        return vec![(wav.to_vec(), 0.0)];
    }

    let max_len = (max_chunk_sec * sample_rate as f64) as usize;
    let expand = (search_expand_sec * sample_rate as f64) as usize;
    let win = ((min_window_ms / 1000.0) * sample_rate as f64).round() as usize;
    let win = win.max(4);

    let mut chunks: Vec<(Vec<f32>, f64)> = Vec::new();
    let mut start = 0usize;
    let mut offset_sec = 0.0f64;

    while total_len.saturating_sub(start) > max_len {
        let cut = start + max_len;
        let left = cut.saturating_sub(expand).max(start);
        let right = (cut + expand).min(total_len);

        let boundary = if right <= left + win {
            cut
        } else {
            let seg = &wav[left..right];
            let seg_abs: Vec<f32> = seg.iter().map(|x| x.abs()).collect();
            let window_sums: Vec<f32> = (0..seg_abs.len().saturating_sub(win) + 1)
                .map(|i| seg_abs[i..i + win].iter().sum())
                .collect();
            let min_pos = window_sums
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let wstart = min_pos;
            let wend = (min_pos + win).min(seg_abs.len());
            let inner = seg_abs[wstart..wend]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut boundary = left + wstart + inner;
            boundary = boundary.max(start + 1).min(total_len);
            boundary
        };

        let chunk = wav[start..boundary].to_vec();
        chunks.push((chunk, offset_sec));
        offset_sec += (boundary - start) as f64 / sample_rate as f64;
        start = boundary;
    }

    let tail = wav[start..total_len].to_vec();
    chunks.push((tail, offset_sec));

    let min_len = (min_chunk_sec * sample_rate as f64).round() as usize;
    chunks
        .into_iter()
        .map(|(mut c, off)| {
            if c.len() < min_len {
                c.resize(min_len, 0.0);
            }
            (c, off)
        })
        .collect()
}

pub const QWEN3_MAX_ASR_CHUNK_SEC: f64 = 1200.0;
pub const QWEN3_MIN_ASR_CHUNK_SEC: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_single_chunk() {
        let wav = vec![0.1f32; 16_000];
        let chunks = split_audio_into_chunks(&wav, 16_000, 1200.0, 5.0, 100.0, 0.5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), 16_000);
    }

    #[test]
    fn long_audio_splits_losslessly() {
        let wav: Vec<f32> = (0..500_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let chunks = split_audio_into_chunks(&wav, 16_000, 10.0, 0.5, 100.0, 0.5);
        assert!(chunks.len() > 1);
        let reconstructed: Vec<f32> = chunks.into_iter().flat_map(|(c, _)| c).collect();
        assert_eq!(reconstructed.len(), wav.len());
        for (a, b) in reconstructed.iter().zip(wav.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}