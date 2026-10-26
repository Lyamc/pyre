/// HF `_get_feat_extract_output_lengths` for mel frame counts (chunks or full utterance).
pub fn cnn_output_length(mel_frames: usize) -> usize {
    audio_token_count(mel_frames)
}

/// HF `_get_feat_extract_output_lengths` for Qwen3-ASR mel frames.
pub fn audio_token_count(mel_frames: usize) -> usize {
    if mel_frames == 0 {
        return 0;
    }
    let input_lengths_leave = mel_frames % 100;
    let feat_lengths = floor_div(input_lengths_leave as i64 - 1, 2) + 1;
    let inner = floor_div(feat_lengths - 1, 2) + 1 - 1;
    let chunk_tokens = floor_div(inner, 2) + 1;
    (mel_frames / 100) * 13 + chunk_tokens as usize
}

fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        d - 1
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hundred_frames_yields_thirteen_tokens() {
        assert_eq!(audio_token_count(100), 13);
        assert_eq!(audio_token_count(200), 26);
    }
}