use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub diarization: Option<String>,
    pub location: Option<f64>,
}

pub fn get_output_path(input: &Path, output_arg: &Option<PathBuf>, format: &str) -> PathBuf {
    if let Some(arg) = output_arg {
        return arg.clone();
    }
    let mut path = input.to_path_buf();
    let ext = format;
    path.set_extension(ext);
    path
}

pub fn format_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = (seconds % 60.0).floor() as u32;
    let millis = ((seconds % 1.0) * 1000.0).floor() as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

pub fn format_srt(segments: &[TranscriptSegment]) -> String {
    segments.iter().enumerate().map(|(i, s)| {
        format!("{}\n{} --> {}\n{}\n\n", i + 1, format_timestamp(s.start), format_timestamp(s.end), s.text)
    }).collect()
}

pub fn parse_time(time_str: &str) -> anyhow::Result<f64> {
    if let Ok(seconds) = time_str.parse::<f64>() {
        return Ok(seconds);
    }
    let parts: Vec<&str> = time_str.split(':').collect();
    match parts.len() {
        1 => {
            let s = parts[0];
            if s.ends_with('s') { Ok(s[..s.len()-1].parse::<f64>()?) }
            else if s.ends_with('m') { Ok(s[..s.len()-1].parse::<f64>()? * 60.0) }
            else if s.ends_with('h') { Ok(s[..s.len()-1].parse::<f64>()? * 3600.0) }
            else { Ok(s.parse::<f64>()?) }
        },
        2 => {
            let m = parts[0].parse::<f64>()?;
            let s = parts[1].parse::<f64>()?;
            Ok(m * 60.0 + s)
        },
        3 => {
            let h = parts[0].parse::<f64>()?;
            let m = parts[1].parse::<f64>()?;
            let s = parts[2].parse::<f64>()?;
            Ok(h * 3600.0 + m * 60.0 + s)
        },
        _ => anyhow::bail!("Invalid time format: {}", time_str),
    }
}

pub fn trim_trailing_hallucination(text: &str) -> String {
    let mut out = text.to_string();
    for marker in [
        " [BLANK_AUDIO]",
        " [BLANK_AUD",
        " [BEEP]",
        " [Music]",
        " [SILENCE]",
        " [NOISE]",
    ] {
        if let Some(idx) = out.find(marker) {
            out.truncate(idx);
            break;
        }
    }

    if let Some(pos) = out.rfind(". ") {
        let tail = out[pos + 2..].trim();
        if !tail.is_empty() && tail.split_whitespace().count() <= 2 && tail.len() < 24 {
            out.truncate(pos + 1);
        }
    }

    out.trim().to_string()
}

pub fn clean_text(text: &str) -> String {
    let patterns = [
        "[silence]", "[static]", "[music]", "[noise]", "[BEEP]", "[beep]",
        "(silence)", "(static)", "(music)", "(noise)", "(BEEP)", "(beep)",
        "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"
    ];
    let mut cleaned = text.to_string();
    for p in patterns {
        let mut start_search = 0;
        while let Some(pos) = cleaned[start_search..].to_lowercase().find(&p.to_lowercase()) {
            let actual_pos = start_search + pos;
            cleaned.replace_range(actual_pos..actual_pos + p.len(), "");
            start_search = actual_pos;
        }
    }
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Append `next` after dropping a shared word prefix that the previous window already emitted.
///
/// Long audio is cut with a 3s overlap. Concatenating each window's full text
/// repeats that overlap (Whisper base inserted 161 words on a 6-minute clip).
/// Drop tokens emitted in the left-context overlap of a later window.
///
/// The previous window already covers that audio. Keeping both copies inserted
/// the overlap (Nemotron's extra words). Matching words in the string and
/// deleting them also ate real words when the boundary text did not line up.
/// Frame timestamps collapsed into the overlap when a long window keeps less than
/// half of its tokens and at least a dozen were thrown away. v2 did this on one
/// 60 s window of the 6-minute clip (151 reference words, all timestamped in the
/// left context and then deleted). The word stitch is the fallback for that window.
pub fn timestamps_collapsed(kept: &[usize], all: &[usize]) -> bool {
    all.len() >= 12 && kept.len() * 2 < all.len()
}

/// Split ids at `min_frame`. The left side is the overlap the previous window
/// already saw. Callers word-dedupe that side instead of deleting it, so a
/// boundary phrase the previous window missed is kept.
pub fn split_at_frame(ids: &[usize], frames: &[usize], min_frame: usize) -> (Vec<usize>, Vec<usize>) {
    if frames.len() != ids.len() || min_frame == 0 {
        return (Vec::new(), ids.to_vec());
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (id, frame) in ids.iter().zip(frames) {
        if *frame < min_frame {
            left.push(*id);
        } else {
            right.push(*id);
        }
    }
    (left, right)
}

/// Split Whisper tokens on the last timestamp before each token.
/// `min_t` is the overlap in seconds. Tokens with no timestamp in the list
/// come back as `(ids, empty)` so the caller can use the plain word stitch.
pub fn split_at_time(ids: &[usize], times: &[Option<f64>], min_t: f64) -> Option<(Vec<usize>, Vec<usize>)> {
    if times.len() != ids.len() || !times.iter().any(|t| t.is_some()) || min_t <= 0.0 {
        return None;
    }
    let mut t = 0.0;
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (id, stamp) in ids.iter().zip(times) {
        if let Some(ts) = stamp {
            t = *ts;
            continue;
        }
        if t < min_t {
            left.push(*id);
        } else {
            right.push(*id);
        }
    }
    Some((left, right))
}

pub fn drop_left_context_tokens(ids: &[usize], frames: &[usize], min_frame: usize) -> Vec<usize> {
    if frames.len() != ids.len() || min_frame == 0 {
        return ids.to_vec();
    }
    ids.iter()
        .zip(frames)
        .filter(|(_, frame)| **frame >= min_frame)
        .map(|(id, _)| *id)
        .collect()
}

pub fn append_without_word_overlap(accumulated: &mut String, next: &str) {
    let next = next.trim();
    if next.is_empty() {
        return;
    }
    if accumulated.is_empty() {
        accumulated.push_str(next);
        return;
    }
    let acc_words: Vec<&str> = accumulated.split_whitespace().collect();
    let next_words: Vec<&str> = next.split_whitespace().collect();
    let max = acc_words.len().min(next_words.len()).min(40);
    let mut overlap = 0usize;
    for n in (1..=max).rev() {
        let same = acc_words[acc_words.len() - n..]
            .iter()
            .zip(&next_words[..n])
            .all(|(a, b)| a.eq_ignore_ascii_case(b));
        if same {
            overlap = n;
            break;
        }
    }
    let rest = next_words[overlap..].join(" ");
    if rest.is_empty() {
        return;
    }
    if !accumulated.ends_with(' ') {
        accumulated.push(' ');
    }
    accumulated.push_str(&rest);
}

pub fn find_chunk_overlap(prev_tokens: &[usize], curr_tokens: &[usize], max_n_offsets: usize, min_n_overlaps: usize) -> Option<(usize, usize)> {
    let mut max_overlap = 0;
    let mut max_overlap_indices = (0, 0);
    let n_offsets = prev_tokens.len().min(curr_tokens.len()).min(max_n_offsets);
    for offset in 0..n_offsets {
        let prev_start_index = prev_tokens.len() - 1 - offset;
        let mut overlap_iter = prev_tokens.iter().skip(prev_start_index).zip(curr_tokens.iter()).enumerate().filter(|&(_, (&old, &new))| old == new);
        let n_overlap = overlap_iter.clone().count();
        if n_overlap > max_overlap {
            max_overlap = n_overlap;
            if let Some((curr_overlap_index, _)) = overlap_iter.next() {
                let prev_overlap_index = prev_start_index + curr_overlap_index;
                max_overlap_indices = (prev_overlap_index, curr_overlap_index)
            }
        }
    }
    if max_overlap >= min_n_overlaps { Some(max_overlap_indices) } else { None }
}

#[cfg(test)]
mod tests {
    use super::{append_without_word_overlap, drop_left_context_tokens, split_at_frame, split_at_time, timestamps_collapsed, trim_trailing_hallucination};

    #[test]
    fn later_window_drops_only_the_left_context_frames() {
        let ids = [1, 2, 3, 4];
        let frames = [1, 4, 9, 12];
        assert_eq!(drop_left_context_tokens(&ids, &frames, 8), vec![3, 4]);
        assert_eq!(drop_left_context_tokens(&ids, &frames, 0), vec![1, 2, 3, 4]);
        assert!(timestamps_collapsed(&[1], &[0; 20]));
        assert!(!timestamps_collapsed(&[1, 2, 3, 4, 5, 6, 7, 8], &[0; 12]));
        let (left, right) = split_at_frame(&[1, 2, 3], &[1, 9, 12], 8);
        assert_eq!(left, vec![1]);
        assert_eq!(right, vec![2, 3]);
        let times = [Some(0.0), None, None, Some(4.0), None];
        let (left, right) = split_at_time(&[9, 1, 2, 8, 3], &times, 3.0).unwrap();
        assert_eq!(left, vec![1, 2]);
        assert_eq!(right, vec![3]);
    }

    #[test]
    fn drops_words_repeated_at_the_window_overlap() {
        let mut text = String::from("ask not what your country can do for you");
        append_without_word_overlap(&mut text, "can do for you ask what you can do for your country");
        assert_eq!(
            text,
            "ask not what your country can do for you ask what you can do for your country"
        );
        append_without_word_overlap(&mut text, "for your country");
        assert_eq!(
            text,
            "ask not what your country can do for you ask what you can do for your country"
        );
    }

    #[test]
    fn trims_short_suffix_after_period() {
        let input = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country. Godfrey.";
        let out = trim_trailing_hallucination(input);
        assert!(!out.contains("Godfrey"));
        assert!(out.ends_with("country."));
    }
}
