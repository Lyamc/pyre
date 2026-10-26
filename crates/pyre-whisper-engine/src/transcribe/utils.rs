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
    use super::trim_trailing_hallucination;

    #[test]
    fn trims_short_suffix_after_period() {
        let input = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country. Godfrey.";
        let out = trim_trailing_hallucination(input);
        assert!(!out.contains("Godfrey"));
        assert!(out.ends_with("country."));
    }
}
