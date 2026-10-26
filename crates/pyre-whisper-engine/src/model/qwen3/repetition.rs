/// HF `detect_and_fix_repetitions` from `qwen_asr/inference/utils.py`.
pub fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let text = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&text, threshold, 20)
}

fn fix_char_repeats(s: &str, thresh: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut res = String::new();
    let mut i = 0usize;
    let n = chars.len();
    while i < n {
        let mut count = 1usize;
        while i + count < n && chars[i + count] == chars[i] {
            count += 1;
        }
        if count > thresh {
            res.push(chars[i]);
        } else {
            for c in &chars[i..i + count] {
                res.push(*c);
            }
        }
        i += count;
    }
    res
}

fn fix_pattern_repeats(s: &str, thresh: usize, max_len: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let min_repeat_chars = thresh * 2;
    if n < min_repeat_chars {
        return s.to_string();
    }

    let mut i = 0usize;
    let mut result = String::new();
    let mut found = false;

    while i <= n.saturating_sub(min_repeat_chars) {
        let mut local_found = false;
        'outer: for k in 1..=max_len {
            if i + k * thresh > n {
                break;
            }
            let pattern: Vec<char> = chars[i..i + k].to_vec();
            let mut valid = true;
            for rep in 1..thresh {
                let start_idx = i + rep * k;
                if chars[start_idx..start_idx + k] != pattern[..] {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            let mut end_index = i + thresh * k;
            while end_index + k <= n && chars[end_index..end_index + k] == pattern[..] {
                end_index += k;
            }
            for c in &pattern {
                result.push(*c);
            }
            let tail: String = chars[end_index..].iter().collect();
            result.push_str(&fix_pattern_repeats(&tail, thresh, max_len));
            i = n;
            local_found = true;
            found = true;
            break 'outer;
        }

        if local_found {
            break;
        }
        result.push(chars[i]);
        i += 1;
    }

    if !found {
        let tail: String = chars[i..].iter().collect();
        result.push_str(&tail);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_long_char_runs() {
        let input = "a".repeat(25);
        let out = detect_and_fix_repetitions(&input, 20);
        assert_eq!(out, "a");
    }

    #[test]
    fn collapses_repeated_word_pattern() {
        let input = "hello ".repeat(25);
        let out = detect_and_fix_repetitions(&input, 20);
        assert_eq!(out, "hello ");
    }

    #[test]
    fn leaves_short_repeats() {
        let input = "hello hello world";
        let out = detect_and_fix_repetitions(input, 20);
        assert_eq!(out, input);
    }

    #[test]
    fn collapses_tail_after_asr_tag() {
        // HF strips before fix; trailing whitespace removal can leave a short tail repeat.
        let input = format!("language English<asr_text>{}", "hi ".repeat(25));
        let out = detect_and_fix_repetitions(input.trim(), 20);
        let tail = out.split("<asr_text>").nth(1).unwrap_or("");
        assert_eq!(tail, "hi hi");
    }
}