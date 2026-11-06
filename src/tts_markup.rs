//! Speech direction markup for TTS/STS.
//!
//! Supported syntax (documented in the UI help popover too):
//!
//! **SSML-compatible (preferred for timing/rate)**
//! - `<break time="500ms"/>` / `<break time="0.5s"/>` / `<break time="1s"/>`
//! - `<break/>` — default 500 ms pause
//! - `<prosody rate="slow|medium|fast|x-slow|x-fast|N%">...</prosody>`
//!
//! **Bracket events as MMS approximations.** These are not Chatterbox-Nano GPT-2
//! ids. Nano's trained tags live in `pyre-voice-clone` (`[laugh]`, `[clear throat]`, …)
//! and are unused until T3 speaks. Here the built-in VITS cannot emit non-speech
//! sounds, so a few bracket names become short silences and are not spoken aloud:
//! - `[breath]`, `[laughing]` / `[laughs]`, `[sigh]` / `[sighs]`, `[Uhm]`, `[Shh]`
//!
//! **Bracket emotions** like `[frustrated]` are accepted as *style hints*. On the built-in
//! MMS engine they adjust speaking rate/noise where a clear mapping exists, otherwise they
//! are stripped and listed as ignored. They are never spoken as words. Engines with native
//! instruct/style (future Pocket TTS voice prompts, Qwen3-TTS-style instruct) can consume
//! them as free-form style without inventing a parallel tag language.
//!
//! Parenthetical VoxCPM style at the start of the text, e.g. `(cheerful, slightly faster)`,
//! is preserved as a style instruct for engines that support it; MMS ignores it (listed).

use regex::Regex;
use serde::Serialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub enum Seg {
    /// Speak this text at the given speed multiplier (1.0 = normal).
    Speak { text: String, speed: f64, noise: f64, style: Option<String> },
    /// Insert silence of this many seconds.
    Silence { seconds: f64 },
}

#[derive(Debug, Clone, Serialize)]
pub struct ParseResult {
    pub segments: Vec<SegJson>,
    pub ignored: Vec<Ignored>,
    pub style_instruct: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SegJson {
    Speak { text: String, speed: f64, noise: f64, style: Option<String> },
    Silence { seconds: f64 },
}

impl From<&Seg> for SegJson {
    fn from(s: &Seg) -> Self {
        match s {
            Seg::Speak { text, speed, noise, style } => SegJson::Speak {
                text: text.clone(), speed: *speed, noise: *noise, style: style.clone(),
            },
            Seg::Silence { seconds } => SegJson::Silence { seconds: *seconds },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Ignored {
    pub tag: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportedTag {
    pub syntax: String,
    pub effect: String,
    pub engines: Vec<&'static str>,
}

pub fn supported_tags() -> Vec<SupportedTag> {
    vec![
        SupportedTag { syntax: r#"<break time="500ms"/>"#.into(), effect: "Insert a timed silence (ms or s).".into(), engines: vec!["mms", "all"] },
        SupportedTag { syntax: "<break/>".into(), effect: "Insert a 500 ms silence.".into(), engines: vec!["mms", "all"] },
        SupportedTag { syntax: r#"<prosody rate="slow">...</prosody>"#.into(), effect: "Change speaking rate for the wrapped text (slow/medium/fast/x-slow/x-fast or N%).".into(), engines: vec!["mms", "all"] },
        SupportedTag { syntax: "[breath]".into(), effect: "Natural breath/pause (silence approximation on MMS).".into(), engines: vec!["voxcpm", "mms-approx"] },
        SupportedTag { syntax: "[laughing] / [laughs]".into(), effect: "Laughter (silence approx on MMS; not spoken as words).".into(), engines: vec!["voxcpm", "mms-approx"] },
        SupportedTag { syntax: "[sigh] / [sighs]".into(), effect: "Sigh (silence approx on MMS).".into(), engines: vec!["voxcpm", "mms-approx"] },
        SupportedTag { syntax: "[Uhm] / [Shh]".into(), effect: "Filler / hush (silence approx on MMS).".into(), engines: vec!["voxcpm", "mms-approx"] },
        SupportedTag { syntax: "(cheerful, slightly faster) at start".into(), effect: "VoxCPM-style control instruction for the whole utterance; ignored on MMS.".into(), engines: vec!["voxcpm"] },
        SupportedTag { syntax: "[emotion] e.g. [frustrated]".into(), effect: "Style hint. Mapped to rate/noise on MMS when possible; otherwise stripped and listed as ignored.".into(), engines: vec!["mms-partial", "instruct-engines"] },
    ]
}

fn re_break() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"(?i)<break\s*(?:time\s*=\s*["']([^"']+)["'])?\s*/?>"#).unwrap())
}
fn re_prosody_open() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"(?i)<prosody\s+rate\s*=\s*["']([^"']+)["']\s*>"#).unwrap())
}
fn re_prosody_close() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)</prosody>").unwrap())
}
fn re_bracket() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\[([^\]]+)\]").unwrap())
}
fn re_style_paren() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?s)^\s*\(([^)]{1,200})\)\s*").unwrap())
}

fn parse_time(s: &str) -> Option<f64> {
    let t = s.trim().to_ascii_lowercase();
    if let Some(ms) = t.strip_suffix("ms") {
        return ms.trim().parse::<f64>().ok().map(|v| v / 1000.0);
    }
    if let Some(sec) = t.strip_suffix('s') {
        return sec.trim().parse::<f64>().ok();
    }
    t.parse::<f64>().ok()
}

fn rate_to_speed(rate: &str) -> Option<f64> {
    let r = rate.trim().to_ascii_lowercase();
    match r.as_str() {
        "x-slow" => Some(0.5),
        "slow" => Some(0.75),
        "medium" | "default" => Some(1.0),
        "fast" => Some(1.25),
        "x-fast" => Some(1.5),
        _ => {
            if let Some(p) = r.strip_suffix('%') {
                p.parse::<f64>().ok().map(|v| (v / 100.0).clamp(0.25, 3.0))
            } else {
                r.parse::<f64>().ok().map(|v| v.clamp(0.25, 3.0))
            }
        }
    }
}

/// Nonverbal VoxCPM tags → silence seconds on engines that cannot emit them.
fn nonverbal_silence(tag: &str) -> Option<f64> {
    // MMS silence stand-ins. Names overlap some Nano tags; the ids are not emitted.
    match tag.trim().to_ascii_lowercase().as_str() {
        "breath" | "clear throat" => Some(0.35),
        "laughing" | "laughs" | "laughter" | "laugh" | "chuckle" => Some(0.45),
        "sigh" | "sighs" | "sighing" => Some(0.4),
        "uhm" | "uh" | "um" => Some(0.25),
        "shh" | "shhh" | "shush" => Some(0.3),
        "cough" | "groan" | "sniff" | "gasp" => Some(0.35),
        "pause" => Some(0.5),
        _ => None,
    }
}

/// Map a bracket emotion to (speed, noise) when MMS can approximate it; None = ignore.
fn emotion_map(tag: &str) -> Option<(f64, f64)> {
    // MMS rate/noise stand-ins for a few delivery words. Not Nano token ids.
    match tag.trim().to_ascii_lowercase().as_str() {
        "slow" | "calm" | "sad" | "tired" | "whisper" | "whispers" | "whispering"
        | "crying" | "fear" => Some((0.75, 0.8)),
        "fast" | "excited" | "angry" | "frustrated" | "hurried" | "dramatic" => Some((1.25, 1.15)),
        "confused" | "hesitant" | "surprised" => Some((0.9, 1.05)),
        "happy" | "cheerful" | "sarcastic" | "advertisement" | "narration" => Some((1.1, 1.0)),
        _ => None,
    }
}

#[derive(Clone)]
struct Prosody { speed: f64, noise: f64, style: Option<String> }

/// Parse `input` into speak/silence segments. `base_speed`/`base_noise` are the caller's defaults.
pub fn parse(input: &str, base_speed: f64, base_noise: f64) -> (Vec<Seg>, ParseResult) {
    let mut ignored = Vec::new();
    let mut style_instruct = None;
    let mut text = input.to_string();

    // Leading (style instruct) — VoxCPM controllable-clone convention.
    if let Some(c) = re_style_paren().captures(&text) {
        let s = c[1].trim().to_string();
        if !s.is_empty() {
            style_instruct = Some(s.clone());
            ignored.push(Ignored {
                tag: format!("({s})"),
                reason: "Leading parenthetical style instruct is native to VoxCPM; built-in MMS ignores it.".into(),
            });
            text = re_style_paren().replace(&text, "").into_owned();
        }
    }

    // Tokenize by walking breaks / prosody / brackets.
    #[derive(Clone)]
    enum Tok {
        Text(String),
        Break(f64),
        ProsodyOpen(f64),
        ProsodyClose,
        Nonverbal(f64, String),
        Emotion(String, Option<(f64, f64)>),
    }

    let mut toks: Vec<Tok> = Vec::new();
    let mut rest = text.as_str();
    while !rest.is_empty() {
        // Find earliest special match.
        let mut next: Option<(usize, usize, Tok)> = None; // start, end, tok
        let consider = |next: &mut Option<(usize, usize, Tok)>, start: usize, end: usize, tok: Tok| {
            if next.as_ref().map(|(s, _, _)| start < *s).unwrap_or(true) {
                *next = Some((start, end, tok));
            }
        };

        if let Some(m) = re_break().find(rest) {
            let caps = re_break().captures(m.as_str()).unwrap();
            let secs = caps.get(1).and_then(|t| parse_time(t.as_str())).unwrap_or(0.5).clamp(0.05, 10.0);
            consider(&mut next, m.start(), m.end(), Tok::Break(secs));
        }
        if let Some(m) = re_prosody_open().find(rest) {
            let caps = re_prosody_open().captures(m.as_str()).unwrap();
            let speed = rate_to_speed(&caps[1]).unwrap_or(1.0);
            consider(&mut next, m.start(), m.end(), Tok::ProsodyOpen(speed));
        }
        if let Some(m) = re_prosody_close().find(rest) {
            consider(&mut next, m.start(), m.end(), Tok::ProsodyClose);
        }
        if let Some(m) = re_bracket().find(rest) {
            let caps = re_bracket().captures(m.as_str()).unwrap();
            let inner = caps[1].to_string();
            if let Some(secs) = nonverbal_silence(&inner) {
                consider(&mut next, m.start(), m.end(), Tok::Nonverbal(secs, format!("[{inner}]")));
            } else {
                let mapped = emotion_map(&inner);
                consider(&mut next, m.start(), m.end(), Tok::Emotion(inner, mapped));
            }
        }

        match next {
            None => {
                toks.push(Tok::Text(rest.to_string()));
                break;
            }
            Some((s, e, tok)) => {
                if s > 0 { toks.push(Tok::Text(rest[..s].to_string())); }
                toks.push(tok);
                rest = &rest[e..];
            }
        }
    }

    let mut segs: Vec<Seg> = Vec::new();
    let mut stack: Vec<Prosody> = vec![Prosody { speed: base_speed, noise: base_noise, style: None }];
    let mut pending_style: Option<String> = None;

    let push_speak = |segs: &mut Vec<Seg>, text: &str, p: &Prosody, pending_style: &mut Option<String>| {
        let t = text.trim();
        if t.is_empty() { return; }
        // Collapse whitespace a bit but keep punctuation.
        let cleaned = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.is_empty() { return; }
        let style = pending_style.take().or_else(|| p.style.clone());
        if let Some(Seg::Speak { text: prev, speed, noise, style: st }) = segs.last_mut() {
            if (*speed - p.speed).abs() < 1e-9 && (*noise - p.noise).abs() < 1e-9 && *st == style {
                prev.push(' ');
                prev.push_str(&cleaned);
                return;
            }
        }
        segs.push(Seg::Speak { text: cleaned, speed: p.speed, noise: p.noise, style });
    };

    for tok in toks {
        let cur = stack.last().cloned().unwrap_or(Prosody { speed: base_speed, noise: base_noise, style: None });
        match tok {
            Tok::Text(t) => push_speak(&mut segs, &t, &cur, &mut pending_style),
            Tok::Break(secs) | Tok::Nonverbal(secs, _) => {
                segs.push(Seg::Silence { seconds: secs });
            }
            Tok::ProsodyOpen(speed) => {
                stack.push(Prosody { speed: base_speed * speed, noise: cur.noise, style: cur.style.clone() });
            }
            Tok::ProsodyClose => {
                if stack.len() > 1 { stack.pop(); }
            }
            Tok::Emotion(name, mapped) => {
                match mapped {
                    Some((sp, nz)) => {
                        pending_style = Some(name.clone());
                        // Apply as a temporary override for the next speak chunk via pending;
                        // also bump current stack top for subsequent text until prosody changes.
                        if let Some(top) = stack.last_mut() {
                            top.speed = base_speed * sp;
                            top.noise = base_noise * nz;
                            top.style = Some(name.clone());
                        }
                    }
                    None => {
                        ignored.push(Ignored {
                            tag: format!("[{name}]"),
                            reason: "No MMS rate/noise mapping; stripped so it is not spoken. Use <prosody> or a cloning-capable engine for richer style.".into(),
                        });
                    }
                }
            }
        }
    }

    // If style_instruct was set and MMS can't use it, it's already in ignored.
    let meta = ParseResult {
        segments: segs.iter().map(SegJson::from).collect(),
        ignored,
        style_instruct,
    };
    (segs, meta)
}

/// Concatenate PCM segments and silences at `rate` Hz.
pub fn render_pcm(segs: &[Seg], rate: u32, synth: &mut dyn FnMut(&str, f64, f64) -> anyhow::Result<Vec<f32>>) -> anyhow::Result<(Vec<f32>, ParseResult)> {
    // meta is filled by caller via parse; this helper only renders.
    let mut out = Vec::new();
    for s in segs {
        match s {
            Seg::Speak { text, speed, noise, .. } => {
                let mut piece = synth(text, *speed, *noise)?;
                out.append(&mut piece);
            }
            Seg::Silence { seconds } => {
                let n = ((*seconds) * rate as f64).round().max(1.0) as usize;
                out.extend(std::iter::repeat(0.0f32).take(n));
            }
        }
    }
    Ok((out, ParseResult { segments: segs.iter().map(SegJson::from).collect(), ignored: vec![], style_instruct: None }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn break_inserts_silence_segment() {
        let (segs, meta) = parse("Hello <break time=\"500ms\"/> world", 1.0, 1.0);
        assert!(meta.ignored.is_empty());
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Seg::Speak { text, .. } if text == "Hello"));
        assert!(matches!(&segs[1], Seg::Silence { seconds } if (*seconds - 0.5).abs() < 1e-6));
        assert!(matches!(&segs[2], Seg::Speak { text, .. } if text == "world"));
    }

    #[test]
    fn unsupported_emotion_not_in_speak_text() {
        let (segs, meta) = parse("[mystical] Hello there", 1.0, 1.0);
        assert!(meta.ignored.iter().any(|i| i.tag == "[mystical]"));
        let spoken: String = segs.iter().filter_map(|s| match s {
            Seg::Speak { text, .. } => Some(text.as_str()),
            _ => None,
        }).collect::<Vec<_>>().join(" ");
        assert!(!spoken.to_ascii_lowercase().contains("mystical"));
        assert!(spoken.contains("Hello"));
    }

    #[test]
    fn prosody_rate_slow() {
        let (segs, _) = parse(r#"<prosody rate="slow">take your time</prosody>"#, 1.0, 1.0);
        assert!(matches!(&segs[0], Seg::Speak { speed, .. } if (*speed - 0.75).abs() < 1e-6));
    }

    #[test]
    fn breath_becomes_silence() {
        let (segs, _) = parse("one [breath] two", 1.0, 1.0);
        assert!(segs.iter().any(|s| matches!(s, Seg::Silence { .. })));
    }

    #[test]
    fn pause_longer_than_without() {
        let (with_p, _) = parse("Hi <break time=\"1s\"/> there", 1.0, 1.0);
        let (no_p, _) = parse("Hi there", 1.0, 1.0);
        let sil: f64 = with_p.iter().map(|s| match s { Seg::Silence { seconds } => *seconds, _ => 0.0 }).sum();
        assert!(sil >= 1.0);
        let sil0: f64 = no_p.iter().map(|s| match s { Seg::Silence { seconds } => *seconds, _ => 0.0 }).sum();
        assert!(sil0 < 0.01);
    }
}

