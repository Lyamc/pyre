//! Inline vocal directions for Chatterbox-Nano / Turbo.
//!
//! The model does not take a side-channel style vector at generate time. Nano's
//! `generate` ignores `exaggeration` and `cfg_weight`. Directions are ordinary text
//! tokens, written in square brackets, and they have to survive as single ids in the
//! GPT-2 tokenizer (vocab size 50276). The ids below are the `added_tokens.json` map
//! shipped with `ResembleAI/chatterbox-nano` and `ResembleAI/chatterbox-turbo`.
//!
//! Two groups share that file:
//! * Event tags (50267–50275) are the nine sounds Resemble's own UI inserts:
//!   `[clear throat] [sigh] [shush] [cough] [groan] [sniff] [gasp] [chuckle] [laugh]`.
//! * Delivery tags (50257–50266) are also single tokens (`[angry]`, `[whispering]`, …).
//!   Some runtimes report that the model was trained to ignore them. They are still
//!   passed through, unchanged, so a later T3 port can decide.
//!
//! `normalize_script` is the Turbo/Nano `punc_norm` cleanup. It does not strip tags.

/// How a bracket tag is meant to be spoken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionKind {
    /// A non-speech vocal event the Nano UI actually drives.
    Event,
    /// A delivery or emotion token present in the tokenizer.
    Delivery,
}

/// One inline direction and where it sits in the normalized script.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Direction {
    /// Canonical spelling, including brackets. This is the tokenizer's token text.
    pub tag: &'static str,
    pub token_id: u32,
    pub kind: DirectionKind,
    /// Byte range of the tag inside [`Script::normalized`].
    pub start: usize,
    pub end: usize,
}

/// Normalized text plus the directions found in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Script {
    pub normalized: String,
    pub directions: Vec<Direction>,
    /// Bracket spans that are not one of the 19 model tags. Left in the text.
    pub unknown: Vec<String>,
}

struct Tag {
    tag: &'static str,
    token_id: u32,
    kind: DirectionKind,
}

const TAGS: &[Tag] = &[
    Tag {
        tag: "[angry]",
        token_id: 50257,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[fear]",
        token_id: 50258,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[surprised]",
        token_id: 50259,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[whispering]",
        token_id: 50260,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[advertisement]",
        token_id: 50261,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[dramatic]",
        token_id: 50262,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[narration]",
        token_id: 50263,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[crying]",
        token_id: 50264,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[happy]",
        token_id: 50265,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[sarcastic]",
        token_id: 50266,
        kind: DirectionKind::Delivery,
    },
    Tag {
        tag: "[clear throat]",
        token_id: 50267,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[sigh]",
        token_id: 50268,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[shush]",
        token_id: 50269,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[cough]",
        token_id: 50270,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[groan]",
        token_id: 50271,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[sniff]",
        token_id: 50272,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[gasp]",
        token_id: 50273,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[chuckle]",
        token_id: 50274,
        kind: DirectionKind::Event,
    },
    Tag {
        tag: "[laugh]",
        token_id: 50275,
        kind: DirectionKind::Event,
    },
];

/// Turbo/Nano `punc_norm`, then tag split.
///
/// Recognized tags are rewritten to the tokenizer's exact spelling (`[Laugh]`
/// becomes `[laugh]`). GPT-2 added tokens are case-sensitive, so the canonical
/// form is what a later BPE pass has to see.
pub fn parse_script(text: &str) -> Script {
    let source = normalize_script(text);
    let mut normalized = String::with_capacity(source.len());
    let mut directions = Vec::new();
    let mut unknown = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(tag) = match_tag(&source[i..]) {
                let start = normalized.len();
                normalized.push_str(tag.tag);
                directions.push(Direction {
                    tag: tag.tag,
                    token_id: tag.token_id,
                    kind: tag.kind,
                    start,
                    end: normalized.len(),
                });
                i += tag.tag.len();
                continue;
            }
            if let Some(rel) = source[i..].find(']') {
                let span = &source[i..i + rel + 1];
                if span.len() <= 40 && !span.contains('\n') {
                    unknown.push(span.to_string());
                }
            }
        }
        let ch = source[i..].chars().next().unwrap();
        normalized.push(ch);
        i += ch.len_utf8();
    }
    Script {
        normalized,
        directions,
        unknown,
    }
}

/// The 19 bracket tags in tokenizer order of their ids.
pub fn all_tags() -> Vec<(&'static str, u32, DirectionKind)> {
    TAGS.iter().map(|t| (t.tag, t.token_id, t.kind)).collect()
}

pub fn tag_token_id(tag: &str) -> Option<u32> {
    let lower = tag.to_ascii_lowercase();
    TAGS.iter().find(|t| t.tag == lower).map(|t| t.token_id)
}

fn match_tag(rest: &str) -> Option<&'static Tag> {
    let lower = rest.to_ascii_lowercase();
    TAGS.iter()
        .filter(|t| lower.starts_with(t.tag))
        .max_by_key(|t| t.tag.len())
}

/// Chatterbox Turbo/Nano `punc_norm`. Tags are left intact.
pub fn normalize_script(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    let mut text = text.to_string();
    if text.chars().next().is_some_and(|c| c.is_lowercase()) {
        let mut chars = text.chars();
        let first = chars.next().unwrap();
        text = first.to_uppercase().collect::<String>() + chars.as_str();
    }
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (old, new) in [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        text = text.replace(old, new);
    }
    text = text.trim_end_matches(' ').to_string();
    let enders = ['.', '!', '?', '-', ','];
    if !text.chars().last().is_some_and(|c| enders.contains(&c)) {
        text.push('.');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_tag_keeps_its_token_id() {
        let s = parse_script("hi there, Sarah here [chuckle], have you got one minute");
        assert_eq!(s.directions.len(), 1);
        assert_eq!(s.directions[0].tag, "[chuckle]");
        assert_eq!(s.directions[0].token_id, 50274);
        assert_eq!(s.directions[0].kind, DirectionKind::Event);
        assert!(s.unknown.is_empty());
        assert!(s.normalized.contains("[chuckle]"));
        assert!(s.normalized.ends_with('.'));
        assert!(s.normalized.starts_with('H'));
    }

    #[test]
    fn two_word_tag_and_casefold() {
        let s = parse_script("wait [CLEAR THROAT] okay");
        assert_eq!(s.directions.len(), 1);
        assert_eq!(s.directions[0].token_id, 50267);
        assert_eq!(
            &s.normalized[s.directions[0].start..s.directions[0].end],
            "[clear throat]"
        );
    }

    #[test]
    fn unknown_bracket_is_reported_and_kept() {
        let s = parse_script("Hello [whisper] there");
        assert!(s.directions.is_empty());
        assert_eq!(s.unknown, vec!["[whisper]".to_string()]);
        assert!(s.normalized.contains("[whisper]"));
    }

    #[test]
    fn delivery_tag_is_not_dropped() {
        let s = parse_script("[whispering] come closer");
        assert_eq!(s.directions[0].kind, DirectionKind::Delivery);
        assert_eq!(s.directions[0].token_id, 50260);
    }

    #[test]
    fn punctuation_matches_turbo_punc_norm() {
        assert_eq!(normalize_script("hello…world"), "Hello, world.");
        assert_eq!(
            normalize_script(""),
            "You need to add some text for me to talk."
        );
        assert_eq!(normalize_script("  hello"), "hello.");
    }

    #[test]
    fn laugh_does_not_eat_laughing() {
        let s = parse_script("a [laughing] b");
        assert!(s.directions.is_empty());
        assert_eq!(s.unknown, vec!["[laughing]".to_string()]);
    }
}
