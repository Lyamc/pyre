//! Reasoning effort: per-request thinking-token budgets for thinking models (Qwen3 `<think>` blocks).
//!
//! Levels: none (empty think block prefilled, no thinking), low (PYRE_THINK_LOW, default 512 thinking tokens),
//! medium (PYRE_THINK_MED, 2048), high (PYRE_THINK_HIGH, 8192), very_high (unlimited = the old behaviour).
//! A request without any effort field gets PYRE_REASONING_DEFAULT (default very_high, i.e. unchanged).
//! When the budget runs out inside the think block the engine stops ("think_budget"), the server appends
//! PYRE_THINK_CLOSE (default "\n\nOkay, time to answer.\n</think>\n\n") to the sequence and continues generating
//! the answer: the continuation prompt is prompt + generated ids + close ids, so it strictly extends the cached
//! state (prefix cache / recurrent state reuse stays valid) and goes through the normal decode (spec/MTP, stops, tools).
//!
//! Request fields (first match wins): `chat_template_kwargs.enable_thinking: false` -> none;
//! `reasoning_effort` (OpenAI: none/minimal/low/medium/high/xhigh, plus very_high/max);
//! `reasoning: {effort | enabled: false}` (OpenRouter style). `thinking_token_budget: N` (pi's budget field)
//! overrides the level's budget. Unknown names (e.g. "auto") fall back to the default.
//! pi (openai-completions, default compat) sends `reasoning_effort` = its level (minimal/low/medium/high/xhigh)
//! and nothing for "off" unless the model's thinkingLevelMap.off is set (use "none").
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Effort { None, Low, Medium, High, VeryHigh }

pub const ALL: [Effort; 5] = [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::VeryHigh];
pub static REQS: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub static FORCED: AtomicU64 = AtomicU64::new(0);
pub static THINK_TOKENS: AtomicU64 = AtomicU64::new(0);

impl Effort {
    /// off/none -> none, minimal/low -> low, medium -> medium, high -> high, xhigh/very_high/max -> very_high.
    pub fn parse(s: &str) -> Option<Effort> {
        let k = s.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        Some(match k.as_str() {
            "none" | "off" | "disabled" | "disable" | "false" | "no" => Effort::None,
            "minimal" | "min" | "low" => Effort::Low,
            "medium" | "med" | "normal" => Effort::Medium,
            "high" => Effort::High,
            "xhigh" | "x_high" | "very_high" | "veryhigh" | "max" | "maximum" | "unlimited" | "full" => Effort::VeryHigh,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self { Effort::None => "none", Effort::Low => "low", Effort::Medium => "medium", Effort::High => "high", Effort::VeryHigh => "very_high" }
    }
    pub fn idx(self) -> usize { ALL.iter().position(|e| *e == self).unwrap() }
    /// Thinking-token budget: Some(0) = no thinking, None = unlimited.
    pub fn budget(self) -> Option<usize> {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d);
        match self {
            Effort::None => Some(0),
            Effort::Low => Some(env("PYRE_THINK_LOW", 512)),
            Effort::Medium => Some(env("PYRE_THINK_MED", 2048)),
            Effort::High => Some(env("PYRE_THINK_HIGH", 8192)),
            Effort::VeryHigh => None,
        }
    }
}

pub fn default_effort() -> Effort {
    std::env::var("PYRE_REASONING_DEFAULT").ok().and_then(|v| Effort::parse(&v)).unwrap_or(Effort::VeryHigh)
}

/// The resolved plan for one request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan { pub effort: Effort, pub budget: Option<usize>, pub explicit: bool }

/// Resolve from the request fields (see module docs).
pub fn resolve(reasoning_effort: Option<&Value>, reasoning: Option<&Value>, kwargs: Option<&Value>, budget_field: Option<&Value>) -> Plan {
    let from_str = |v: Option<&Value>| v.and_then(|x| x.as_str()).and_then(Effort::parse);
    let mut e = None;
    if kwargs.and_then(|k| k.get("enable_thinking")).and_then(|b| b.as_bool()) == Some(false) { e = Some(Effort::None); }
    if e.is_none() { e = from_str(reasoning_effort); }
    if e.is_none() { if let Some(r) = reasoning {
        if r.get("enabled").and_then(|b| b.as_bool()) == Some(false) { e = Some(Effort::None); }
        else { e = from_str(r.get("effort")).or_else(|| from_str(Some(r))); }
    } }
    let explicit = e.is_some();
    let effort = e.unwrap_or_else(default_effort);
    let mut budget = effort.budget();
    if effort != Effort::None { if let Some(n) = budget_field.and_then(|b| b.as_u64()) { budget = Some(n as usize); } }
    Plan { effort, budget, explicit }
}

/// Text appended when the thinking budget runs out inside the think block.
pub fn close_text() -> String {
    std::env::var("PYRE_THINK_CLOSE").map(|s| s.replace("\\n", "\n")).unwrap_or_else(|_| "\n\nOkay, time to answer.\n</think>\n\n".to_string())
}

/// Engine-side check: stop for a forced close once `n_out` tokens are out and the think block is still open.
pub fn over_budget(budget: Option<usize>, n_out: usize, text: &str) -> bool {
    matches!(budget, Some(b) if b > 0 && n_out >= b && !text.contains("</think>"))
}

pub fn count(p: &Plan) { REQS[p.effort.idx()].fetch_add(1, Relaxed); }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn parse_names() {
        assert_eq!(Effort::parse("off"), Some(Effort::None));
        assert_eq!(Effort::parse("none"), Some(Effort::None));
        assert_eq!(Effort::parse("minimal"), Some(Effort::Low));
        assert_eq!(Effort::parse("Low"), Some(Effort::Low));
        assert_eq!(Effort::parse("medium"), Some(Effort::Medium));
        assert_eq!(Effort::parse("high"), Some(Effort::High));
        assert_eq!(Effort::parse("xhigh"), Some(Effort::VeryHigh));
        assert_eq!(Effort::parse("very high"), Some(Effort::VeryHigh));
        assert_eq!(Effort::parse("very-high"), Some(Effort::VeryHigh));
        assert_eq!(Effort::parse("max"), Some(Effort::VeryHigh));
        assert_eq!(Effort::parse("auto"), None);
        assert_eq!(Effort::parse("bogus"), None);
    }
    #[test]
    fn resolve_fields() {
        if std::env::var("PYRE_REASONING_DEFAULT").is_err() {
            let p = resolve(None, None, None, None);
            assert_eq!((p.effort, p.budget, p.explicit), (Effort::VeryHigh, None, false));
            assert_eq!(resolve(Some(&json!("auto")), None, None, None).effort, Effort::VeryHigh);
        }
        assert_eq!(resolve(Some(&json!("low")), None, None, None).effort, Effort::Low);
        assert_eq!(resolve(None, Some(&json!({"effort": "high"})), None, None).effort, Effort::High);
        assert_eq!(resolve(None, Some(&json!({"enabled": false})), None, None).effort, Effort::None);
        assert_eq!(resolve(Some(&json!("high")), None, Some(&json!({"enable_thinking": false})), None).effort, Effort::None);
        assert_eq!(resolve(Some(&json!("medium")), None, Some(&json!({"enable_thinking": true})), None).effort, Effort::Medium);
        assert_eq!(resolve(Some(&json!("medium")), None, None, Some(&json!(300))).budget, Some(300));
        assert_eq!(resolve(Some(&json!("none")), None, None, Some(&json!(300))).budget, Some(0));
        assert!(resolve(Some(&json!("xhigh")), None, None, None).explicit);
    }
    #[test]
    fn budgets() {
        if std::env::var("PYRE_THINK_LOW").is_err() { assert_eq!(Effort::Low.budget(), Some(512)); }
        if std::env::var("PYRE_THINK_MED").is_err() { assert_eq!(Effort::Medium.budget(), Some(2048)); }
        if std::env::var("PYRE_THINK_HIGH").is_err() { assert_eq!(Effort::High.budget(), Some(8192)); }
        assert_eq!(Effort::None.budget(), Some(0));
        assert_eq!(Effort::VeryHigh.budget(), None);
        assert!(!over_budget(None, 100000, "x"));
        assert!(!over_budget(Some(0), 10, "x"));
        assert!(!over_budget(Some(512), 511, "x"));
        assert!(over_budget(Some(512), 512, "thinking..."));
        assert!(!over_budget(Some(512), 900, "thought</think>answer"));
        assert!(close_text().ends_with("</think>\n\n"));
    }
}
