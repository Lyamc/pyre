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
//! auto (`reasoning_effort: "auto"`): the engine picks the level per request from the model itself (no second model).
//! After prefill it snapshots the state, feeds PYRE_THINK_AUTO_PROMPT (a short self-rating question inside the open think
//! block, ending in "["), reads the model's probabilities of the digits 0..4, and restores the snapshot (the answer never
//! sees the probe; cost = one short forward + snapshot/restore). Signal E = expected rating in [0, 4].
//! Bands PYRE_THINK_AUTO_T="a,b,c,d": E < a none, < b low, < c medium, < d high, else very_high.
//! (Tried first and rejected: entropy of the first predicted think tokens. The first token has p1 ~1.0 on every prompt;
//! the mean entropy of the first 8 tokens was 0.33-0.64 nats with no relation to difficulty: trivial 0.575, hard math 0.391.)
//! Structural bias (server, auto_bias): last message is a tool result -> PYRE_THINK_AUTO_TOOL (default -1 level);
//! last user message longer than PYRE_THINK_AUTO_LONG chars (default 2000) or containing a code fence -> +1.
//! Result clamped to PYRE_THINK_AUTO_MIN..PYRE_THINK_AUTO_MAX (default none..high), with a medium ceiling unless the
//! rating clears the band edge by PYRE_THINK_AUTO_MARGIN (see auto_pick). The chosen level's budget
//! (as above) then applies; "none" closes the think block right away.
//! pi (openai-completions, default compat) sends `reasoning_effort` = its level (minimal/low/medium/high/xhigh)
//! and nothing for "off" unless the model's thinkingLevelMap.off is set (use "none").
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Effort { None, Low, Medium, High, VeryHigh, Auto }

pub const ALL: [Effort; 6] = [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::VeryHigh, Effort::Auto];
pub const LEVELS: [Effort; 5] = [Effort::None, Effort::Low, Effort::Medium, Effort::High, Effort::VeryHigh];
pub static REQS: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// auto's choices by level (LEVELS order)
pub static AUTO_CHOICE: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub static FORCED: AtomicU64 = AtomicU64::new(0);
/// Engine requests that were continuations of a forced think close (subtracted from pyre_requests_total).
pub static CONTS: AtomicU64 = AtomicU64::new(0);
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
            "auto" | "adaptive" => Effort::Auto,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self { Effort::None => "none", Effort::Low => "low", Effort::Medium => "medium", Effort::High => "high", Effort::VeryHigh => "very_high", Effort::Auto => "auto" }
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
            Effort::VeryHigh | Effort::Auto => None,
        }
    }
}

/// Entropy (nats, T=1), top-1 probability and argmax of a logit vector.
pub fn entropy(l: &[f32]) -> (f32, f32, u32) {
    let (mut mx, mut am) = (f32::NEG_INFINITY, 0usize);
    for (i, &x) in l.iter().enumerate() { if x > mx { mx = x; am = i; } }
    let (mut z, mut s) = (0f64, 0f64);
    for &x in l { let d = (x - mx) as f64; let e = d.exp(); z += e; s += e * d; }
    if z <= 0.0 || !z.is_finite() { return (0.0, 1.0, am as u32); }
    ((z.ln() - s / z) as f32, (1.0 / z) as f32, am as u32)
}

fn envf(k: &str) -> Option<String> { std::env::var(k).ok().filter(|s| !s.trim().is_empty()) }
/// Entropy band edges (nats) between none|low|medium|high|very_high.
pub fn auto_thresholds() -> [f32; 4] {
    let d = [AUTO_T_DEFAULT[0], AUTO_T_DEFAULT[1], AUTO_T_DEFAULT[2], AUTO_T_DEFAULT[3]];
    let Some(s) = envf("PYRE_THINK_AUTO_T") else { return d };
    let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if v.len() == 4 { [v[0], v[1], v[2], v[3]] } else { d }
}
/// Calibrated on P62 (int53d, 6 prompts, E): trivial 1.19, one-liner 1.79, bat/ball 2.18, x^2-y^2 2.23, ISO-duration code 2.44,
/// after a tool result 1.37. The model's ratings are compressed (1.2-2.5), so the bands are narrow.
pub const AUTO_T_DEFAULT: [f32; 4] = [0.9, 1.5, 2.3, 2.9];
pub fn auto_prompt() -> String {
    envf("PYRE_THINK_AUTO_PROMPT").map(|s| s.replace("\\n", "\n")).unwrap_or_else(|| "\nFirst, how much thinking does this request need? On a scale from 0 (none: trivial, answer directly) to 4 (very deep: hard multi-step problem), my rating is [".to_string())
}
/// The probe phrasings (PYRE_THINK_AUTO_NPROBE, default 2, are averaged; PYRE_THINK_AUTO_PROMPT2 overrides the second).
pub fn auto_prompts() -> Vec<String> {
    let n = envf("PYRE_THINK_AUTO_NPROBE").and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(2).clamp(1, 2);
    let p2 = envf("PYRE_THINK_AUTO_PROMPT2").map(|s| s.replace("\\n", "\n")).unwrap_or_else(|| "\nLet me first judge the difficulty. 0 = trivial, 1 = easy, 2 = moderate, 3 = hard, 4 = very hard. Difficulty: [".to_string());
    let mut v = vec![auto_prompt()]; if n > 1 { v.push(p2); } v
}
/// Expected rating over the digit tokens (normalised among them) and their total probability mass.
pub fn rating(logits: &[f32], digits: &[u32]) -> (f32, f32) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|x| ((x - mx) as f64).exp()).sum();
    let p: Vec<f64> = digits.iter().map(|&d| logits.get(d as usize).map(|x| ((x - mx) as f64).exp() / z).unwrap_or(0.0)).collect();
    let m: f64 = p.iter().sum();
    if m <= 0.0 { return (2.0, 0.0); }
    ((p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>() / m) as f32, m as f32)
}
/// Level when the probe could not run (no snapshot support): medium shifted by the structural bias.
pub fn auto_fallback(bias: i32) -> Effort { LEVELS[(2 + bias).clamp(0, 4) as usize] }
/// Level from the entropy signal plus a structural bias (in levels), clamped to PYRE_THINK_AUTO_MIN..MAX.
/// Ceiling: medium unless the rating is clearly in the upper bands (>= band edge + PYRE_THINK_AUTO_MARGIN, default 0.4):
/// then high, and very_high only when also allowed by PYRE_THINK_AUTO_MAX (default high). High's 8192-token budget is
/// long (t4 at auto: rating 2.62 -> high -> 8175 thinking tokens, 927 s vs 358 s at medium), so it needs strong evidence;
/// the structural bias never lifts a level above that ceiling.
pub fn auto_pick(h: f32, bias: i32) -> Effort { auto_pick_t(h, bias, false) }
/// `tools`: the request offers tools. none needs a clearly low rating (< band edge - PYRE_THINK_AUTO_NONE_MARGIN, 0.15),
/// and with tools offered a rating below PYRE_THINK_AUTO_TOOLS_NONE (0.5); otherwise the floor is low (P64: a TCP/UDP
/// question rated 0.88 at none answered with a stray tool call).
pub fn auto_pick_t(h: f32, bias: i32, tools: bool) -> Effort {
    let t = auto_thresholds();
    let base = t.iter().filter(|x| h >= **x).count() as i32;
    let lv = |k: &str, d: usize| envf(k).and_then(|v| Effort::parse(&v)).filter(|e| *e != Effort::Auto).map(|e| e.idx()).unwrap_or(d) as i32;
    let margin = envf("PYRE_THINK_AUTO_MARGIN").and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(0.4);
    let strong = if h >= t[3] + margin { 4 } else if h >= t[2] + margin { 3 } else { 2 };
    let (lo, hi) = (lv("PYRE_THINK_AUTO_MIN", 0), lv("PYRE_THINK_AUTO_MAX", 3).min(strong));
    let mut i = (base + bias).clamp(lo.min(hi), hi);
    let fv = |k: &str, d: f32| envf(k).and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(d);
    if i == 0 && (h >= t[0] - fv("PYRE_THINK_AUTO_NONE_MARGIN", 0.15) || (tools && h >= fv("PYRE_THINK_AUTO_TOOLS_NONE", 0.5))) { i = 1.min(hi.max(1)); }
    LEVELS[i as usize]
}
/// Structural cue from the chat messages (see module docs).
pub fn auto_bias(msgs: &[Value]) -> i32 {
    let Some(last) = msgs.last() else { return 0 };
    let iv = |k: &str, d: i64| envf(k).and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(d);
    match last.get("role").and_then(|r| r.as_str()) {
        Some("tool") | Some("function") => iv("PYRE_THINK_AUTO_TOOL", -1) as i32,
        Some("user") => { let c = last.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if c.chars().count() as i64 > iv("PYRE_THINK_AUTO_LONG", 2000) || c.contains("```") { 1 } else { 0 } }
        _ => 0,
    }
}
pub fn count_auto(e: Effort) { if let Some(i) = LEVELS.iter().position(|x| *x == e) { AUTO_CHOICE[i].fetch_add(1, Relaxed); } }

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
        assert_eq!(Effort::parse("auto"), Some(Effort::Auto));
        assert_eq!(Effort::parse("bogus"), None);
    }
    #[test]
    fn resolve_fields() {
        if std::env::var("PYRE_REASONING_DEFAULT").is_err() {
            let p = resolve(None, None, None, None);
            assert_eq!((p.effort, p.budget, p.explicit), (Effort::VeryHigh, None, false));
            assert_eq!(resolve(Some(&json!("bogus")), None, None, None).effort, Effort::VeryHigh);
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
    #[test]
    fn auto_logic() {
        assert_eq!(resolve(Some(&json!("auto")), None, None, None).effort, Effort::Auto);
        assert_eq!(Effort::Auto.budget(), None);
        let (h, p, a) = entropy(&[10.0, 0.0, 0.0]); assert!(h < 0.01 && p > 0.99 && a == 0);
        let (h, p, _) = entropy(&[1.0; 4]); assert!((h - 4f32.ln()).abs() < 1e-4 && (p - 0.25).abs() < 1e-5);
        let (e, m) = rating(&[0.0, 0.0, 0.0, 10.0, 0.0, 0.0], &[1, 2, 3, 4, 5]); assert!((e - 2.0).abs() < 0.01 && m > 0.99);
        let (e, _) = rating(&[0.0; 6], &[1, 2, 3, 4, 5]); assert!((e - 2.0).abs() < 1e-4);
        assert_eq!(auto_fallback(0), Effort::Medium); assert_eq!(auto_fallback(-1), Effort::Low);
        if std::env::var("PYRE_THINK_AUTO_T").is_err() && std::env::var("PYRE_THINK_AUTO_MIN").is_err() && std::env::var("PYRE_THINK_AUTO_MAX").is_err() {
            let t = AUTO_T_DEFAULT;
            assert_eq!(auto_pick(t[0] - 0.01, 0), Effort::Low);       // not clearly low
            assert_eq!(auto_pick(t[0] - 0.2, 0), Effort::None);
            assert_eq!(auto_pick_t(t[0] - 0.2, 0, true), Effort::Low);  // tools offered: floor low
            assert_eq!(auto_pick_t(0.3, 0, true), Effort::None);
            assert_eq!(auto_pick(t[0], 0), Effort::Low);
            assert_eq!(auto_pick(t[1], 0), Effort::Medium);
            assert_eq!(auto_pick(t[2], 0), Effort::Medium);          // in the high band but not clearly: medium
            assert_eq!(auto_pick(t[2] + 0.41, 0), Effort::High);
            assert_eq!(auto_pick(t[3] + 1.0, 0), Effort::High);      // PYRE_THINK_AUTO_MAX default high
            assert_eq!(auto_pick(t[1], 1), Effort::Medium);          // bias cannot pass the medium ceiling
            assert_eq!(auto_pick(t[1], -1), Effort::Low);
            assert_eq!(auto_pick(t[0] + 0.1, -1), Effort::Low);      // tool-result bias to none, rating not clearly low
            assert_eq!(auto_pick(0.0, -1), Effort::None);
        }
        if std::env::var("PYRE_THINK_AUTO_TOOL").is_err() && std::env::var("PYRE_THINK_AUTO_LONG").is_err() {
            assert_eq!(auto_bias(&[json!({"role": "user", "content": "hi"}), json!({"role": "tool", "content": "ok"})]), -1);
            assert_eq!(auto_bias(&[json!({"role": "user", "content": "fix ```x```"})]), 1);
            assert_eq!(auto_bias(&[json!({"role": "user", "content": "a".repeat(2500)})]), 1);
            assert_eq!(auto_bias(&[json!({"role": "user", "content": "what is 2+2"})]), 0);
            assert_eq!(auto_bias(&[]), 0);
        }
    }
}
