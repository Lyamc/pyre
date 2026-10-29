//! Speculative decoding for single-slot models (qwen-next): drafts are verified in ONE multi-token
//! target pass (`Target::verify`), the longest accepted prefix is kept and the state rolled back past it.
//!
//! Ideas borrowed (not code) from Strata (github.com/Niko1221/Strata): `--spec N` draft length, the
//! model's own MTP layer as drafter, a prompt-lookup / suffix drafter, and probability-preserving
//! acceptance for sampled requests (`STRATA_SPEC_PROB`). Greedy output is identical to plain decode;
//! for temperature > 0 the standard speculative-sampling rule (accept with min(1, p/q), else sample the
//! residual max(p - q, 0)) keeps the target distribution exactly.
use crate::sampler::Sampler;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

// ---- process-wide settings (config file / CLI / env; per-request override via the API) ----
static DRAFT: AtomicUsize = AtomicUsize::new(usize::MAX); // MAX = not set yet -> env/default
static NGRAM: AtomicBool = AtomicBool::new(true);
static MTP: AtomicBool = AtomicBool::new(true);
static ROUTE_PRIOR: AtomicU32 = AtomicU32::new(u32::MAX);

/// Default draft length: PYRE_SPEC, else 0 (off by default until measured faster on this model).
pub fn draft_len() -> usize {
    let v = DRAFT.load(Relaxed);
    if v != usize::MAX { return v; }
    let d = std::env::var("PYRE_SPEC").ok().and_then(|s| s.parse().ok()).unwrap_or(0usize);
    DRAFT.store(d, Relaxed); d
}
pub fn set_draft_len(n: usize) { DRAFT.store(n.min(8), Relaxed); }
pub fn ngram_on() -> bool { NGRAM.load(Relaxed) && std::env::var("PYRE_SPEC_NGRAM").map(|v| v != "0").unwrap_or(true) }
pub fn set_ngram(on: bool) { NGRAM.store(on, Relaxed); }
pub fn mtp_on() -> bool { MTP.load(Relaxed) && std::env::var("PYRE_SPEC_MTP").map(|v| v != "0").unwrap_or(true) }
pub fn set_mtp(on: bool) { MTP.store(on, Relaxed); }
/// Residency-biased routing strength (0 = off, the default). PYRE_ROUTE_PRIOR overrides.
pub fn route_prior() -> f32 {
    let v = ROUTE_PRIOR.load(Relaxed);
    if v != u32::MAX { return f32::from_bits(v); }
    let p = std::env::var("PYRE_ROUTE_PRIOR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0f32).max(0.0);
    ROUTE_PRIOR.store(p.to_bits(), Relaxed); p
}
pub fn set_route_prior(p: f32) { ROUTE_PRIOR.store(p.max(0.0).to_bits(), Relaxed); }

// ---- statistics for /api/status ----
pub static ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static DRAFTED: AtomicU64 = AtomicU64::new(0);
pub static ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static EMITTED: AtomicU64 = AtomicU64::new(0);
pub static CTX_USED: AtomicUsize = AtomicUsize::new(0);
pub static CTX_MAX: AtomicUsize = AtomicUsize::new(0);
pub static LAST_TPS: AtomicU32 = AtomicU32::new(0);
pub static LAST_ACC: AtomicU32 = AtomicU32::new(0);
pub static LAST_DRAFT: AtomicUsize = AtomicUsize::new(0);
pub static DRAFTER: std::sync::Mutex<&'static str> = std::sync::Mutex::new("none");

pub fn status() -> serde_json::Value {
    let (d, a) = (DRAFTED.load(Relaxed), ACCEPTED.load(Relaxed));
    serde_json::json!({
        "speculative": {"enabled": draft_len() > 0, "draft_len": draft_len(), "ngram": ngram_on(), "mtp": mtp_on(),
            "drafter": *DRAFTER.lock().unwrap(), "rounds": ROUNDS.load(Relaxed), "drafted": d, "accepted": a,
            "acceptance": if d > 0 { a as f64 / d as f64 } else { 0.0 }, "tokens": EMITTED.load(Relaxed),
            "last": {"draft_len": LAST_DRAFT.load(Relaxed), "acceptance": f32::from_bits(LAST_ACC.load(Relaxed)), "tok_per_s": f32::from_bits(LAST_TPS.load(Relaxed))}},
        "route_prior": {"enabled": route_prior() > 0.0, "strength": route_prior()},
        "context": {"used": CTX_USED.load(Relaxed), "max": CTX_MAX.load(Relaxed)},
    })
}

/// A model that can verify several tokens in one pass and roll back.
pub trait Target {
    /// Feed one token, logits of the next position.
    fn step(&mut self, tok: u32) -> Vec<f32>;
    /// Feed `toks`; logits after each of them.
    fn verify(&mut self, toks: &[u32]) -> Vec<Vec<f32>>;
    /// Keep the first `keep` of the last `t` fed tokens.
    fn keep(&mut self, t: usize, keep: usize);
    /// Whether `verify` + `keep` are available (else plain decode).
    fn can_verify(&self) -> bool;
    /// Longest window `verify` may take right now while staying identical to plain decode.
    fn max_window(&self) -> usize { usize::MAX }
    /// Model drafter (MTP): up to `k` tokens following `history` (whose last token is not yet fed),
    /// each with its draft distribution when sampling (None = greedy / point mass).
    fn draft(&mut self, _history: &[u32], _k: usize, _sample: Option<&mut Sampler>) -> Option<Vec<(u32, Option<Vec<(u32, f32)>>)>> { None }
}

/// Prompt-lookup drafter: the most recent earlier occurrence of the longest suffix (n = max..=min tokens)
/// of `h`, and the tokens that followed it.
pub fn ngram_draft(h: &[u32], k: usize, min_n: usize, max_n: usize) -> Vec<u32> {
    let len = h.len();
    if k == 0 || len < min_n + 1 { return vec![]; }
    for n in (min_n..=max_n.min(len - 1)).rev() {
        let suf = &h[len - n..];
        // scan backwards over start positions i with h[i..i+n] == suf, i + n < len
        let mut i = len - n;
        while i > 0 {
            i -= 1;
            if h[i + n - 1] == suf[n - 1] && &h[i..i + n] == suf {
                let s = i + n; let e = (s + k).min(len);
                if e > s { return h[s..e].to_vec(); }
            }
        }
    }
    vec![]
}

pub struct Outcome { pub drafted: usize, pub accepted: usize, pub rounds: usize }

/// Generate after `logits` (the next-token distribution following `history`). `emit(tok)` returns false
/// to stop (EOS, stop string, length, client gone). `k` = max draft length (0 = plain decode).
pub fn generate(target: &mut dyn Target, sampler: &mut Sampler, history: &mut Vec<u32>, mut logits: Vec<f32>, k: usize, max_new: usize,
                emit: &mut dyn FnMut(u32) -> bool) -> Outcome {
    let greedy = sampler.params.temperature <= 0.0 || sampler.params.top_k == 1;
    let ng = ngram_on(); let (ng_min, ng_max) = (env("PYRE_SPEC_NGRAM_MIN", 3), env("PYRE_SPEC_NGRAM_MAX", 8));
    let mut out = Outcome { drafted: 0, accepted: 0, rounds: 0 };
    let mut produced = 0usize;
    // adaptive draft length (PYRE_SPEC_ADAPT=0 turns it off): one verify pass of t tokens costs about
    // (1 + alpha (t-1)) plain steps here (CPU experts scale with the distinct experts of the window), so a
    // window only pays when its expected accepted length beats that. Timings and acceptance are tracked online.
    let adapt = std::env::var("PYRE_SPEC_ADAPT").map(|v| v != "0").unwrap_or(true);
    let mut ts = 0f64; let mut tv = [0f64; 10]; let mut p_acc = 0.7f64; let mut skipped = 0usize;
    let mut pending = sampler.sample(&mut logits, history);
    loop {
        history.push(pending); produced += 1;
        if !emit(pending) || produced >= max_new { break; }
        let left = max_new - produced; // tokens we may still emit
        let kk = k.min(left.saturating_sub(1)).min(8).min(target.max_window().saturating_sub(1));
        let mut drafts: Vec<(u32, Option<Vec<(u32, f32)>>)> = vec![];
        if kk > 0 && target.can_verify() {
            if mtp_on() { if let Some(d) = target.draft(history, kk, if greedy { None } else { Some(&mut *sampler) }) { drafts = d; } }
            if drafts.is_empty() && ng { drafts = ngram_draft(history, kk, ng_min, ng_max).into_iter().map(|t| (t, None)).collect(); }
        }
        if adapt && !drafts.is_empty() && ts > 0.0 {
            let est = |t: usize| if tv[t] > 0.0 { tv[t] } else { ts * (1.0 + 0.75 * (t as f64 - 1.0)) };
            let (mut best, mut bk) = (1.0f64, 0usize); let mut e = 1.0; let mut pp = 1.0;
            for kq in 1..=drafts.len() { pp *= p_acc; e += pp; let g = e * ts / est(kq + 1); if g > best { best = g; bk = kq; } }
            if bk == 0 && skipped >= 24 { bk = drafts.len().min(2); } // probe now and then so the estimates can recover
            if bk == 0 { skipped += 1; drafts.clear(); } else { skipped = 0; drafts.truncate(bk); }
        }
        if drafts.is_empty() {
            let t0 = std::time::Instant::now();
            logits = target.step(pending);
            let dt = t0.elapsed().as_secs_f64(); ts = if ts == 0.0 { dt } else { 0.8 * ts + 0.2 * dt };
            pending = sampler.sample(&mut logits, history);
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut toks = vec![pending]; toks.extend(drafts.iter().map(|d| d.0));
        let ls = target.verify(&toks);
        { let dt = t0.elapsed().as_secs_f64(); let t = toks.len(); tv[t] = if tv[t] == 0.0 { dt } else { 0.7 * tv[t] + 0.3 * dt }; }
        // accept the longest prefix
        let mut acc = 0usize; let mut next = None;
        let mut hist = history.clone();
        for (i, (dt, q)) in drafts.iter().enumerate() {
            let mut lg = ls[i].clone();
            if greedy {
                sampler.apply_penalties(&mut lg, &hist);
                let t = crate::sampler::argmax(&lg);
                if t == *dt { acc += 1; hist.push(t); continue; }
                next = Some(t); break;
            }
            let p = sampler.dist(&mut lg, &hist);
            let pd = p.iter().find(|x| x.0 == *dt).map(|x| x.1).unwrap_or(0.0);
            let qd = q.as_ref().map(|q| q.iter().find(|x| x.0 == *dt).map(|x| x.1).unwrap_or(0.0)).unwrap_or(1.0);
            if qd > 0.0 && sampler.uniform() < (pd / qd).min(1.0) { acc += 1; hist.push(*dt); continue; }
            // residual: max(p - q, 0), renormalised
            let mut res: Vec<(u32, f32)> = p.iter().map(|&(t, pv)| { let qv = match q { Some(q) => q.iter().find(|x| x.0 == t).map(|x| x.1).unwrap_or(0.0), None => if t == *dt { 1.0 } else { 0.0 } }; (t, (pv - qv).max(0.0)) }).collect();
            if res.iter().all(|x| x.1 <= 0.0) { res = p.clone(); }
            next = Some(sampler.pick(&res)); break;
        }
        let nx = match next { Some(t) => t, None => { let mut lg = ls[drafts.len()].clone(); sampler.sample(&mut lg, &hist) } };
        target.keep(toks.len(), acc + 1);
        p_acc = 0.7 * p_acc + 0.3 * (acc as f64 / drafts.len() as f64);
        out.rounds += 1; out.drafted += drafts.len(); out.accepted += acc;
        ROUNDS.fetch_add(1, Relaxed); DRAFTED.fetch_add(drafts.len() as u64, Relaxed); ACCEPTED.fetch_add(acc as u64, Relaxed);
        if std::env::var("PYRE_SPEC_DEBUG").is_ok() { eprintln!("spec: at {produced} drafted {} accepted {acc} {:?}", drafts.len(), drafts.iter().map(|d| d.0).collect::<Vec<_>>()); }
        let mut stop = false;
        for (dt, _) in &drafts[..acc] { history.push(*dt); produced += 1; if !emit(*dt) || produced >= max_new { stop = true; break; } }
        if stop { break; }
        pending = nx;
    }
    LAST_DRAFT.store(k, Relaxed);
    LAST_ACC.store((if out.drafted > 0 { out.accepted as f32 / out.drafted as f32 } else { 0.0 }).to_bits(), Relaxed);
    out
}

fn env(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ngram_finds_latest_longest() {
        let h = [1, 2, 3, 9, 1, 2, 3, 4, 5, 7, 1, 2, 3];
        assert_eq!(ngram_draft(&h, 2, 2, 4), vec![4, 5]);
        assert_eq!(ngram_draft(&[1, 2, 3], 2, 2, 4), Vec::<u32>::new());
    }
}
