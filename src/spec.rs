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
static DRAFT_MIN_P: AtomicU32 = AtomicU32::new(u32::MAX);
/// Later-draft confidence gate set by the high-acceptance mode for this round (None = the drafter default)
pub fn draft_min_p() -> Option<f32> { let v = DRAFT_MIN_P.load(Relaxed); if v == u32::MAX { None } else { Some(f32::from_bits(v)) } }
// with PYRE_QN_NATIVE=2 (the VRAM-light profile) long drafts default on: 0.5 / up to 6 drafts, verified greedy-identical
fn native2() -> bool { crate::model::qn_cube::native2() }
fn hi_acc() -> f32 { std::env::var("PYRE_SPEC_HI").ok().and_then(|v| v.parse().ok()).unwrap_or(if native2() { 0.5 } else { 0.0 }) }
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
/// Draft length set explicitly (set_draft_len or PYRE_SPEC); None = let the model pick its default.
pub fn draft_len_explicit() -> Option<usize> {
    let v = DRAFT.load(Relaxed);
    if v != usize::MAX { return Some(v); }
    std::env::var("PYRE_SPEC").ok().and_then(|s| s.parse().ok())
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
    let p = std::env::var("PYRE_ROUTE_PRIOR").ok().and_then(|s| s.parse().ok()).unwrap_or(if fast() { 1.5 } else { 0.0f32 }).max(0.0);
    ROUTE_PRIOR.store(p.to_bits(), Relaxed); p
}
/// Lossy speed shortcuts in one switch (ON by default; PYRE_QN_FAST=0 or `serve --fast false` turns them off, and then
/// outputs are exactly the exact path's). Reimplemented from Strata (github.com/Niko1221/Strata, MIT): the route prior
/// 1.5 (STRATA_ROUTE_PRIOR, residency-biased routing) and the route tail skip at rank 7 (STRATA_ROUTE_TAIL_SKIP, Strata's
/// CUDA default). Explicit PYRE_ROUTE_PRIOR / PYRE_QN_TAIL_SKIP values win. Drafting stays verified (lossless).
static FAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub fn fast() -> bool { match FAST.load(Relaxed) { 0 => false, 1 => true, _ => std::env::var("PYRE_QN_FAST").map(|v| v != "0").unwrap_or(true) } }
pub fn set_fast(on: bool) { FAST.store(on as u8, Relaxed); }
pub fn set_route_prior(p: f32) { ROUTE_PRIOR.store(p.max(0.0).to_bits(), Relaxed); }

// ---- statistics for /api/status ----
pub static ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static LOOKUPS: AtomicU64 = AtomicU64::new(0);
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
        "fast": fast(),
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
    // PYRE_SPEC_TIME: per-phase wall time (draft, verify by window, keep, plain steps), printed at the end
    let (mut tm_d, mut tm_k, mut tm_s, mut ns_) = (0f64, 0f64, 0f64, 0usize); let (mut pos_r, mut pos_a) = ([0usize; 8], [0usize; 8]); let mut tm_v = [(0f64, 0usize); 10];
    loop {
        history.push(pending); produced += 1;
        if !emit(pending) || produced >= max_new { break; }
        let left = max_new - produced; // tokens we may still emit
        // PYRE_SPEC_HI=acc (opt-in, e.g. 0.95): while the recent acceptance is at least acc, drafts may run to
        // PYRE_SPEC_HI_K (4) with the later-draft confidence gate at PYRE_SPEC_HI_P (0.6); otherwise the normal k and gate
        // the policy also runs for sampled requests (its own statistics): the MTP drafts are the drafter's argmax (a point-mass q),
        // and the window only depends on the drafter's confidences, never on a target sample, so min(1, p/q) acceptance with the
        // residual resample stays distribution-correct. PYRE_SPEC_POLICY_SAMPLED=0 restores the greedy-only policy.
        let pol = policy_on() && (greedy || env("PYRE_SPEC_POLICY_SAMPLED", 1) == 1); PMODE.with(|m| m.set(!greedy as usize)); pol_clear();
        let hi = !pol && hi_acc() > 0.0 && p_acc >= hi_acc() as f64;
        let k = if pol { env("PYRE_SPEC_POLICY_K", 3).max(1) } else if hi { env("PYRE_SPEC_HI_K", if native2() { 6 } else { 4 }).max(k) } else { k };
        DRAFT_MIN_P.store(if hi { std::env::var("PYRE_SPEC_HI_P").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.6).to_bits() } else { u32::MAX }, Relaxed);
        let kk = k.min(left.saturating_sub(1)).min(8).min(target.max_window().saturating_sub(1));
        let mut drafts: Vec<(u32, Option<Vec<(u32, f32)>>)> = vec![];
        if kk > 0 && target.can_verify() {
            // PYRE_SPEC_LOOKUP=n (opt-in): when the last n+ tokens already occurred earlier in the context, draft their
            // continuation (up to PYRE_SPEC_LOOKUP_K tokens) instead of running the MTP drafter: lookup drafting mixed
            // with MTP as in Strata (src/spec/draft_policy.cpp, MIT); verification keeps the output identical
            let lmin = env("PYRE_SPEC_LOOKUP", 0);
            if lmin > 0 { let kl = env("PYRE_SPEC_LOOKUP_K", 3).min(left.saturating_sub(1)).min(8).min(target.max_window().saturating_sub(1));
                let d = ngram_draft(history, kl, lmin, 32); if d.len() >= 2 { drafts = d.into_iter().map(|t| (t, None)).collect(); LOOKUPS.fetch_add(1, Relaxed); } }
            if pol { pol_begin(kk + 1, ts, &tv); }
            let td = std::time::Instant::now();
            if drafts.is_empty() && mtp_on() { if let Some(d) = target.draft(history, kk, if greedy { None } else { Some(&mut *sampler) }) { drafts = d; } }
            tm_d += td.elapsed().as_secs_f64();
            if pol { pol_draft_time(td.elapsed().as_secs_f64()); }
            if drafts.is_empty() && ng { drafts = ngram_draft(history, kk, ng_min, ng_max).into_iter().map(|t| (t, None)).collect(); }
        }
        if let Some(br) = if pol && !drafts.is_empty() { pol_rows() } else { None } {
            if br <= 1 { if skipped >= 24 { skipped = 0; drafts.truncate(2); } else { skipped += 1; drafts.clear(); } } else { skipped = 0; drafts.truncate(br - 1); }
        } else if adapt && !drafts.is_empty() && ts > 0.0 {
            let est = |t: usize| if tv[t] > 0.0 { tv[t] } else { ts * (1.0 + 0.75 * (t as f64 - 1.0)) };
            let (mut best, mut bk) = (1.0f64, 0usize); let mut e = 1.0; let mut pp = 1.0;
            for kq in 1..=drafts.len() { pp *= p_acc; e += pp; let g = e * ts / est(kq + 1); if g > best { best = g; bk = kq; } }
            if bk == 0 && skipped >= 24 { bk = drafts.len().min(2); } // probe now and then so the estimates can recover
            if bk == 0 { skipped += 1; drafts.clear(); } else { skipped = 0; drafts.truncate(bk); }
        }
        if drafts.is_empty() {
            let t0 = std::time::Instant::now();
            logits = target.step(pending);
            let dt = t0.elapsed().as_secs_f64(); tm_s += dt; ns_ += 1; if policy_on() { pol_vcost(1, dt); } ts = if ts == 0.0 { dt } else { 0.8 * ts + 0.2 * dt };
            if trace_on() { let am = crate::sampler::argmax(&logits); eprintln!("spectrace: at {produced} step {} -> argmax {am} logit {:.3}", history.last().copied().unwrap_or(0), logits[am as usize]); }
            pending = sampler.sample(&mut logits, history);
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut toks = vec![pending]; toks.extend(drafts.iter().map(|d| d.0));
        let ls = target.verify(&toks);
        { let dt = t0.elapsed().as_secs_f64(); let t = toks.len(); tm_v[t].0 += dt; tm_v[t].1 += 1; tv[t] = if tv[t] == 0.0 { dt } else { 0.7 * tv[t] + 0.3 * dt }; if pol { pol_vcost(t, dt); } }
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
        let tk = std::time::Instant::now(); target.keep(toks.len(), acc + 1); tm_k += tk.elapsed().as_secs_f64();
        p_acc = 0.7 * p_acc + 0.3 * (acc as f64 / drafts.len() as f64);
        if pol { pol_observe(drafts.len(), acc); }
        for i in 0..drafts.len().min(8) { if i <= acc { pos_r[i] += 1; } if i < acc { pos_a[i] += 1; } }
        out.rounds += 1; out.drafted += drafts.len(); out.accepted += acc;
        ROUNDS.fetch_add(1, Relaxed); DRAFTED.fetch_add(drafts.len() as u64, Relaxed); ACCEPTED.fetch_add(acc as u64, Relaxed);
        if trace_on() { let top = |l: &[f32]| { let mut v: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect(); v.sort_by(|a, b| b.1.total_cmp(&a.1)); v.truncate(3); v }; eprintln!("spectrace: at {produced} window {toks:?} acc {acc} next {nx} tops {:?}", ls.iter().map(|l| top(l)).collect::<Vec<_>>()); }
        if std::env::var("PYRE_SPEC_DEBUG").is_ok() { eprintln!("spec: at {produced} drafted {} accepted {acc} {:?}", drafts.len(), drafts.iter().map(|d| d.0).collect::<Vec<_>>()); }
        let mut stop = false;
        for (dt, _) in &drafts[..acc] { history.push(*dt); produced += 1; if !emit(*dt) || produced >= max_new { stop = true; break; } }
        if stop { break; }
        pending = nx;
    }
    if std::env::var("PYRE_SPEC_TIME").is_ok() { eprint!("spectime: draft {:.1} ms keep {:.1} ms plain {} x {:.2} ms |", tm_d * 1e3, tm_k * 1e3, ns_, if ns_ > 0 { tm_s * 1e3 / ns_ as f64 } else { 0.0 }); for (t, &(s, n)) in tm_v.iter().enumerate() { if n > 0 { eprint!(" verify{t} {n} x {:.2} ms", s * 1e3 / n as f64); } } eprintln!(" | rounds {} lookup rounds {} draft/round {:.2} ms", out.rounds, LOOKUPS.load(Relaxed), if out.rounds > 0 { tm_d * 1e3 / out.rounds as f64 } else { 0.0 }); }
    if std::env::var("PYRE_SPEC_TIME").is_ok() { eprintln!("specpos: (reached, accepted) per draft position {:?} | tokens/window {:.2} | policy {}", (0..8).filter(|&i| pos_r[i] > 0).map(|i| (pos_r[i], pos_a[i])).collect::<Vec<_>>(), if out.rounds > 0 { (out.accepted + out.rounds) as f64 / out.rounds as f64 } else { 0.0 }, pol_text()); }
    LAST_DRAFT.store(k, Relaxed);
    LAST_ACC.store((if out.drafted > 0 { out.accepted as f32 / out.drafted as f32 } else { 0.0 }).to_bits(), Relaxed);
    out
}


// ---- PYRE_SPEC_POLICY=1 (default on under PYRE_QN_XEXACT=1, =0 off; opt-in elsewhere; greedy only): cost-aware MTP window, after Strata's DraftPolicy::keep_draft /
// observe_probs (include/strata/spec/draft_policy.hpp, github.com/Niko1221/Strata, MIT; reimplemented, no code copied).
// After draft j (drafter probability p_j): E_j = 1 + q_0 + .. + q_j, q_i = prod_{k<=i} a(p_k), a() = the measured
// acceptance of drafts that came with that probability (8 buckets, decayed counts); the window is the width with the best
// E / cost, cost(rows) = the measured verify time of that width + the measured drafter time for rows-1 drafts; drafting
// stops once one or two more drafts could no longer beat the best (so a low-probability draft step is not even computed).
// PYRE_SPEC_POLICY_K (default 3) caps the drafts (the XEXACT verify path also caps the window via PYRE_QN_XWMAX).
// It only chooses which drafts are verified: greedy output is unchanged.
pub fn policy_on() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_SPEC_POLICY").map(|v| v == "1").unwrap_or_else(|_| std::env::var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false))) }
const PB: usize = 8;
struct Pol { ok: [f64; PB], n: [f64; PB], dcost: [f64; 10], cost: [f64; 10], cap: usize, q: f64, e: f64, best_v: f64, best_rows: usize,
    probs: Vec<f32>, on: bool, stop: bool, hist: [usize; 10], vc: [f64; 10], vn: [f64; 10], last: [usize; 10], rounds: usize, nv: usize, nd: usize }
thread_local! { static POL: std::cell::RefCell<Vec<Pol>> = std::cell::RefCell::new(vec![Pol::new(), Pol::new()]); static PMODE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
fn pm() -> usize { PMODE.with(|m| m.get()) }
impl Pol {
    fn new() -> Self { let (mut ok, mut n) = ([0.0; PB], [0.0; PB]); for b in 0..PB { n[b] = 2.0; ok[b] = 2.0 * (b as f64 + 0.5) / PB as f64; }
        Pol { ok, n, dcost: [0.0; 10], cost: [0.0; 10], cap: 1, q: 1.0, e: 1.0, best_v: 0.0, best_rows: 1, probs: vec![], on: false, stop: false, hist: [0; 10], vc: [0.0; 10], vn: [0.0; 10], last: [0; 10], rounds: 0, nv: 0, nd: 0 } }
    fn bucket(p: f32) -> usize { ((p.clamp(0.0, 1.0) * PB as f32) as usize).min(PB - 1) }
    fn rate(&self, p: f32) -> f64 { let b = Self::bucket(p); self.ok[b] / self.n[b].max(1e-9) }
    /// drafter time for d drafts: measured, else the nearest measured count scaled by a step of ~4% of a plain step
    fn dc(&self, d: usize) -> f64 { if d == 0 { return 0.0; } if self.dcost[d] > 0.0 { return self.dcost[d]; }
        let step = (1..10).filter(|&i| i + 1 < 10 && self.dcost[i] > 0.0 && self.dcost[i + 1] > 0.0).map(|i| self.dcost[i + 1] - self.dcost[i]).next().filter(|&x| x > 0.0).unwrap_or(0.04 * self.cost[1]);
        match (1..10).filter(|&i| self.dcost[i] > 0.0).min_by_key(|&i| (i as isize - d as isize).abs()) { Some(i) => (self.dcost[i] + step * (d as f64 - i as f64)).max(step), None => step * d as f64 } }
    /// verify cost per width: the robust measured value, else extrapolated from the measured widths (a width step of
    /// the nearest measured pair, else 25% of the narrowest measured window per row)
    fn extrap(&self) -> [f64; 10] { let mut c = [0.0; 10]; let m: Vec<usize> = (1..10).filter(|&t| self.vn[t] > 0.0).collect(); if m.is_empty() { return c; }
        let step = m.windows(2).map(|w| (self.vc[w[1]] - self.vc[w[0]]) / (w[1] - w[0]) as f64).filter(|&x| x > 0.0).last().unwrap_or(0.2 * self.vc[m[0]]);
        for t in 1..10 { c[t] = if self.vn[t] > 0.0 { self.vc[t] } else { let i = *m.iter().min_by_key(|&&i| (i as isize - t as isize).abs()).unwrap(); (self.vc[i] + step * (t as f64 - i as f64)).max(0.5 * self.vc[m[0]]) }; }
        // a narrower window never costs more than a wider one (a stale first-use sample must not price a width out)
        for t in (1..9).rev() { if c[t + 1] > 0.0 { c[t] = c[t].min(c[t + 1]); } } for t in 2..10 { c[t] = c[t].max(c[t - 1]); } c }
    fn c(&self, rows: usize) -> f64 { self.cost[rows.min(9)] + self.dc(rows - 1) }
}
fn pol_clear() { POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; s.probs.clear(); s.stop = false; }) }
fn pol_begin(cap: usize, ts: f64, tv: &[f64; 10]) {
    POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; s.cap = cap.clamp(1, 9); s.probs.clear(); s.stop = false; s.q = 1.0; s.e = 1.0;
        let _ = (ts, tv); s.rounds += 1;
        // a width not measured for 24 rounds is forgotten and priced from the fresher ones again (Strata re-probes stale sizes too)
        for t in 2..10 { if s.vn[t] > 0.0 && s.rounds - s.last[t] > 24 { s.vn[t] = 0.0; } }
        let cost = s.extrap(); s.cost = cost;
        s.on = s.cost[1] > 0.0 && (2..10).any(|t| s.vn[t] > 0.0);
        s.best_rows = 1; s.best_v = if s.on { 1.0 / s.cost[1] } else { 0.0 }; })
}
// robust samples (as Strata's STRATA_SPEC_COST robust mode): a time far above what is known for that width (a first use
// that captures a graph, a stalled round) counts as a tenth of a sample, so one capture cannot price a width out of use
fn robust(v: &mut f64, n: &mut f64, exp: f64, x: f64) { let w = if exp > 0.0 && x > 1.6 * exp { 0.1 } else { 1.0 };
    if *n == 0.0 && w == 1.0 { *v = x; } else { let base = if *n == 0.0 { exp } else { *v }; let a = 0.3 * w; *v = (1.0 - a) * base + a * x; } *n += w; }
fn pol_draft_time(secs: f64) { POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; let d = s.probs.len().min(9); if d == 0 { return; } s.nd += 1; if s.nd <= 2 { return; } let exp = if s.dcost[d] > 0.0 || s.cost[1] > 0.0 { s.dc(d) } else { 0.0 };
    let (mut v, mut n) = (s.dcost[d], if s.dcost[d] > 0.0 { 1.0 } else { 0.0 }); robust(&mut v, &mut n, exp, secs); s.dcost[d] = v; }) }
fn pol_vcost(t: usize, secs: f64) { POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; let t = t.min(9); s.nv += 1; if s.nv <= 2 { return; } s.last[t] = s.rounds; let exp = s.extrap()[t]; let (mut v, mut n) = (s.vc[t], s.vn[t]); robust(&mut v, &mut n, exp, secs); s.vc[t] = v; s.vn[t] = n; }) }
fn pol_rows() -> Option<usize> { POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; if s.probs.is_empty() { return None; } let r = s.best_rows.min(s.probs.len() + 1); s.hist[r.min(9)] += 1; Some(r) }) }
fn pol_observe(verified: usize, acc: usize) { POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()];
    for i in 0..verified.min(s.probs.len()) { if i > acc { break; } let b = Pol::bucket(s.probs[i]); s.n[b] = 0.97 * s.n[b] + 1.0; s.ok[b] = 0.97 * s.ok[b] + if i < acc { 1.0 } else { 0.0 }; } }) }
fn pol_text() -> String { if !policy_on() { return "off".into(); } POL.with(|c| { let g = c.borrow(); let s = &g[pm()];
    format!("rows {:?} rate/bucket {:?} cost ms {:?} draft ms {:?}", &s.hist[1..6], (0..PB).map(|b| (s.ok[b] / s.n[b].max(1e-9) * 100.0).round() / 100.0).collect::<Vec<_>>(), (1..6).map(|t| (s.cost[t] * 1e4).round() / 10.0).collect::<Vec<_>>(), (1..5).map(|d| (s.dcost[d] * 1e4).round() / 10.0).collect::<Vec<_>>()) }) }
/// Drafter hook, called after draft `j` came with probability `p`: whether to keep it (false = drop it and stop).
/// Without PYRE_SPEC_POLICY this is the old gate (a later draft needs p >= min_p).
pub fn draft_gate(j: usize, p: f32, min_p: f32) -> bool {
    if !policy_on() { return !(j > 0 && p < min_p); }
    POL.with(|c| { let mut g = c.borrow_mut(); let s = &mut g[pm()]; s.probs.push(p); let rows = j + 2;
        if !s.on { s.best_rows = rows; s.stop = rows >= s.cap || p < 0.9; return true; }
        let a = s.rate(p); s.q *= a; s.e += s.q; let v = s.e / s.c(rows);
        if v > s.best_v { s.best_v = v; s.best_rows = rows; }
        let mut more = false;
        if rows < s.cap { let am = s.rate(1.0); let (q1, e1) = (s.q * am, s.e + s.q * am);
            if e1 / s.c(rows + 1) > s.best_v { more = true; } else if rows + 1 < s.cap && (e1 + q1 * am) / s.c(rows + 2) > s.best_v { more = true; } }
        s.stop = !more; true })
}
/// Drafter hook before computing draft `j` > 0: the policy decided to stop after the previous draft.
pub fn draft_stop() -> bool { policy_on() && POL.with(|c| c.borrow()[pm()].stop) }

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

/// PYRE_SPEC_TRACE=1 (diagnostic, off by default): per-round window, accepted count and top-3 logits of every verify position.
pub fn trace_on() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_SPEC_TRACE").map(|v| v == "1").unwrap_or(false)) }
