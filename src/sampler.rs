//! Token sampler: repetition/presence/frequency penalties, temperature, top-k, top-p, min-p.
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

pub struct Sampler {
    pub params: SamplingParams,
    rng: StdRng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or_else(rand::random);
        Self { params, rng: StdRng::seed_from_u64(seed) }
    }

    /// Apply llama.cpp-style penalties over the last `repeat_last_n` tokens of `history`.
    pub fn apply_penalties(&self, logits: &mut [f32], history: &[u32]) {
        let p = &self.params;
        if p.repeat_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 {
            return;
        }
        let start = if p.repeat_last_n == 0 { history.len() } else { history.len().saturating_sub(p.repeat_last_n) };
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &t in &history[start..] {
            *counts.entry(t).or_default() += 1;
        }
        for (&tok, &c) in &counts {
            let Some(l) = logits.get_mut(tok as usize) else { continue };
            if p.repeat_penalty != 1.0 {
                if *l > 0.0 { *l /= p.repeat_penalty } else { *l *= p.repeat_penalty }
            }
            *l -= c as f32 * p.frequency_penalty + p.presence_penalty;
        }
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.apply_penalties(logits, history);
        let p = self.params.clone();
        if p.temperature <= 0.0 || p.top_k == 1 {
            return argmax(logits);
        }
        // candidates sorted by logit desc
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        let k = if p.top_k == 0 { cand.len() } else { p.top_k.min(cand.len()) };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        // temperature + softmax
        let max = cand[0].1;
        let mut probs: Vec<f32> = cand.iter().map(|c| ((c.1 - max) / p.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= sum);
        // min-p
        let mut n = probs.len();
        if p.min_p > 0.0 {
            let thr = probs[0] * p.min_p;
            n = probs.iter().take_while(|&&x| x >= thr).count().max(1);
        }
        // top-p
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut m = 0;
            for &x in &probs[..n] {
                acc += x;
                m += 1;
                if acc >= p.top_p { break; }
            }
            n = m.max(1);
        }
        let total: f32 = probs[..n].iter().sum();
        let mut r = self.rng.random::<f32>() * total;
        for i in 0..n {
            r -= probs[i];
            if r <= 0.0 { return cand[i].0; }
        }
        cand[n - 1].0
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] { best = i; }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> { vec![1.0, 3.0, 2.0, 0.5, 2.9] }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, ..Default::default() });
        assert_eq!(s.sample(&mut logits(), &[]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, top_k: 1, ..Default::default() });
        for _ in 0..10 { assert_eq!(s.sample(&mut logits(), &[]), 1); }
    }

    #[test]
    fn seed_is_reproducible() {
        let p = SamplingParams { temperature: 1.5, top_k: 0, top_p: 1.0, min_p: 0.0, seed: Some(42), ..Default::default() };
        let a: Vec<u32> = { let mut s = Sampler::new(p.clone()); (0..50).map(|_| s.sample(&mut logits(), &[])).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(p); (0..50).map(|_| s.sample(&mut logits(), &[])).collect() };
        assert_eq!(a, b);
        assert!(a.iter().any(|&t| t != 1), "high temperature should explore");
    }

    #[test]
    fn top_p_restricts_candidates() {
        // probs heavily on token 1 -> top_p tiny means only token 1
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.01, min_p: 0.0, seed: Some(1), ..Default::default() });
        for _ in 0..20 { assert_eq!(s.sample(&mut logits(), &[]), 1); }
    }

    #[test]
    fn min_p_filters() {
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.95, seed: Some(3), ..Default::default() });
        for _ in 0..20 { let t = s.sample(&mut logits(), &[]); assert!(t == 1 || t == 4); }
    }

    #[test]
    fn repeat_penalty_changes_greedy_choice() {
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, repeat_penalty: 1.5, ..Default::default() });
        // token 1 (3.0) penalized -> 2.0, so token 4 (2.9) wins
        assert_eq!(s.sample(&mut logits(), &[1]), 4);
    }

    #[test]
    fn presence_and_frequency_penalty() {
        let s = Sampler::new(SamplingParams { presence_penalty: 0.5, frequency_penalty: 0.25, ..Default::default() });
        let mut l = logits();
        s.apply_penalties(&mut l, &[2, 2]);
        assert!((l[2] - (2.0 - 0.5 - 0.5)).abs() < 1e-6);
        assert_eq!(l[0], 1.0);
    }
}
