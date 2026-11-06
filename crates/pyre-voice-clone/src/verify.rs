//! Clone check on top of L2-normalized d-vectors.
//!
//! `VoiceEncoder.voice_similarity` is the dot product of two L2-normalized
//! embeddings, which is cosine similarity. The checkpoint also stores
//! `similarity_weight` (init 10) and `similarity_bias` (init -5). Those scale the
//! GE2E training loss. Inference does not apply them, and neither does this check.
//!
//! `CLONE_COSINE_FLOOR` is an operating point for "these two clips are the same
//! speaker", not a value the model was trained to threshold. Same-speaker pairs
//! from this encoder usually land well above it; a decision should still read the
//! raw cosine.

use crate::encoder::EMBED;

/// Same-speaker pairs from the GE2E d-vector usually clear this. Different
/// speakers and non-speech often do not. It is a reporting threshold.
pub const CLONE_COSINE_FLOOR: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerMatch {
    pub cosine: f32,
    pub accepts_clone: bool,
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

pub fn match_speakers(reference: &[f32], other: &[f32]) -> SpeakerMatch {
    let cosine = cosine(reference, other);
    SpeakerMatch {
        accepts_clone: cosine >= CLONE_COSINE_FLOOR
            && reference.len() == EMBED
            && other.len() == EMBED,
        cosine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_unit_vectors_accept() {
        let v = vec![1.0 / 16.0; EMBED];
        let m = match_speakers(&v, &v);
        assert!((m.cosine - 1.0).abs() < 1e-5);
        assert!(m.accepts_clone);
    }

    #[test]
    fn opposite_vectors_reject() {
        let a = vec![1.0; EMBED];
        let b = vec![-1.0; EMBED];
        let m = match_speakers(&a, &b);
        assert!(m.cosine < -0.99);
        assert!(!m.accepts_clone);
    }
}
