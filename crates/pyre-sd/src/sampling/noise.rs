//! Ported from cabin (cabin-nodes/src/noise.rs).
//! Seeded Gaussian noise on the host, so a seed gives the same starting
//! latents on every backend.

/// SplitMix64: small, fast, and good enough for sampling noise.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1]: never zero, so `ln` is safe.
    pub fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
}

/// `n` standard normal samples from `seed` (Box-Muller).
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (rng.next_unit(), rng.next_unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(n);
    out
}

