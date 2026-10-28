//! Ported from cabin (cabin-nodes/src/schedule.rs).
//! Noise and timestep schedule for KSampler. The node calls [`sampler_inputs`]
//! and steps the Burn scheduler with the list this returns.

use anyhow::{bail, Result};
use crate::pipelines::stable_diffusion::StableDiffusionConfig;
use crate::schedulers::euler_discrete::{
    EulerDiscreteScheduler, EulerDiscreteSchedulerConfig,
};
use crate::schedulers::BetaSchedule;

use super::noise::gaussian;

pub const SCHEDULERS: [&str; 6] = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
];

/// What one KSampler call feeds the Burn scheduler: noise and the timestep list.
pub struct SamplerInputs {
    pub noise: Vec<f32>,
    pub shape: [usize; 4],
    pub timesteps: Vec<f64>,
    /// Euler-family sigmas, one longer than `timesteps` (trailing zero).
    pub sigmas: Option<Vec<f64>>,
}

/// Noise shaped like `latent` and the timestep list for `sampler` / `scheduler`.
pub fn sampler_inputs(
    seed: u64,
    steps: usize,
    sampler: &str,
    scheduler: &str,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<SamplerInputs> {
    let steps = steps.max(1);
    let (timesteps, sigmas) = schedule(steps, sampler, scheduler)?;
    let n = channels * height * width;
    Ok(SamplerInputs {
        noise: gaussian(seed, n),
        shape: [1, channels, height, width],
        timesteps,
        sigmas,
    })
}

/// Timesteps (and euler sigmas) the sampler loop consumes.
pub fn schedule(
    steps: usize,
    sampler: &str,
    scheduler: &str,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let steps = steps.max(1);
    match sampler {
        "euler" | "euler_ancestral" => {
            let (timesteps, sigmas) = euler_schedule(steps, scheduler)?;
            Ok((timesteps, Some(sigmas)))
        }
        "heun" | "lms" | "dpm_2" | "dpm_2_ancestral" | "ddpm" | "pndm" => {
            builtin_only(sampler, scheduler)?;
            Ok((vec![0.0; steps], None))
        }
        "ddim" => Ok((ddim_timesteps(steps, ddim_offset(scheduler)?), None)),
        "dpmpp_2m" => {
            match scheduler {
                "normal" | "sgm_uniform" => {}
                "karras" | "exponential" | "simple" | "ddim_uniform" => {
                    bail!("scheduler `{scheduler}` is not implemented for sampler `{sampler}`")
                }
                other => bail!(
                    "unknown scheduler `{other}` (have: {})",
                    SCHEDULERS.join(", ")
                ),
            }
            Ok((
                dpm_timesteps(steps).into_iter().map(|t| t as f64).collect(),
                None,
            ))
        }
        other => bail!("unknown sampler `{other}`"),
    }
}

/// DDIM `steps_offset` for this scheduler name.
pub fn ddim_offset(scheduler: &str) -> Result<usize> {
    match scheduler {
        "normal" => Ok(1),
        "sgm_uniform" => Ok(0),
        "karras" | "exponential" | "simple" | "ddim_uniform" => {
            bail!("scheduler `{scheduler}` is not implemented for sampler `ddim`")
        }
        other => bail!(
            "unknown scheduler `{other}` (have: {})",
            SCHEDULERS.join(", ")
        ),
    }
}

fn euler_config() -> EulerDiscreteSchedulerConfig {
    let sd = StableDiffusionConfig::v1_5(None, None, None);
    EulerDiscreteSchedulerConfig {
        beta_start: sd.beta_start,
        beta_end: sd.beta_end,
        beta_schedule: sd.beta_schedule,
        prediction_type: sd.prediction_type,
        train_timesteps: sd.train_timesteps,
    }
}

fn euler_schedule(steps: usize, scheduler: &str) -> Result<(Vec<f64>, Vec<f64>)> {
    let base = EulerDiscreteScheduler::new(steps, euler_config());
    match scheduler {
        "normal" => Ok((base.timesteps().to_vec(), base.sigmas().to_vec())),
        "karras" | "exponential" | "sgm_uniform" | "simple" | "ddim_uniform" => {
            resample(&base, scheduler, steps)
        }
        other => bail!(
            "unknown scheduler `{other}` (have: {})",
            SCHEDULERS.join(", ")
        ),
    }
}

fn resample(
    base: &EulerDiscreteScheduler,
    scheduler: &str,
    steps: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let full = training_sigmas();
    let sigmas = base.sigmas();
    let sigma_max = sigmas[0];
    let sigma_min = sigmas[sigmas.len() - 2].max(1e-10);
    let spaced = match scheduler {
        "karras" => karras_sigmas(steps, sigma_min, sigma_max),
        "exponential" => exponential_sigmas(steps, sigma_min, sigma_max),
        "sgm_uniform" => {
            let ts = dpm_timesteps(steps);
            let mut sigmas = interp_sigmas(&ts, &full);
            sigmas.push(0.0);
            return Ok((ts.into_iter().map(|t| t as f64).collect(), sigmas));
        }
        "simple" => simple_sigmas(steps, &full),
        "ddim_uniform" => {
            let ts = ddim_timesteps(steps, 1);
            let mut sigmas =
                interp_sigmas(&ts.iter().map(|t| *t as usize).collect::<Vec<_>>(), &full);
            sigmas.push(0.0);
            return Ok((ts, sigmas));
        }
        _ => bail!("unknown scheduler `{scheduler}`"),
    };
    let timesteps = spaced.iter().map(|&s| sigma_to_t(s, &full)).collect();
    let mut sigmas = spaced;
    sigmas.push(0.0);
    Ok((timesteps, sigmas))
}

/// Training sigmas for the SD 1.5 scaled-linear beta schedule (index = timestep).
fn training_sigmas() -> Vec<f64> {
    let cfg = euler_config();
    let betas = match cfg.beta_schedule {
        BetaSchedule::ScaledLinear => {
            let start = cfg.beta_start.sqrt();
            let end = cfg.beta_end.sqrt();
            let step = (end - start) / (cfg.train_timesteps - 1) as f64;
            (0..cfg.train_timesteps)
                .map(|i| {
                    let v = start + step * i as f64;
                    v * v
                })
                .collect::<Vec<_>>()
        }
        BetaSchedule::Linear => {
            let step = (cfg.beta_end - cfg.beta_start) / (cfg.train_timesteps - 1) as f64;
            (0..cfg.train_timesteps)
                .map(|i| cfg.beta_start + step * i as f64)
                .collect()
        }
        BetaSchedule::SquaredcosCapV2 => panic!("SD 1.5 euler does not use squaredcos"),
    };
    let mut cumprod = 1.0;
    betas
        .iter()
        .map(|b| {
            cumprod *= 1.0 - b;
            ((1.0 - cumprod) / cumprod).sqrt()
        })
        .collect()
}

/// ComfyUI `simple`: stride through the training sigmas from the noisiest end.
fn simple_sigmas(steps: usize, full: &[f64]) -> Vec<f64> {
    let n = full.len().max(1);
    let ss = n as f64 / steps.max(1) as f64;
    (0..steps.max(1))
        .map(|x| {
            let back = 1 + (x as f64 * ss) as usize;
            full[n - back.min(n)]
        })
        .collect()
}

fn builtin_only(sampler: &str, scheduler: &str) -> Result<()> {
    match scheduler {
        "normal" | "sgm_uniform" => Ok(()),
        "karras" | "exponential" | "simple" | "ddim_uniform" => {
            bail!("scheduler `{scheduler}` is only implemented for euler and euler_ancestral (sampler is `{sampler}`)")
        }
        other => bail!(
            "unknown scheduler `{other}` (have: {})",
            SCHEDULERS.join(", ")
        ),
    }
}

fn karras_sigmas(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    let rho = 7.0;
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    (0..n)
        .map(|i| {
            let ramp = if n <= 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            (max_inv + ramp * (min_inv - max_inv)).powf(rho)
        })
        .collect()
}

fn exponential_sigmas(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![sigma_max];
    }
    let log_max = sigma_max.ln();
    let log_min = sigma_min.ln();
    (0..n)
        .map(|i| {
            let ramp = i as f64 / (n - 1) as f64;
            (log_max + ramp * (log_min - log_max)).exp()
        })
        .collect()
}

fn sigma_to_t(sigma: f64, sigmas: &[f64]) -> f64 {
    let log_sigma = sigma.max(1e-10).ln();
    let log_sigmas: Vec<f64> = sigmas.iter().map(|s| s.max(1e-10).ln()).collect();
    let mut low = 0usize;
    for (i, &ls) in log_sigmas.iter().enumerate() {
        if log_sigma >= ls {
            low = i;
        }
    }
    let low = low.min(log_sigmas.len().saturating_sub(2));
    let high = low + 1;
    let denom = log_sigmas[low] - log_sigmas[high];
    let w = if denom.abs() < 1e-12 {
        0.0
    } else {
        ((log_sigmas[low] - log_sigma) / denom).clamp(0.0, 1.0)
    };
    (1.0 - w) * low as f64 + w * high as f64
}

fn interp_sigmas(timesteps: &[usize], full: &[f64]) -> Vec<f64> {
    timesteps
        .iter()
        .map(|&t| {
            let t = t.min(full.len() - 1);
            full[t]
        })
        .collect()
}

fn ddim_timesteps(steps: usize, offset: usize) -> Vec<f64> {
    let step_ratio = 1000 / steps.max(1);
    (0..steps.max(1))
        .map(|s| (s * step_ratio + offset) as f64)
        .rev()
        .collect()
}

fn dpm_timesteps(steps: usize) -> Vec<usize> {
    let steps = steps.max(1);
    let step = 999.0 / steps as f64;
    let mut ts: Vec<usize> = (0..=steps)
        .map(|i| (i as f64 * step).round() as usize)
        .skip(1)
        .collect();
    ts.reverse();
    ts
}

/// Rectified-flow sigmas for SD3. `shift` is 3 for SD 3.5 Medium.
/// Timesteps are `sigma * 1000`. The sigma list ends at 0.
pub fn flow_schedule(steps: usize, shift: f64) -> (Vec<f64>, Vec<f64>) {
    let steps = steps.max(1);
    let raw = linspace(1.0, 1.0 / 1000.0, steps);
    let sigmas: Vec<f64> = raw
        .into_iter()
        .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
        .collect();
    let timesteps = sigmas.iter().map(|s| s * 1000.0).collect();
    let mut sigmas = sigmas;
    sigmas.push(0.0);
    (timesteps, sigmas)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
        .collect()
}

