//! Text plus a reference clip to a 24 kHz waveform.
//!
//! The path is the native one: voice-encoder embedding into T3, S3 prompt
//! tokens, CAMPPlus into mean-flow, 24 kHz prompt mel, two Euler steps, HiFT.
//! Mean-flow noise stays random. A seed makes the draw repeatable; it does
//! not replace the normal with zeros. HiFT's harmonic source uses the same
//! seed on its own generator.
//!
//! [`crate::text_to_cloned_speech_ready`] stays false. A saved audition wav is
//! not a claim that the voice matches the reference.

use crate::camp::CampPlus;
use crate::encoder::VoiceEncoder;
use crate::flow::{Flow, SplitMix, S3GEN_SIL};
use crate::frontend::prompt_mel_24k;
use crate::hift::Hift;
use crate::s3::{S3Tokenizer, MAX_PROMPT_TOKENS};
use crate::t3::TurboDecode;
use crate::t3_model::{T3Context, T3Nano};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const SPEECH_VOCAB: u32 = 6561;
pub const CLONE_SR: u32 = 24_000;

/// Run `body` against the process-wide stack, loading it on first use.
pub fn with_stack<R>(body: impl FnOnce(&NanoStack) -> Result<R>) -> Result<R> {
    let stack = NanoStack::shared()?;
    body(&stack)
}

fn discover_checkpoints() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let s3 = crate::s3::discover_s3().context(
        "s3gen_meanflow.safetensors not found (PYRE_CHATTERBOX_S3 or PYRE_CHATTERBOX_NANO)",
    )?;
    let t3 = crate::t3_model::discover_t3()
        .context("t3_nano_v1.safetensors not found (PYRE_CHATTERBOX_T3 or PYRE_CHATTERBOX_NANO)")?;
    let ve = discover_ve()
        .context("ve.safetensors not found (PYRE_CHATTERBOX_VE or PYRE_CHATTERBOX_NANO)")?;
    Ok((s3, t3, ve))
}

fn discover_ve() -> Option<PathBuf> {
    for key in ["PYRE_CHATTERBOX_VE", "PYRE_CHATTERBOX_NANO"] {
        let Some(p) = std::env::var_os(key) else {
            continue;
        };
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        let file = p.join("ve.safetensors");
        if file.is_file() {
            return Some(file);
        }
    }
    None
}

pub struct NanoPaths<'a> {
    pub s3: &'a Path,
    pub t3: &'a Path,
    pub ve: &'a Path,
}

pub struct NanoStack {
    s3: S3Tokenizer,
    t3: T3Nano,
    ve: VoiceEncoder,
    flow: Flow,
    hift: Hift,
    camp: CampPlus,
}

pub struct CloneWave {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Prompt tokens after the 2-frames-per-token crop.
    pub prompt_tokens: Vec<u32>,
    /// T3 ids below 6561, then three silence tokens.
    pub speech_tokens: Vec<u32>,
}

impl NanoStack {
    /// First call loads T3, the voice encoder, and the mean-flow file (S3,
    /// CAMPPlus, flow, HiFT). Later calls reuse that stack.
    pub fn shared() -> Result<std::sync::MutexGuard<'static, NanoStack>> {
        static STACK: OnceLock<Mutex<NanoStack>> = OnceLock::new();
        if STACK.get().is_none() {
            let (s3, t3, ve) = discover_checkpoints()?;
            let stack = NanoStack::load(&NanoPaths {
                s3: &s3,
                t3: &t3,
                ve: &ve,
            })?;
            let _ = STACK.set(Mutex::new(stack));
        }
        let mutex = STACK.get().context("nano stack was not stored")?;
        mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("nano stack lock poisoned"))
    }

    pub fn load(paths: &NanoPaths<'_>) -> Result<Self> {
        Ok(Self {
            s3: S3Tokenizer::load(paths.s3)?,
            t3: T3Nano::load(paths.t3)?,
            ve: VoiceEncoder::load(paths.ve)?,
            flow: Flow::load(paths.s3)?,
            hift: Hift::load(paths.s3)?,
            camp: CampPlus::load(paths.s3)?,
        })
    }

    /// `wav_16k` and `wav_24k` are the same utterance. Text ids are GPT-2 ids
    /// from the Nano tokenizer. `max_gen_len` is T3's loop bound after the
    /// first sample.
    pub fn cloned_waveform(
        &self,
        wav_16k: &[f32],
        wav_24k: &[f32],
        text: &[u32],
        max_gen_len: usize,
        seed: u64,
    ) -> Result<CloneWave> {
        if text.is_empty() {
            bail!("text tokens are empty");
        }
        // Upstream crops: T3 prompt tokens from the first 15 s at 16 kHz
        // (ENC_COND_LEN); S3Gen's ref dict (prompt tokens, prompt mel,
        // CAMPPlus) from the first 10 s at 24 kHz (DEC_COND_LEN). The voice
        // encoder sees the whole clip.
        let enc16 = &wav_16k[..wav_16k.len().min(15 * 16_000)];
        let dec24 = &wav_24k[..wav_24k.len().min(10 * CLONE_SR as usize)];
        let dec16 = &wav_16k[..wav_16k.len().min(10 * 16_000)];
        let t3_prompt = self.s3.encode_prompt(enc16, MAX_PROMPT_TOKENS)?;
        let mut prompt = if dec16.len() == enc16.len() {
            t3_prompt.clone()
        } else {
            self.s3.encode_prompt(dec16, MAX_PROMPT_TOKENS)?
        };
        let mel = prompt_mel_24k(dec24);
        let frames = mel.len() / 80;
        let n = prompt.len().min(frames / 2);
        if n == 0 {
            bail!("reference produced no aligned prompt tokens");
        }
        prompt.truncate(n);
        let prompt_mel = &mel[..n * 2 * 80];
        let speaker_t3 = self.ve.embed_wav(wav_16k, 16_000)?;
        let speaker_flow = self.camp.embed(dec16)?;
        let generated = self.t3.generate_seeded(
            &T3Context {
                speaker: speaker_t3,
                prompt_speech: t3_prompt,
                text: text.to_vec(),
            },
            &TurboDecode {
                max_gen_len,
                ..TurboDecode::default()
            },
            max_gen_len,
            seed,
        )?;
        let mut speech: Vec<u32> = generated
            .into_iter()
            .filter(|id| *id < SPEECH_VOCAB)
            .collect();
        speech.extend(std::iter::repeat(S3GEN_SIL).take(3));
        let total = (prompt.len() + speech.len()) * 2;
        let gen = speech.len() * 2;
        let mut rng = SplitMix::new(seed);
        let noise_full = rng.fill(80 * total);
        let noise_tail = rng.fill(80 * gen);
        let step = self.flow.mel_from_tokens(
            &prompt,
            &speech,
            prompt_mel,
            &speaker_flow,
            &noise_full,
            &noise_tail,
        )?;
        if step.mel.iter().any(|v| !v.is_finite()) {
            bail!("mean-flow mel is not finite");
        }
        let samples = self.hift.vocode(&step.mel, seed)?;
        Ok(CloneWave {
            samples,
            sample_rate: CLONE_SR,
            prompt_tokens: prompt,
            speech_tokens: speech,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn f32s(name: &str) -> Vec<f32> {
        let bytes = std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("testdata")
                .join(name),
        )
        .unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn env_path(key: &str) -> Option<PathBuf> {
        std::env::var_os(key)
            .map(PathBuf::from)
            .filter(|p| p.exists())
    }

    #[test]
    #[ignore = "needs PYRE_CHATTERBOX_S3, T3, VE, and TOKENIZER; writes testdata/clone_hello.wav"]
    fn cloned_wav_depends_on_text_and_reference() {
        let Some(s3) = env_path("PYRE_CHATTERBOX_S3") else {
            eprintln!("PYRE_CHATTERBOX_S3 unset");
            return;
        };
        let Some(t3) = env_path("PYRE_CHATTERBOX_T3") else {
            eprintln!("PYRE_CHATTERBOX_T3 unset");
            return;
        };
        let Some(ve) = env_path("PYRE_CHATTERBOX_VE") else {
            eprintln!("PYRE_CHATTERBOX_VE unset");
            return;
        };
        let Some(tok_dir) = env_path("PYRE_CHATTERBOX_TOKENIZER") else {
            eprintln!("PYRE_CHATTERBOX_TOKENIZER unset");
            return;
        };
        let stack = NanoStack::load(&NanoPaths {
            s3: &s3,
            t3: &t3,
            ve: &ve,
        })
        .unwrap();
        let tok = crate::tokenizer::NanoTokenizer::load(&tok_dir).unwrap();
        let hello = tok.encode_like_generate("Hello.").unwrap().1;
        let bye = tok.encode_like_generate("Goodbye.").unwrap().1;
        let wav16 = f32s("ldc_half_16k.f32");
        let wav24 = f32s("ldc_half_24k.f32");
        // Seed 4 is a fixed SplitMix draw. On this clip the Turbo categorical
        // sample for "Hello." and "Goodbye." diverges; seed 1 happens to land
        // on the same first id and then the stop token. The noise is still a
        // normal, not a zero tensor.
        let a = stack.cloned_waveform(&wav16, &wav24, &hello, 6, 4).unwrap();
        let again = stack.cloned_waveform(&wav16, &wav24, &hello, 6, 4).unwrap();
        assert_eq!(a.samples, again.samples);
        assert_eq!(a.sample_rate, 24_000);
        assert!(a.samples.iter().all(|s| s.is_finite()));
        assert_eq!(a.samples.len() % 480, 0);
        assert!(a.samples.iter().any(|s| s.abs() > 1e-4));
        // The three silence tokens are the tail. Text has to have moved the
        // ids in front of them.
        assert!(a.speech_tokens.len() > 3, "{:?}", a.speech_tokens);
        assert!(a.speech_tokens[..a.speech_tokens.len() - 3]
            .iter()
            .any(|id| *id != S3GEN_SIL));
        let b = stack.cloned_waveform(&wav16, &wav24, &bye, 6, 4).unwrap();
        assert_ne!(a.speech_tokens, b.speech_tokens);
        assert_ne!(a.samples, b.samples);
        let tone16 = f32s("camp_tone.f32");
        let mut tone24 = vec![0f32; 9600];
        for (i, s) in tone24.iter_mut().enumerate() {
            *s = 0.2 * (2.0 * std::f32::consts::PI * 140.0 * i as f32 / 24_000.0).sin();
        }
        let c = stack
            .cloned_waveform(&tone16, &tone24, &hello, 6, 4)
            .unwrap();
        assert_ne!(a.prompt_tokens, c.prompt_tokens);
        assert_ne!(a.samples, c.samples);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("clone_hello.wav");
        std::fs::write(&path, pcm16_wav(&a.samples, 24_000)).unwrap();
        eprintln!(
            "clone wav {} samples, prompt {:?}, speech {:?}",
            a.samples.len(),
            a.prompt_tokens,
            a.speech_tokens
        );
    }

    fn pcm16_wav(samples: &[f32], rate: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(44 + samples.len() * 2);
        let data = (samples.len() * 2) as u32;
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * 2).to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data.to_le_bytes());
        for x in samples {
            v.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        v
    }
}
