//! Chatterbox-Nano voice cloning, native Burn. This is the speaker-encoder
//! slice, not text-to-speech.
//!
//! Chosen model: [Chatterbox-Nano](https://huggingface.co/ResembleAI/chatterbox-nano)
//! (Resemble AI, MIT). About 110M parameters in the GPT-2 text-to-token model,
//! plus a shared ~1 GB mean-flow HiFT decoder. It clones from a reference clip
//! and takes inline vocal directions as bracket tokens (`[laugh]`, `[sigh]`, …).
//! Official numbers: about 3× realtime on 8 CPU cores, about 10× on a GPU.
//! That CPU path is the one that can sit beside Qwen-Next on a 12 GB RTX 3060.
//! The encoder runs on Burn's flex CPU device, the same CPU backend pyre uses
//! when `--backend cpu` is selected.
//!
//! Why not the other candidates, in short:
//! * Qwen3-TTS 0.6B Base clones, Apache-2.0, but natural-language voice design
//!   is the 1.7B checkpoint. The 12 Hz stack is a Qwen LM plus a 16-codebook
//!   tokenizer and a causal ConvNet. Too wide to keep resident next to Qwen-Next,
//!   and the dual-track sampler is a larger port than Nano's first stage.
//! * VoxCPM2 is 2B, Apache-2.0, and does voice design, at ~8 GB. It does not fit
//!   beside Qwen-Next.
//! * OpenVoice V2 is the smallest converter and it reuses the VITS flow and
//!   HiFi-GAN already in `src/tts.rs`. It does not speak inline directions. The
//!   base voice would also have to be MeloTTS, not the single-speaker MMS model.
//! * XTTS v2 is non-commercial (Coqui Public Model License).
//!
//! What runs here: the voice encoder on real `ve.safetensors`, clone cosine, the
//! inline-tag parser, and the GPT-2 BPE tokenizer (vocab 50276, tags as one id).
//! T3-Nano's GPT-2 forward ([`t3_model`]) turns text plus a speaker vector and
//! optional prompt-speech ids into speech-token ids. [`s3`] turns a 16 kHz
//! reference into those prompt ids (25 Hz, at most 375). [`hift`] turns an
//! 80-bin 24 kHz mel into samples. Mean-flow, which would make that mel from
//! the tokens, is not implemented, so this is still not text to a waveform.
//! See [`plan`].

pub mod directions;
pub mod encoder;
pub mod hift;
pub mod mel;
pub mod plan;
pub mod s3;
pub mod t3;
pub mod t3_model;
pub mod tokenizer;
pub mod verify;

pub use directions::{parse_script, Direction, DirectionKind, Script};
pub use encoder::VoiceEncoder;
pub use hift::Hift;
pub use plan::{stages, text_to_cloned_speech_ready, Stage, StageState};
pub use s3::{discover_s3, S3Tokenizer, MAX_PROMPT_TOKENS};
pub use t3::{check_checkpoint, TurboDecode};
pub use t3_model::{discover_t3, T3Nano};
pub use tokenizer::{discover_tokenizer_dir, NanoTokenizer};
pub use verify::{match_speakers, SpeakerMatch, CLONE_COSINE_FLOOR};
