//! Switches used to ablate padding, the decoder, and the window schedule.
//! Production zero-pads the log-mel to one encoder context. The decoder KV
//! cache is the beam search: on real tiny weights it decoded the same text as
//! the eager full-sequence beam (LDC93S1, beam 1 and beam 5). Windows stop
//! once a window already covers the file, so a 12 s clip is not decoded twice.
//! Truncating the mel tail does not match. Tests wrap a call in [`with_whisper_pass`].

use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelTail {
    /// Zero-fill log-mel frames up to `n_audio_ctx * 2`, then add absolute positions.
    ZeroPad,
    /// Feed only the real frames. This drops the positional tail and changes text.
    Truncate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderKind {
    /// Full `forward_decoder` on every beam step (the previous production path).
    Eager,
    /// KV cache. Kept only when real-weight token sequences match [`DecoderKind::Eager`].
    Cached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Windowing {
    /// `(len - 1) / shift + 1` starts. This is what tiny/base used before this branch.
    Legacy,
    /// No window whose start is already inside a previous window that reached the end.
    Covering,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhisperPass {
    pub tail: MelTail,
    pub decoder: DecoderKind,
    pub windows: Windowing,
}

impl WhisperPass {
    pub const fn historical() -> Self {
        Self {
            tail: MelTail::ZeroPad,
            decoder: DecoderKind::Cached,
            windows: Windowing::Covering,
        }
    }
}

thread_local! {
    static PASS: Cell<WhisperPass> = const { Cell::new(WhisperPass::historical()) };
}

pub fn current() -> WhisperPass {
    PASS.with(|p| p.get())
}

pub fn with_whisper_pass<R>(pass: WhisperPass, f: impl FnOnce() -> R) -> R {
    PASS.with(|slot| {
        let prev = slot.replace(pass);
        let out = f();
        slot.set(prev);
        out
    })
}
