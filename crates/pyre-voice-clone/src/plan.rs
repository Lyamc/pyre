//! What this crate runs today, and the stages still required before
//! text becomes cloned speech. Mean-flow feeds HiFT from S3 and T3 tokens.
//! [`text_to_cloned_speech_ready`] stays false until that waveform has been
//! auditioned. A passing numeric test is not that audition.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageState {
    /// Runs in this crate, on CPU, with real or synthetic inputs.
    Running,
    /// Specified, not implemented.
    NotStarted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    pub id: &'static str,
    pub state: StageState,
    /// Hugging Face file under `ResembleAI/chatterbox-nano`. Empty if none.
    pub weights: &'static str,
    /// Pyre code that is the same kind of op, and why it is or is not a drop-in.
    pub reuses: &'static str,
    /// Ops or graphs Burn does not already have for this stage.
    pub missing: &'static str,
}

const STAGES: &[Stage] = &[
    Stage {
        id: "speaker-encoder",
        state: StageState::Running,
        weights: "ve.safetensors (5.5 MB, f32, MIT). Three-layer LSTM 40→256, linear 256, ReLU, L2 norm.",
        reuses: "Slaney mel and periodic Hann are the same family as pyre-whisper-engine audio::mel / audio::stft. Those use Whisper's 80-bin hop-160 filterbank, so the 40-bin n_fft=400 contract is reimplemented here. No Whisper weights are loaded.",
        missing: "Nothing for this stage. Resampling other than 16 kHz is linear; upstream is librosa kaiser_fast. GE2E similarity_weight/bias are loaded and unused, matching inference.",
    },
    Stage {
        id: "inline-directions",
        state: StageState::Running,
        weights: "vocab.json, merges.txt, added_tokens.json, tokenizer_config.json. About 1.4 MB. Not the 1.9 GB weight set.",
        reuses: "The same `tokenizers` GPT-2 byte-level BPE pyre already uses for Qwen3. This crate does not call the Whisper tokenizer, which adds Whisper special tokens.",
        missing: "Nothing for text ids. `encode_like_generate` is punc_norm then BPE and matches HF, including leaving [CLEAR THROAT] split. `encode_script` casefolds a known tag onto its single id. The server's MMS markup is a different parser: it turns some brackets into silence or rate and does not emit these ids. Nano generate() ignores exaggeration and cfg_weight.",
    },
    Stage {
        id: "t3-nano",
        state: StageState::Running,
        weights: "t3_nano_v1.safetensors (~870 MB). GPT-2 small (`llama_config_name = GPT2_small`), text vocab 50276, speech vocab 6563, speech_cond_prompt_len 375, no perceiver, emotion_adv off.",
        reuses: "Burn matmul, softmax, and layer-norm on the flex CPU device. Qwen and Whisper attention are a different graph (RoPE, GQA, SiLU), so the GPT-2 block is its own.",
        missing: "Nothing for speech-token ids. `gelu_new`, absolute `wpe` on the concatenated cond/text/speech embeds, KV cache, and the Turbo logit warpers run. `generate_seeded` draws one categorical sample after those warpers. Greedy argmax on a short clip can ignore the script even when the logits move.",
    },
    Stage {
        id: "s3-tokenizer",
        state: StageState::Running,
        weights: "tokenizer.* inside s3gen_meanflow.safetensors. 16 kHz reference, first 15 s, at most 375 tokens at 25 Hz. The file also holds HiFT; this stage loads only the tokenizer tensors.",
        reuses: "The voice-encoder mel uses the same periodic Hann and Slaney scale, but 40 bins. S3 is 128 bins plus Whisper-style log compression, then a 6-layer FSMN encoder. It is not the Whisper encoder.",
        missing: "Nothing for prompt-token ids. Finite scalar quantization (3^8 = 6561). A speaker-encoder cosine on a synthetic harmonic is not evidence these ids are a voice.",
    },
    Stage {
        id: "hift",
        state: StageState::Running,
        weights: "mel2wav.* inside s3gen_meanflow.safetensors. HiFTGenerator: upsample [8, 5, 3], iSTFT n_fft 16 hop 4, 24 kHz, 8 harmonics.",
        reuses: "Burn is not required for the 16-point FFT. Conv, Snake, and iSTFT are local. The VITS HiFi-GAN in src/tts.rs is a different vocoder and is not loaded.",
        missing: "Nothing for mel-to-wave given an excitation. The harmonic source is seeded here (SplitMix64); it is not torch's RNG. A reconstruction of a reference mel is not a cloned voice.",
    },
    Stage {
        id: "meanflow",
        state: StageState::Running,
        weights: "flow.* and speaker_encoder.* (CAMPPlus, 192-d) in the same file. S3Gen(meanflow=True), n_cfm_timesteps=2, no CFG. Prompt mel is 24 kHz, 80 bins, hop 480, two frames per S3 token.",
        reuses: "HiFT consumes the generated mel tail. T3 and S3 emit the token ids. The voice encoder's 256-d vector is only for T3. CAMPPlus is a separate 192-d embedding from a Kaldi fbank (dither 0).",
        missing: "Nothing for the two-step Euler mel on a batch-1 clip. Prompt and speech noise are separate normals; a seed replays them and does not zero them. Audio longer than the 30 s S3 window is not implemented. The server resamples with linear interpolation; the numeric test uses a librosa kaiser pair.",
    },
    Stage {
        id: "watermark",
        state: StageState::NotStarted,
        weights: "None in the checkpoint. `perth.PerthImplicitWatermarker` runs after the vocoder.",
        reuses: "No pyre audio watermark.",
        missing: "Optional. Audio can be returned without it. Upstream always applies it.",
    },
    Stage {
        id: "server-wireup",
        state: StageState::Running,
        weights: "Discovered the same way as the crate: PYRE_CHATTERBOX_S3, PYRE_CHATTERBOX_T3, PYRE_CHATTERBOX_VE, PYRE_CHATTERBOX_TOKENIZER, then PYRE_CHATTERBOX_NANO, then the HF hub cache. Nothing is downloaded.",
        reuses: "POST /v1/audio/speech. voice_id mms is still the MMS VITS path, including SSML breaks. A saved voice_id loads reference.wav and runs NanoStack, cached after the first call.",
        missing: "text_to_cloned_speech_ready stays false. The list endpoint still marks cloned voices as not synthesizable so the picker does not advertise them. Watermark is not applied. The first request loads about 1.9 GB on CPU.",
    },
];

pub fn stages() -> &'static [Stage] {
    STAGES
}

/// Cloned speech is a Burn waveform from text plus a reference clip.
///
/// This stays `false` until a person has listened to a waveform from this
/// path and accepted it. It is not derived from [`stages`]. A HiFT
/// reconstruction of a reference mel is not a cloned voice, and neither is
/// `testdata/clone_hello.wav` until that audition. MMS bracket markup is still
/// a silence/rate approximation, not these token ids.
pub fn text_to_cloned_speech_ready() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_runs_and_waveform_does_not() {
        assert!(stages()
            .iter()
            .any(|s| s.id == "speaker-encoder" && s.state == StageState::Running));
        assert!(stages()
            .iter()
            .any(|s| s.id == "inline-directions" && s.state == StageState::Running));
        assert!(stages()
            .iter()
            .any(|s| s.id == "t3-nano" && s.state == StageState::Running));
        assert!(stages()
            .iter()
            .any(|s| s.id == "s3-tokenizer" && s.state == StageState::Running));
        // Tokenizer, T3, and S3 ids still do not mean cloned audio.
        assert!(!text_to_cloned_speech_ready());
        assert!(stages()
            .iter()
            .any(|s| s.id == "hift" && s.state == StageState::Running));
        assert!(stages()
            .iter()
            .any(|s| s.id == "meanflow" && s.state == StageState::Running));
        assert!(stages()
            .iter()
            .any(|s| s.id == "server-wireup" && s.state == StageState::Running));
    }
}
