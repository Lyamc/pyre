//! What this crate runs today, and the stages still required before
//! text becomes cloned speech. Nothing here claims a waveform.

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
        weights: "added_tokens.json token ids 50257–50275, hardcoded. vocab.json and merges.txt are not read yet.",
        reuses: "Nothing from MMS. MMS is a character vocab with no bracket tags. Qwen's chat template is a different tokenizer.",
        missing: "GPT-2 BPE (vocab 50276, bos off, pad = eos). Tags must stay single ids. Nano generate() ignores exaggeration and cfg_weight, so those are not a syntax.",
    },
    Stage {
        id: "t3-nano",
        state: StageState::NotStarted,
        weights: "t3_nano_v1.safetensors (~870 MB). GPT-2 small (`llama_config_name = GPT2_small`), text vocab 50276, speech vocab 6563, speech_cond_prompt_len 375, no perceiver, emotion_adv off.",
        reuses: "Qwen and Whisper already have causal self-attention, RMS/LayerNorm, and a sampler. The graph is not theirs: a GPT-2 backbone with a speech-token head and Turbo decoding (temperature 0.8, top-k 1000, top-p 0.95, repetition penalty 1.2, cfg 0).",
        missing: "GPT-2 weight load, dual text/speech embeddings, the T3 conditioning path (speaker embedding + S3 prompt tokens), and the Turbo token loop. Output is speech-token ids, not audio.",
    },
    Stage {
        id: "s3-tokenizer",
        state: StageState::NotStarted,
        weights: "Inside s3gen_meanflow.safetensors. 16 kHz reference audio, first 15 s, becomes up to 375 prompt tokens at 25 Hz.",
        reuses: "Not the Whisper encoder and not the VibeVoice acoustic encoder. S3Tokenizer is the CosyVoice speech tokenizer (convolutional encoder, vector quantizer). Sharing the Whisper graph would not load these weights.",
        missing: "The tokenizer module, the 6561-code vocabulary (ids below 6561 are kept; 6561 is silence padding), and the prompt-token length crop.",
    },
    Stage {
        id: "meanflow-hift",
        state: StageState::NotStarted,
        weights: "s3gen_meanflow.safetensors (~1.06 GB). `S3Gen(meanflow=True)`; generate() asks for `n_cfm_timesteps=2`. Waveform is 24 kHz. `conds.pt` is an optional built-in voice, not required for cloning.",
        reuses: "src/tts.rs already runs a VITS residual-coupling flow and a HiFi-GAN decoder. That vocoder is the wrong one: S3Gen's decoder is HiFT (harmonic source-filter plus iSTFT), conditioned on a speaker embedding and a reference mel. conv1d and transposed conv exist. Snake, harmonic excitation, and iSTFT do not.",
        missing: "Mean-flow estimator (distilled flow matching, 1–2 steps), reference mel embed (`embed_ref`, 10 s at 24 kHz), HiFT generator, 24 kHz framing. This is the stage that produces samples.",
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
        state: StageState::NotStarted,
        weights: "A directory of the Nano files above, pointed at by a future store entry. Do not replace the MMS `TtsModel` until text produces samples.",
        reuses: "The registry already has a Tts kind. Leave facebook/mms-tts on that path.",
        missing: "A clone request shape: reference audio plus script. CPU residency so the 12 GB 3060 stays with Qwen-Next. Full Nano weights are about 1.9 GB and the project quotes about 3× realtime on 8 CPU cores once every stage exists.",
    },
];

pub fn stages() -> &'static [Stage] {
    STAGES
}

pub fn text_to_cloned_speech_ready() -> bool {
    stages().iter().all(|s| s.state == StageState::Running)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_runs_and_waveform_does_not() {
        assert!(stages()
            .iter()
            .any(|s| s.id == "speaker-encoder" && s.state == StageState::Running));
        assert!(!text_to_cloned_speech_ready());
        assert!(stages()
            .iter()
            .any(|s| s.id == "meanflow-hift" && s.state == StageState::NotStarted));
    }
}
