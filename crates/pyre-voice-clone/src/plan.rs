//! What this crate runs today, and the stages still required before
//! text becomes cloned speech. HiFT can turn a mel into samples.
//! [`text_to_cloned_speech_ready`] stays false until mean-flow feeds that mel
//! from S3 and T3 tokens and the result is a real clone, not a reconstruction.

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
        missing: "Nothing for speech-token ids. `gelu_new`, absolute `wpe` on the concatenated cond/text/speech embeds, KV cache, and the Turbo logit warpers run. Output is speech-token ids, not audio. S3 and HiFT are still required before a waveform.",
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
        missing: "Nothing for mel-to-wave given an excitation. The harmonic source is seeded here (SplitMix64); it is not torch's RNG. Mean-flow still has to produce the mel from S3 and T3 tokens. A reconstruction of a reference mel is not a cloned voice.",
    },
    Stage {
        id: "meanflow",
        state: StageState::NotStarted,
        weights: "flow.* and speaker_encoder.* (CAMPPlus, 192-d) in the same file. S3Gen(meanflow=True), n_cfm_timesteps=2, no CFG. Prompt mel is 24 kHz, 80 bins, hop 480, two frames per S3 token.",
        reuses: "HiFT consumes the mel. T3 and S3 already emit the token ids that are concatenated with the prompt tokens (drop ids >= 6561, then three silence tokens 4299).",
        missing: "UpsampleConformerEncoder (6 blocks, relative position), token embedding 6561x512, speaker affine 192 to 80, CausalConditionalCFM basic_euler over t and r with the diagonal time mixer, ConditionalDecoder UNet1D (channels 256, 4 transformer blocks down, 12 mid, 4 up, causal conv). Until this runs, HiFT is fed a mel, not speech tokens.",
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

/// Cloned speech is a Burn waveform from text plus a reference clip.
///
/// This stays `false` until that path exists. It is not derived from [`stages`]:
/// a running tokenizer, T3 or S3 speech-token ids, or the MMS bracket approximations
/// in `src/tts_markup.rs` do not make it true. Those approximations insert
/// silence or change rate; they are not Nano GPT-2 token ids and they are not
/// a cloned voice. A HiFT reconstruction of a reference mel is not one either.
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
            .any(|s| s.id == "meanflow" && s.state == StageState::NotStarted));
    }
}
