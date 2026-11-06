# Voice cloning engine choice

**Model:** [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano) (MIT)

**Why:** Smallest cloning stack with official inline bracket directions that can run on CPU beside Qwen-Next on a 12 GB RTX 3060. Official: ~3× realtime on 8 CPU cores. Full Nano files ~1.9 GB (T3 ~870 MB + mean-flow HiFT ~1.06 GB + ve 5.5 MB).

**Native direction syntax:** GPT-2 added tokens 50257–50275 — event tags `[laugh]` `[chuckle]` `[sigh]` `[cough]` … and delivery tags `[angry]` `[whispering]` `[happy]` … See `crates/pyre-voice-clone` and `GET /v1/audio/directions`.

**Portable timing (pyre):** SSML `<break time="…"/>` and `<prosody rate="…">` applied on every engine, including built-in MMS.

**Status:** Speaker encoder, GPT-2 BPE (`encode_script`), T3-Nano speech-token sampling, the S3 reference tokenizer, and the HiFT vocoder (mel to 24 kHz) run in `crates/pyre-voice-clone`. Mean-flow (tokens to mel) is not implemented, so HiFT is not yet fed by T3 or S3. Server `/v1/audio/speech/preview` exposes Nano single-token ids when the tokenizer is discoverable. `text_to_cloned_speech_ready()` stays false. MMS still owns production TTS. A HiFT reconstruction of a reference mel is not a cloned voice. Do not claim cloned audio.
