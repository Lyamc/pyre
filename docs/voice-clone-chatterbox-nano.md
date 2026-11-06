# Voice cloning engine choice

**Model:** [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano) (MIT)

**Why:** Smallest cloning stack with official inline bracket directions that can run on CPU beside Qwen-Next on a 12 GB RTX 3060. Official: ~3× realtime on 8 CPU cores. Full Nano files ~1.9 GB (T3 ~870 MB + mean-flow HiFT ~1.06 GB + ve 5.5 MB).

**Native direction syntax:** GPT-2 added tokens 50257–50275 — event tags `[laugh]` `[chuckle]` `[sigh]` `[cough]` … and delivery tags `[angry]` `[whispering]` `[happy]` … See `crates/pyre-voice-clone` and `GET /v1/audio/directions`.

**Portable timing (pyre):** SSML `<break time="…"/>` and `<prosody rate="…">` applied on every engine, including built-in MMS.

**Status:** Speaker encoder, GPT-2 BPE (`encode_script`), T3-Nano speech-token sampling, and the S3 reference tokenizer (25 Hz, at most 375 ids from the first 15 s) run in Burn (`crates/pyre-voice-clone`). Server `/v1/audio/speech/preview` exposes Nano single-token ids when the tokenizer is discoverable. `text_to_cloned_speech_ready()` stays false until mean-flow + HiFT emit a waveform. MMS still owns production TTS; SSML breaks/prosody and event-tag silence approximations apply there. A speaker-encoder cosine on a synthetic harmonic is not a voice. Do not claim cloned audio.
