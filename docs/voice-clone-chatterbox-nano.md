# Voice cloning engine choice

**Model:** [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano) (MIT)

**Why:** Smallest cloning stack with official inline bracket directions that can run on CPU beside Qwen-Next on a 12 GB RTX 3060. Official: ~3× realtime on 8 CPU cores. Full Nano files ~1.9 GB (T3 ~870 MB + mean-flow HiFT ~1.06 GB + ve 5.5 MB).

**Native direction syntax:** GPT-2 added tokens 50257–50275 — event tags `[laugh]` `[chuckle]` `[sigh]` `[cough]` … and delivery tags `[angry]` `[whispering]` `[happy]` … See `crates/pyre-voice-clone` and `GET /v1/audio/directions`.

**Portable timing (pyre):** SSML `<break time="…"/>` and `<prosody rate="…">` applied on every engine, including built-in MMS.

**Status:** Speaker encoder + direction parser run in Burn (`crates/pyre-voice-clone`). `text_to_cloned_speech_ready()` is false until GPT-2 BPE → T3-Nano → S3Tokenizer → MeanFlow+HiFT land. Built-in MMS remains the speaking path. Voice identities can already be saved via `/v1/audio/voices`.
