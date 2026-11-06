# Voice cloning engine choice

**Model:** [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano) (MIT)

**Why:** Smallest cloning stack with official inline bracket directions that can run on CPU beside Qwen-Next on a 12 GB RTX 3060. Official: ~3× realtime on 8 CPU cores. Full Nano files ~1.9 GB (T3 ~870 MB + mean-flow HiFT ~1.06 GB + ve 5.5 MB).

**Native direction syntax:** GPT-2 added tokens 50257–50275 — event tags `[laugh]` `[chuckle]` `[sigh]` `[cough]` … and delivery tags `[angry]` `[whispering]` `[happy]` … See `crates/pyre-voice-clone` and `GET /v1/audio/directions`.

**Portable timing (pyre):** SSML `<break time="…"/>` and `<prosody rate="…">` applied on every engine, including built-in MMS.

**Status:** Speaker encoder, GPT-2 BPE, T3-Nano (greedy and seeded), S3 prompt tokens, CAMPPlus, the 24 kHz prompt mel, mean-flow (two Euler steps, no CFG), and HiFT run in `crates/pyre-voice-clone`. `NanoStack::cloned_waveform` chains them. `testdata/clone_hello.wav` is that path on a half-second of LDC93S1 with the script "Hello." and seed 4. It is a numeric result, not an audition. `text_to_cloned_speech_ready()` stays false. `POST /v1/audio/speech` with `voice_id=mms` is still MMS. A saved `voice_id` runs the native stack when the checkpoints are on disk; the voice list does not advertise that. The voice encoder (256-d) conditions T3. CAMPPlus (192-d, Kaldi fbank, dither 0) conditions the flow. They are different embeddings.
