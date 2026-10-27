# pyre

A llama.cpp `llama-server`-style inference server written in Rust where **[Burn](https://github.com/tracel-ai/burn)** (one version for everything: [Lyamc/burn](https://github.com/Lyamc/burn) @ `9db7598`, 0.23.0-pre.1) hosts and runs the model. No llama.cpp, candle or ggml does any of the compute: every matmul, attention and norm is a Burn tensor op on CUDA (CubeCL), wgpu (Vulkan/DX12/Metal) or the CPU (Flex / NdArray).

## Build

```powershell
cargo build --release            # cuda + wgpu + cpu(flex) + ndarray
cargo build --release --no-default-features --features wgpu   # smaller build
cargo test --release
```

* Toolchain: `rust-toolchain.toml` pins `stable-x86_64-pc-windows-msvc`, and `.cargo/config.toml` pins the MSVC `link.exe` linker. This overrides the user-level `~/.cargo/config.toml` (which sets `rust-lld` for msvc and MinGW gcc for gnu) and the `-gnu` default host. **Build scripts must be compiled for the msvc host**: with a `-gnu` host, `tracel-llvm-bundler` (used by `burn-cuda`) emits `-lstdc++`, which fails to link.
* CUDA: the `cuda` backend JIT-compiles kernels through CubeCL's bundled LLVM/NVPTX, so you don't need `nvcc` at build time. You need an NVIDIA driver at runtime.

## Run

```powershell
pyre serve -m Qwen/Qwen2.5-0.5B-Instruct -b cuda --port 8080
pyre serve -m qwen -b wgpu --api-key secret          # alias from the pyre store
pyre serve -m C:\models\SmolLM2-360M-Instruct -b cpu  # plain folder
pyre serve -m Qwen/Qwen3-0.6B --prompt "hi"           # one-shot greedy run + tok/s, then exit
```

| flag | meaning |
|---|---|
| `-m, --model` | alias, store name, HF repo id `org/name[@rev]`, or a folder |
| `-b, --backend` | `cuda` \| `wgpu` \| `vulkan` (feature) \| `cpu` (Flex) \| `ndarray` |
| `--dtype` | `auto` (bf16 on CUDA for bf16 checkpoints, f16 on wgpu, f32 on CPU) \| `f32` \| `f16` \| `bf16` |
| `-c, --ctx-size` | context / KV-cache length (default 4096, capped at `max_position_embeddings`) |
| `-n, --n-predict` | default max tokens |
| `--api-key` / `PYRE_API_KEY` | require `Authorization: Bearer <key>` (everything except `/health`) |
| `--host`, `--port`, `-a/--alias` | the same as llama-server |

## Storage

pyre **reuses the standard Hugging Face hub cache in place**, so a model you already downloaded with `huggingface-cli`, Python `transformers`/`huggingface_hub` or another tool **costs no extra disk**, and pyre's downloads are readable by those tools.

* HF cache location: `--hf-cache` > `HF_HUB_CACHE` > `$HF_HOME/hub` > `$XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub` (on Windows `%USERPROFILE%\.cache\huggingface\hub`). The layout is `models--org--name/{blobs,refs,snapshots/<commit>}`. Snapshot entries may be symlinks into `blobs/` (Linux/macOS, or Windows with Developer Mode) or plain files/hardlinks (Windows without symlink rights); pyre handles both.
* Downloads go through `hf-hub` 1.0 into that same cache. On Windows without symlinks, hf-hub 1.0 writes *both* a blob and a snapshot **copy**, doubling disk use. pyre then replaces each snapshot copy with a **hardlink** to its blob, which saves the space and stays compatible.
* `HF_TOKEN` (or the `huggingface-cli login` token file) is used by hf-hub for gated repos and is never printed. `HF_HUB_OFFLINE=1` or `--offline` disables all network access.
* **pyre store** (`--models-dir` > `PYRE_MODELS_DIR` > `models_dir` in the config file `%APPDATA%\pyre\config.toml` / `~/.config/pyre/config.toml` > `%LOCALAPPDATA%\pyre\models` / `~/.local/share/pyre/models`). The layout is `<kind>/<org>/<name>/<rev>/` plus `pyre-model.toml` (source repo/revision, kind, architecture, dtype, files with sizes and sha256, import mode, optional alias).
  * `import --link` (default) hardlinks the files, so they take **no extra space**. If hardlinks fail (different volume) it falls back to symlinks, then to copying, and it tells you which it used.
  * `--copy` copies the files. `--move` takes them out of the cache. `--register` records a local folder in place and copies nothing.
* Search order for `-m`: explicit path → store alias/name → HF cache → download (unless offline). Change it with `--search-order path,store,hf,download` or `search_order = [...]` in the config file.

```text
pyre models list [--json]
pyre models pull <repo>[@rev] [--store hf|pyre]      # hf = cache only (default); pyre = also link into store
pyre models import <repo|path> [--link|--copy|--move|--register] [--name N] [--kind llm|image|stt|tts]
pyre models rm <name> [--from-hf-cache] [-y]         # never deletes HF blobs unless --from-hf-cache (asks first)
pyre models alias <name> <alias>
pyre models info <name>
```

## Speech to text (whisburn)

STT is backed by [whisburn](file:///C:/Build/machine-learning/whisburn), Lyam's Burn speech project. pyre
uses it unmodified through a thin adapter (`src/stt.rs`).

- **Code:** whisburn @ `d27f1fc` is vendored as `crates/pyre-whisper-{core,audio,engine,models}` and
  ported to the Lyamc/burn device-dispatch API (no backend generics). The whisburn repo itself is untouched.
- **One Burn:** `cargo tree` shows a single burn (0.23.0-pre.1, Lyamc/burn#9db7598) and a single cubecl
  (4fdcb6e) for LLM, STT and SD. Whisper weights are re-converted from HF into burnpack (`model.bpk`);
  the published 0.21 `.mpk` bundles are not readable by this Burn and are skipped.
- **Formats:** wav/mp3/flac/aac/m4a, Vorbis ogg, and Opus in webm/ogg (browser mic recordings) via a
  pure-Rust Opus decoder (`opus-rs`) registered into symphonia. Verified: `samples/speech.webm` and
  `speech.ogg` (ffmpeg libopus) transcribe the same as the wav.
- **Backends:** whisburn has wgpu (any GPU, including NVIDIA through Vulkan/DX12) and ndarray (CPU).
  `--backend cuda`/`wgpu` both map to whisburn's wgpu; `cpu`/`flex`/`ndarray` map to its CPU runtime.
- **Models:** any whisburn ASR registry name (`tiny`, `tiny_en`, `base`, `small_en`, `large-v3-turbo`,
  `distil-large-v3`, `parakeet-tdt-0.6b-v3`, `moonshine-tiny`, `qwen3-asr-0.6b`, ...), its HF repo id
  (`openai/whisper-tiny.en`), or a path to an existing whisburn bundle directory.
  Bundles are whisburn's converted format (`model.mpk` + tokenizer + config), not raw HF safetensors.
  They live in `WHISBURN_MODELS_DIR` if set, else `<pyre store>/stt/whisburn/<name>`. A missing bundle
  is fetched by whisburn's own downloader (its published bundle from `lyamc/whisburn`, or a conversion
  from upstream HF). `--offline` disables this. Because `WHISBURN_MODELS_DIR` is process-global, all
  STT models in one pyre process share one bundle dir.

```
pyre serve -m qwen --stt-model tiny_en
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@speech.wav -F model=whisper-1 -F response_format=verbose_json
```

`response_format`: `json` (default), `text`, `srt`, `vtt`, `verbose_json` (segments + duration + language).
`/v1/audio/translations` runs whisburn's translate task. `language` is honoured; `prompt`, `temperature`
and `timestamp_granularities` are accepted and ignored. Input is decoded by whisburn-audio (symphonia:
wav/mp3/flac/ogg-vorbis/m4a-aac). Browser mic recordings (webm/opus) may not decode.

Measured on an RTX 3060 12 GB (512x512, 30 steps default, f32):

| | before (burn 0.20 diffusers-burn) | now (Lyamc/burn, persistent cache + warm-up) |
|---|---|---|
| CUDA first image after startup | ~9 min | **14.1 s** (startup incl. warm-up ~4 min on a cold cache, ~28 s warm) |
| CUDA steady state | ~9.4 s (20 steps) | 13.7 s (30 steps) |
| wgpu first / steady | n/a | 511 s / 521 s (startup 183 s) - wgpu conv kernels are far slower; use CUDA |

The CubeCL kernel/autotune cache is now persisted (`target/environment/default.db`, burn `persistence`
feature). On Windows this needed a local cubecl fix: its Turso cache requested multiprocess WAL, which
Turso's Windows IO backend rejects, so the cache silently fell back to memory every run. Fix is on a local
branch (`C:/Build/ai/cubecl-pyre-fix`, `pyre-windows-wal-fix`) wired in with `[patch]`; not pushed.
Load-time warm-up: `PYRE_SD_WARMUP=0` disables, `PYRE_SD_WARMUP_STEPS` (default 2).

## Text to speech (stub)

`/v1/audio/speech` returns **501** with a clear message, and `--tts-model` registers a placeholder.
The UI's TTS card stays disabled. I looked at these ports and didn't do any of them yet:

- **Kokoro-82M via burn-onnx** (burn 0.22, Apache weights): tracel's model check compiles the
  2,464-node graph (about 7 minutes of codegen), but its output still diverges from ONNX Runtime
  (Pearson r ≈ 0.69, peaks about 1.3× too large; burn-onnx#371). It also needs a phonemizer
  (espeak-ng is a system library and GPL, which would mean installing something; a pure-Rust G2P
  would be needed instead).
- **A hand port of Kokoro (StyleTTS2 + ISTFTNet)**: about 2–3k lines (BERT/ALBERT, LSTM, AdaIN
  decoders, iSTFT). It's feasible on Burn 0.22 but is a multi-day job.
- **Piper (VITS)**: also needs espeak-ng phonemes.

Plan: Kokoro through burn-onnx once #371 lands, plus a Rust G2P (misaki-style lexicon), behind a
`tts` feature, emitting 24 kHz wav/mp3/opus.

## Desktop GUI (GPUI)

`gui/` is a separate workspace (`pyre-gui`) on [Lyamc/gpui](https://github.com/Lyamc/gpui) @ `08629d4`.
It talks to the server over HTTP only and does not link Burn. Pages: Chat (SSE streaming), Models
(load/unload/pull/import/alias), Images, Audio (transcribe a file, TTS), Status.

```powershell
cargo build --release --manifest-path gui/Cargo.toml
gui\target\x86_64-pc-windows-msvc\release\pyre-gui.exe http://127.0.0.1:8080
```

Screenshots: `samples/gui-{chat,models,images,audio,status}.png`.

## Roadmap / known gaps

- GGUF / quantized weights (Q8_0, Q4_K) - not done. Options: port Lyamc/burn-gguf (targets upstream
  burn 0.22, needs porting to the fork's dispatch API) or a pyre-side GGUF reader that dequantizes to
  f16/bf16 at load (memory-saving only, no quantized matmul).
- Continuous batching - not done; each model has one worker and a FIFO queue. Needs batched KV cache
  (per-sequence positions + padding masks) in `src/model/llama.rs` and a scheduler in the registry.
- MoE (Qwen3-MoE) and Gemma (2/3) - not done. Gemma needs: GeGLU, (1+w) RMSNorm, embedding scaling,
  logit soft-capping and sliding-window layers; Qwen3-MoE needs router top-k + expert dispatch.
- TTS (see above) - still a 501 stub.
- wgpu Stable Diffusion is ~40x slower than CUDA (see table).
- LLM parity: Qwen2.5-0.5B greedy, 15 tokens, matches HF f32 on the first 14; token 15 is a near tie
  (`.` vs `.\n`), identical to the pre-port build. Decode (0.5B, RTX 3060): CUDA 43.3 tok/s bf16 /
  38.2 f32 (was 30.5), wgpu 41.9 tok/s (was 38.4).
